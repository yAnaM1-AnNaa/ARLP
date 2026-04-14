"""
Region-specific similarity-projection dataset (minimal version).

Returned item dict:
    {
        "text"     : str,
        "text_emb" : FloatTensor(1024),
        "sim_proj" : FloatTensor(H, W),
        "rgb"      : FloatTensor(3, H, W)
    }
"""
import os, json, argparse, random
from typing import List, Tuple
import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Union, List

import torch.nn.functional as F
import torchvision.transforms as T

def transform_imgs(imgs, blur=True):
    """
    Transform image before passing to DINO model
    ::param imgs:: np.array of shape (H, W, C) or (bs, H, W, C)
    ::param blur:: bool, whether to apply Gaussian blur before resizing

    ::return:: list of transformed images
    """
    # handles both single image and batch of images
    if len(imgs.shape) == 3:
        H, W, C = imgs.shape
        imgs = imgs[None, ...]
        bs = 1
    else:
        bs, H, W, C = imgs.shape

    H *= 2
    W *= 2

    patch_h = H // 14
    patch_w = W // 14

    if blur:
        transform_lst = [T.GaussianBlur(9, sigma=(1.0, 2.0))]
    else:
        transform_lst = []
    transform_lst += [
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    transform = T.Compose(transform_lst)
    
    transformed_imgs = []
    for i in range(bs):
        temp = imgs[i].copy()
        if temp.max() <= 1.1: # handle images with values in [0, 1]
            temp = (temp * 255)
        temp = temp.astype(np.uint8).clip(0, 255)
        transformed_imgs.append(transform(Image.fromarray(temp)))
    
    return transformed_imgs

# --------------------------------------------------------------------------- #
#                                DATASET CLASS                                #
# --------------------------------------------------------------------------- #
class RegionSimDataset(Dataset):
    """
    Builds a flat in-memory list of samples from one or many .h5 files.

    Parameters
    ----------
    h5_paths    : str | list[str]
        Path(s) to .h5 files (each file contains many 'instances').
    db_path     : str | str
        The database contains thr following keys:
        - category name. eg.'chair'
        - instance name. eg.'chair_001'
        - frame idx. eg. '1', '10'...
        - clustered img path. eg.'clustered_imgs/chair_001_1.png'
        - status
        - vlm response. eg. 'a red chair with four legs'
        - error msg
    random_pad  : bool                  (default True)
        If True, applies translation-style augmentation by padding both RGB and
        sim with the *same* random margins, then resizing back to original size.
    thresh      : float                 (default 0.5)
        Shutdown threshold used before blurring sim maps.
    blur        : bool                  (default True)
        Whether to apply GaussianBlur(k=3, σ=1) after thresholding.
    embedding_type : str                (default "embeddings_oai")
        Which embedding group to use from the H5 file. Options:
        - "embeddings_oai": openai embeddings api
        - "embeddings_st": sentence transformer embeddings
    """
    # ................................................................. #
    def __init__(self,
                 h5_paths,
                 db_path,
                 *,
                 random_pad: bool = True,
                 thresh: float = 0.5,
                 blur: bool = True,
                 bg_dir: Union[str, List[str], None] = None,
                 embedding_type: str = "embeddings"):

        super().__init__()

        if isinstance(h5_paths, (str, os.PathLike)):
            h5_paths = [str(h5_paths)]

        self._rand_pad = random_pad
        self._thresh   = float(thresh)
        self._blur     = bool(blur)
        self._bg_dir   = bg_dir
        self._bg_bank  = None 
        self._embedding_size = None  # Will be set when first embedding is loaded

        # flat list: (text, emb, raw_sim, processed_rgb)
        self.samples: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for fp in tqdm(h5_paths):
            self._ingest_file(fp, embedding_type)

        if not self.samples:
            raise RuntimeError("No valid samples found in any provided .h5 files")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {len(h5_paths)} file(s) using {embedding_type}.")
        if self._embedding_size is not None:
            print(f"[Dataset] Embedding dimension: {self._embedding_size}")

    # ------------------------------------------------------------------ #
    #                        DATA INGESTION                               #
    # ------------------------------------------------------------------ #
    def _ingest_file(self, fp: str, embedding_type: str):
        """Read one .h5 file and cache everything as tensors."""
        with h5py.File(fp, "r") as f:
            for inst_key in f:
                grp = f[inst_key]

                needed = ["region_matching", "color_label_names",
                          "similarity_projections", embedding_type, "top_k_rgb"]
                
                if not all(k in grp for k in needed):
                    missing = [k for k in needed if k not in grp]
                    print(f"Skipping {inst_key}: missing {missing}")
                    continue                          # skip incomplete instance

                # ---- look-ups ------------------------------------------------ #
                color_names = [n.decode() if isinstance(n, bytes) else str(n)
                               for n in grp["color_label_names"][()]]
                color_to_idx = {c: i for i, c in enumerate(color_names)}

                region_map = json.loads(grp["region_matching"][()].decode("utf-8"))

                sim_all = grp["similarity_projections"][()]        # (C, 3, H, W)

                raw_rgbs = grp["top_k_rgb"][()]             # (k,H,W,3) uint8
                rgb_all  = transform_imgs(raw_rgbs)          # list or tensor
                if isinstance(rgb_all, list):
                    rgb_all = torch.stack(rgb_all, 0)        # (k,3,H',W')

                # ---------- build foreground masks ----------
                # target spatial size comes from the already-transformed RGB tensor
                H_px, W_px = rgb_all.shape[2:]                      # e.g. (448, 448)
                mask_list = []

                for raw in raw_rgbs:                                # iterate 3 camera views
                    # 0/1 background flag in raw image space
                    bg_bool = (raw[..., 0] > 253) & \
                            (raw[..., 1] > 253) & \
                            (raw[..., 2] > 253)
                    fg_mask = (~bg_bool).astype(np.float32)         # 1 = foreground, 0 = bg

                    # tensorify and resize to (H_px, W_px) with *nearest* to keep binary
                    m = torch.from_numpy(fg_mask)                   # (H0, W0)
                    m = F.interpolate(m.unsqueeze(0).unsqueeze(0),
                                    size=(H_px, W_px),
                                    mode='nearest').squeeze(0).squeeze(0)  # (H_px, W_px)
                    mask_list.append(m)

                mask_all = torch.stack(mask_list, 0)                # (3, H_px, W_px)

                emb_grp = grp[embedding_type]

                # ---- build samples ------------------------------------------ #
                for key, colour in region_map.items():
                    if colour not in color_to_idx or colour not in emb_grp:
                        continue
                    cidx      = color_to_idx[colour]
                    desc_list = json.loads(key)                  # list[str]
                    emb_mat   = emb_grp[colour][()]              # (N, D) - dimension depends on embedding type
                    if emb_mat.shape[0] != len(desc_list):
                        continue       # mis-aligned, skip

                    # Set embedding size from first valid embedding
                    if self._embedding_size is None and emb_mat.shape[1] > 0:
                        self._embedding_size = emb_mat.shape[1]

                    for j, desc in enumerate(desc_list):
                        emb = torch.from_numpy(emb_mat[j]).float()
                        for cam in range(3):
                            sim = torch.from_numpy(sim_all[cidx, cam]).float()  # (H_m, W_m)
                            rgb = rgb_all[cam]                                   # (3, H_px, W_px)
                            mask = mask_all[cam]             # (H',W')
                            self.samples.append((desc, emb, sim, rgb, mask))

    # ------------------------------------------------------------
    def _build_bg_bank(self):
        """
        Build background tensor list from one or more directories.
        Images are resized to match the foreground size. Must be called
        *after* the dataset has at least one sample, e.g.:

            ds = RegionSimDataset(h5_files, random_pad=True)
            ds._build_bg_bank("/path/to/bg_imgs")
        """
        if not self.samples:
            raise RuntimeError("Dataset empty – cannot infer target shape")
        if self._bg_dir is None:
            raise RuntimeError("Background directory not provided")

        if isinstance(self._bg_dir, str):
            bg_dirs_to_process = [self._bg_dir]
        elif isinstance(self._bg_dir, list):
            bg_dirs_to_process = self._bg_dir
        else: # Should not happen based on type hint, but defensive check
             raise TypeError(f"bg_dir must be a string or list of strings, got {type(self._bg_dir)}")

        # target spatial size from first sample's RGB
        _, H, W = self.samples[0][3].shape           # [3 , H , W]
        target_size = (H, W)

        bank = []
        num_processed = 0
        for dir_path in bg_dirs_to_process:
            if not os.path.isdir(dir_path):
                print(f"[BG] WARNING: Skipping non-existent directory: {dir_path}")
                continue

            print(f"[BG] Scanning directory: {dir_path}")
            for fp in glob.glob(os.path.join(dir_path, "*")):
                try:
                    arr = np.array(Image.open(fp).convert("RGB"))
                except Exception as e:
                    # print(f"[BG] Skipping file {fp} due to error: {e}") # Optional: more verbose logging
                    continue

                # same preprocessing as foreground (no blur)
                t = transform_imgs(arr, blur=False)[0]   # (3,h,w)

                # resize if needed
                if t.shape[1:] != target_size:
                    t = F.interpolate(t.unsqueeze(0),
                                      size=target_size,
                                      mode='bilinear',
                                      align_corners=False).squeeze(0)
                bank.append(t)
                num_processed += 1

        if bank:
            self._bg_bank = bank
            print(f"[BG] built bank with {len(bank)} images "
                  f"from {len(bg_dirs_to_process)} director{'y' if len(bg_dirs_to_process) == 1 else 'ies'} "
                  f"→ tensor shape {bank[0].shape}")
        else:
            print(f"[BG] WARNING: no usable images found in the provided director{'y' if len(bg_dirs_to_process) == 1 else 'ies'}.")
            self._bg_bank = None


    # ------------------------------------------------------------------ #
    #                       AUGMENTATION HELPERS                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rand_pad_same(rgb: torch.Tensor, sim: torch.Tensor, msk: torch.Tensor):
        """
        Pad both tensors with identical margins (replicate for RGB, zeros for sim),
        then resize back so shapes stay unchanged.
        """
        _, Hpx, Wpx = rgb.shape
        Hm,  Wm     = sim.shape
        patch       = Hpx // Hm                     # expected 7

        # random pad sizes in sim units
        pad_top, pad_bottom, pad_left = torch.randint(0, int(Hm * 0.8), (3,))
        pad_right = max(0, pad_top + pad_bottom - pad_left)

        # pad
        rgb = F.pad(rgb,
                    (pad_left*patch, pad_right*patch,
                     pad_top*patch,  pad_bottom*patch),
                    mode='replicate')
        sim = F.pad(sim,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', value=0)
        msk = F.pad(msk, (pad_left*patch, pad_right*patch, pad_top*patch, pad_bottom*patch), 
                    mode='constant', value=0)

        # resize back
        rgb = F.interpolate(rgb.unsqueeze(0),
                            size=(Hpx, Wpx),
                            mode='bilinear', align_corners=False).squeeze(0)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0),
                            size=(Hm, Wm), mode='nearest').squeeze()
        msk = F.interpolate(msk.unsqueeze(0).unsqueeze(0),
                            size=(Hpx, Wpx), mode='nearest').squeeze()

        return rgb, sim, msk

    # ................................................................. #
    def _process_sim_map(self, sim: torch.Tensor, rgb: torch.Tensor):
        """
        Apply shutdown-threshold, optional Gaussian blur, and down-scale by 7.
        """
        # threshold
        sim = torch.where(sim > self._thresh,
                          sim,
                          torch.tensor(0.0, device=sim.device))

        # blur
        if self._blur:
            sim = T.GaussianBlur(kernel_size=3, sigma=1.0)(
                      sim.unsqueeze(0).unsqueeze(0)).squeeze()

        # down-scale to DINO patch grid (factor 7)
        _, Hpx, Wpx = rgb.shape
        h, w = Hpx // 14, Wpx // 14
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0),
                            size=(h, w), mode='area').squeeze()
        return sim

    def get_embedding_size(self):
        """Return the embedding dimension size."""
        return self._embedding_size

    # ------------------------------------------------------------------ #
    #                      TORCH DATASET API                              #
    # ------------------------------------------------------------------ #
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        text, emb, sim_raw, rgb, mask = self.samples[idx]

        if self._rand_pad:
            rgb, sim_raw, mask = self._rand_pad_same(rgb, sim_raw, mask)

        # -------- background replacement ----------
        if self._bg_bank:
            bg = random.choice(self._bg_bank)                # (3,H',W')
            # ensure size match (should be identical after transform_imgs)
            if bg.shape != rgb.shape:
                bg = F.interpolate(bg.unsqueeze(0),
                                    size=rgb.shape[1:], mode='bilinear',
                                    align_corners=False).squeeze(0)
            rgb = rgb * mask + bg * (1 - mask)               # broadcast mask

        sim_proc = self._process_sim_map(sim_raw, rgb)

        return {
            "text"          : text,
            "text_emb"      : emb,         # (D,) - dimension depends on embedding type
            "sim_proj"      : sim_proc,    # (H_px/7, W_px/7)
            "processed_img" : rgb          # (3, H_px, W_px)  DINO-ready
        }

# --------------------------------------------------------------------------- #
#                             Visualize sample                             #
# --------------------------------------------------------------------------- #
def _demo_vis(dataset: RegionSimDataset, n_show: int = 4, seed: int = 0):
    """Plot a few (rgb, sim heat-map) pairs to visually inspect correctness."""
    import matplotlib.pyplot as plt

    random.seed(seed)
    idxs = random.sample(range(len(dataset)), k=min(n_show, len(dataset)))

    for i, idx in enumerate(idxs, 1):
        sample = dataset[idx]
        rgb = sample["processed_img"].permute(1, 2, 0).numpy()     # H W 3
        sim = sample["sim_proj"].numpy()                 # H W
        txt = sample["text"]

        plt.figure(figsize=(6, 3))
        plt.suptitle(f"[{idx}] {txt}", fontsize=8)

        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("RGB")

        plt.subplot(1, 2, 2)
        plt.imshow(sim, cmap="viridis")
        plt.axis("off")
        plt.title("Similarity")

        plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.show() # visualize data sample


def create_dataset(data_root, use_categories=None, random_pad=False, bg_dir=None, embedding_type="embeddings_oai"):
    """
    Create a dataset from a root directory of h5 files.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing data files under h5 folder
    use_categories : str or list[str], optional
        Specific categories to use. If None, uses all available h5 files.
    random_pad : bool
        Whether to apply random padding augmentation
    bg_dir : str or list[str], optional
        Directory(ies) containing background images
    embedding_type : str
        Which embedding to use ("embeddings" or "st_embeddings")
    """
    if not os.path.exists(os.path.join(data_root, 'h5')):
        raise FileNotFoundError(f"h5 folder not found in {data_root}, please pass in the path to the h5 folder")
    
    if use_categories:
        if isinstance(use_categories, str):
            use_categories = [use_categories]
        all_category_h5_paths = []
        for cat_name in use_categories:
            category_h5_path = os.path.join(data_root, 'h5', f'{cat_name}.h5')
            if os.path.exists(category_h5_path):
                all_category_h5_paths.append(category_h5_path)

    else:
        all_category_h5_paths = glob.glob(os.path.join(data_root, 'h5', '*.h5'))

    return RegionSimDataset(all_category_h5_paths, random_pad=random_pad, bg_dir=bg_dir, 
                           embedding_type=embedding_type)

# --------------------------------------------------------------------------- #
#                                ENTRY POINT                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual sanity-check for RegionSimDataset with configurable embeddings"
    )
    parser.add_argument("--data_root", type=str, 
                        required=True,
                        help="Root directory containing a h5 folder with .h5 files")
    parser.add_argument("--categories", nargs='+', default=None,
                        help="Categories to use (default: all)")
    parser.add_argument("--embedding_type", type=str, default="embeddings_oai",
                        choices=["embeddings_oai", "embeddings_st"],
                        help="Which embedding type to use")
    parser.add_argument("--n_show", type=int, default=1,
                        help="Number of samples to visualize")
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    # Create dataset with specified embedding type
    ds = create_dataset(args.data_root, args.categories, random_pad=False, 
                       bg_dir=None, embedding_type=args.embedding_type)
    ds._rand_pad = True
    
    print(f"Dataset size: {len(ds)}")
    print(f"Using embedding type: {args.embedding_type}")
    
    if len(ds) > 0:
        # Check embedding dimension of first sample
        first_sample = ds[0]
        print(f"Embedding dimension from sample: {first_sample['text_emb'].shape}")
        print(f"Embedding dimension from dataset: {ds.get_embedding_size()}")
        
        _demo_vis(ds, n_show=args.n_show)
    else:
        print("No samples found in dataset!")

    # save dataset to pt file
    name = args.name
    dataset_path = os.path.join(args.data_root, "dataset", f'{args.embedding_type}_{name}.pt')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save(ds, dataset_path)
    print(f"Saved dataset to {dataset_path}")