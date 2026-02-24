import argparse, datetime, os
from pathlib import Path
import yaml

import numpy as np
from PIL import Image

import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model.dataset import RegionSimDataset

from utils.file_utils import load_config

try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    print("wandb not found – install it or use --no_wandb")

from model.network import Conv2DFiLMNet
from utils.img_utils import load_pretrained_dino, get_dino_features_from_transformed_imgs

# =====================================================================
class Trainer:
    def __init__(self, args):

        # Load configuration from YAML file
        cfg = load_config(args.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- Dataset ----------------------------------------------------
        ds_list = [torch.load(p, map_location="cpu", weights_only=False) for p in args.data]
        print("Loaded", len(ds_list), "datasets with sizes:",
            [len(d) for d in ds_list])

        # Apply same background / threshold tweaks to each dataset
        for ds in ds_list:
            ds._bg_dir = cfg["dataset_bg_dir"]
            ds._build_bg_bank()
            ds._thresh = args.thresh

            if cfg["trainer"]["random_pad"] != getattr(ds, "_rand_pad", None):
                print("Trainer random_pad", cfg["trainer"]["random_pad"],
                    "≠ dataset setting", getattr(ds, "_rand_pad", None))
                
            assert ds._embedding_size == cfg["model"]["lang_emb_dim"], f"Dataset lang_emb_dim {ds._embedding_size} != model lang_emb_dim {cfg['model']['lang_emb_dim']}"

        # -------- train / val splits inside every dataset -----------------
        val_split = cfg["trainer"]["val_split"]
        tr_parts, val_parts = [], []
        for ds in ds_list:
            n_val = max(1, int(val_split * len(ds)))
            n_tr  = len(ds) - n_val
            tr, val = random_split(ds, [n_tr, n_val])
            tr_parts.append(tr)
            val_parts.append(val)

        train_ds = torch.utils.data.ConcatDataset(tr_parts)
        val_ds   = torch.utils.data.ConcatDataset(val_parts)

        # -------- balanced sampler ----------------------------------------
        # weight = 1/len(dataset_i)  → equal dataset probability
        weights = []
        for part, orig_ds in zip(tr_parts, ds_list):
            weights.extend([1.0 / len(orig_ds)] * len(part))

        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
        )

        bs = cfg["trainer"]["batch_size"]
        self.train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler,
            num_workers=4, pin_memory=True)
        self.val_loader   = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=True)

        # ---------------- Model -------------------------------------
        model_cfg = cfg["model"]
        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)

        lr = args.lr or cfg["optim"]["lr"]
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=cfg["optim"]["weight_decay"]
        )
        self.lr_sched = StepLR(self.optim,
                               step_size=cfg["optim"]["step_size"],
                               gamma    =cfg["optim"]["gamma"])
        
        self.global_step = 0  

        self.criterion = torch.nn.BCEWithLogitsLoss()

        dino_model_type = "dinov2_vitl14"
        self.dino = load_pretrained_dino(
            dino_model_type, 
            use_registers=True, 
            torch_path=cfg["torch_home"],
        ).to(self.device).eval()
        # self.dino = load_pretrained_dino('dinov2_vitl14', use_registers=True, torch_path=cfg["torch_home"]).to(self.device).eval()

        # ---------------- Logging / dirs ----------------------------
        date = datetime.datetime.now().strftime("%Y%m%d")
        exp = args.run_name

        self.log_dir = Path(__file__).parent.parent / "logs" / date / exp
        (self.log_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        self.use_wandb = (not args.no_wandb) and _wandb_available
        if self.use_wandb:
            wandb.init(project="Affordance_train",
                    name=exp,
                    config={"batch": bs, "lr": lr})

        # epochs & intervals
        self.num_epochs = args.epochs or cfg["trainer"]["num_epochs"]
        self.save_every = cfg["trainer"]["save_every"]
        self.val_every  = cfg["trainer"]["val_every"]

        # resume if supplied
        if args.resume_ckpt:
            self._load_ckpt(args.resume_ckpt)

    # ----------------------------------------------------------------
    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total, vis_batch = 0.0, None

        for batch in loader:

            rgb  = batch["processed_img"].to(self.device, non_blocking=True)
            tgt  = batch["sim_proj"].to(self.device, non_blocking=True)
            emb  = batch["text_emb"].to(self.device, non_blocking=True)

            feat = get_dino_features_from_transformed_imgs(
                       self.dino, rgb, repeat_to_orig_size=False
                   ).permute(0, 3, 1, 2)

            self.optim.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                out   = self.model(feat, emb).squeeze(1)
                loss  = self.criterion(out, tgt)

                # Save vis_batch with inputs AND outputs on CPU
                if vis_batch is None:
                    with torch.no_grad():
                        pred = torch.sigmoid(out)  # Get predictions
                        vis_batch = {
                            "sim_proj": tgt.cpu(), 
                            "text_emb": emb.cpu(),
                            "text": batch["text"],
                            "predictions": pred.cpu()
                        }

                if train:
                    loss.backward()

                    # ---------- gradient L2 norm ----------
                    grad_norm = torch.sqrt(
                        sum(p.grad.detach().pow(2).sum()
                            for p in self.model.parameters()
                            if p.grad is not None)
                    ).item()

                    self.optim.step()

                    # ---------- per-batch WandB log ----------
                    if self.use_wandb:
                        wandb.log({
                            "train/loss"      : loss.item(),
                            "train/grad_norm" : grad_norm,
                            "step"            : self.global_step
                        }, commit=False)

                    self.global_step += 1

            total += loss.item() * rgb.size(0)

        return total / len(loader.dataset), vis_batch
    
    def _wandb_vis_samples(self, batch, max_n: int = 3):
        """
        Logs up to max_n samples from batch to WandB with 3 panels each.
        Now expects batch to contain pre-computed predictions on CPU.
        """
        if not self.use_wandb or batch is None:
            return
        
        tgt = batch["sim_proj"]         # (B,h,w) on CPU  
        pred = batch["predictions"]     # (B,h,w) on CPU - pre-computed!
        
        n_show = min(max_n, tgt.size(0))
        for i in range(n_show):
            caption = batch["text"][i]
            
            img_gt = wandb.Image(
                TF.to_pil_image(
                    (tgt[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | GT mask"
            )
            img_pred = wandb.Image(
                TF.to_pil_image(
                    (pred[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | Pred mask"
            )
            
            wandb.log({
                f"samples/{i+1}/gt": img_gt,  
                f"samples/{i+1}/pred": img_pred,
            })

    # ---------------- checkpoint utils ------------------------------
    def _save_ckpt(self, tag):
        f = self.log_dir / "ckpts" / f"{tag}.pth"
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": tag,
        }, f)

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        print(f"Resumed from checkpoint {path}")

    # ------------------- train loop ---------------------------------
    def fit(self):
        best = float("inf")
        for ep in range(1, self.num_epochs + 1):
            tr_loss, vis_batch = self._run_epoch(self.train_loader, train=True)
            self.lr_sched.step()

            log = {"epoch": ep, "train_loss": tr_loss,
                   "lr": self.lr_sched.get_last_lr()[0]}

            if ep % self.val_every == 0:
                val_loss, vis_batch = self._run_epoch(self.val_loader, train=False)
                log["val_loss"] = val_loss
                if val_loss < best:
                    best = val_loss
                    self._save_ckpt("best")
                
            # log training samples if not val epoch
            self._wandb_vis_samples(vis_batch) 

            if ep % self.save_every == 0:
                self._save_ckpt(f"epoch_{ep}")

            if self.use_wandb:
                wandb.log(log)

            # console
            msg = (f"[{ep}/{self.num_epochs}] "
                   f"train {tr_loss:.4f}  "
                   + (f"| val {val_loss:.4f}  " if "val_loss" in log else "")
                   + f"| lr {log['lr']:.2e}")
            print(msg)

        self._save_ckpt("final")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--data", required=True, nargs='+',
                help="One or more RegionSimDataset .pt files")
    ap.add_argument("--run_name", required=True, help="Run name")
    
    ap.add_argument("--thresh", type=float, default=0.5, help="Threshold for logits")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch",  type=int)
    ap.add_argument("--lr",     type=float)
    ap.add_argument("--resume_ckpt", help="Checkpoint path to resume")
    ap.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    args = ap.parse_args()

    Trainer(args).fit()


if __name__ == "__main__":
    main()

