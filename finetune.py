import argparse
import datetime
from pathlib import Path
import __main__
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split

import wandb

from model.dataset import RegionSimDataset
from model.network import Conv2DFiLMNet
from utils.file_utils import load_config
from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs

def register_legacy_pickle_classes():
    # Datasets saved by running model/dataset.py directly were pickled as __main__.RegionSimDataset.
    if not hasattr(__main__, "RegionSimDataset"):
        __main__.RegionSimDataset = RegionSimDataset


def tv_loss(logits):
    # logits: (B, 1, H, W) or (B, H, W), the output of the model before sigmoid.
    if logits.dim() == 4:
        logits = logits.squeeze(1) # (B, H, W)
    prob = torch.sigmoid(logits)
    tv_h = (prob[:, 1:, :] - prob[:, :-1, :]).abs().mean()
    tv_w = (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()
    return tv_h + tv_w


class Finetuner:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_data()
        self.build_model()
        self.build_dino()
        self.build_optimizer()

        date = datetime.datetime.now().strftime("%Y%m%d")
        self.out_dir = Path(args.output_dir) / date / args.run_name
        (self.out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.lambda_tv = args.lambda_tv
        self.global_step = 0

        self.use_wandb = (not args.no_wandb)
        if self.use_wandb:
            wandb.init(
                project="Affordance_train",
                name=args.run_name,
                config={
                    "batch": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "lambda_tv": args.lambda_tv,
                    "val_split": args.val_split,
                    "init_ckpt": args.init_ckpt,
                    "data": args.data,
                },
            )

    def build_data(self):
        # data was processed into .pt file.
        register_legacy_pickle_classes()
        ds_list = [torch.load(p, map_location="cpu", weights_only=False) for p in self.args.data]
        print("Loaded", len(ds_list), "datasets with sizes:", [len(ds) for ds in ds_list])
        for ds in ds_list:
            ds._bg_dir = self.cfg['dataset_bg_dir']
            ds._build_bg_bank()
            ds._thresh = self.args.thresh

        dataset = torch.utils.data.ConcatDataset(ds_list)
        val_ratio = self.args.val_split if self.args.val_split else 0.1
        n_val = max(1, int(len(dataset) * val_ratio))
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def build_model(self):
        model_cfg = self.cfg['model']
        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)

        ckpt = torch.load(self.args.init_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)

    def build_dino(self):
        dino_model_type = self.cfg.get("dino_model_type", "dinov2_vitl14")
        dino_use_registers = self.cfg.get("dino_use_registers", False)
        self.dino = load_pretrained_dino(
            dino_model_type,
            use_registers=dino_use_registers,
            torch_path=self.cfg.get("torch_home", None),
        ).to(self.device).eval()
        for p in self.dino.parameters():
              p.requires_grad_(False)

    def build_optimizer(self):
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        
    def run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0
        total_bce = 0.0
        total_tv = 0.0
        vis_batch = None
        for batch in loader:
            rgb = batch['processed_img'].to(self.device, non_blocking=True)
            tgt = batch["sim_proj"].to(self.device, non_blocking=True)
            emb = batch["text_emb"].to(self.device, non_blocking=True)

            with torch.no_grad():
                feat = get_dino_features_from_transformed_imgs(
                    self.dino,
                    rgb,
                    repeat_to_orig_size=False,
                ).permute(0, 3, 1, 2)

            with torch.set_grad_enabled(train):
                # train can be set to False when we want to evaluate the model without updating weights, e.g. during validation.
                logits = self.model(feat, emb).squeeze(1)
                loss_bce = self.bce(logits, tgt)
                loss_tv = tv_loss(logits)
                loss = loss_bce + self.lambda_tv * loss_tv

                if vis_batch is None:
                    with torch.no_grad():
                        vis_batch = {
                            "sim_proj": tgt.detach().cpu(),
                            "predictions": torch.sigmoid(logits).detach().cpu(),
                            "text": batch.get("text", [""] * rgb.size(0)),
                        }

                if train:
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.sqrt(
                        sum(
                            p.grad.detach().pow(2).sum()
                            for p in self.model.parameters()
                            if p.grad is not None
                        )
                    ).item()
                    self.optim.step()

                    if self.use_wandb:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/bce": loss_bce.item(),
                            "train/tv": loss_tv.item(),
                            "train/grad_norm": grad_norm,
                            "step": self.global_step,
                        }, commit=False)

                    self.global_step += 1

            bs = rgb.size(0) # batch size
            total_loss += loss.item() * bs
            total_bce += loss_bce.item() * bs
            total_tv += loss_tv.item() * bs

        n = len(loader.dataset)
        return {'loss': total_loss / n, 'bce': total_bce / n, 'tv': total_tv / n}, vis_batch

    def _wandb_vis_samples(self, batch, max_n: int = 3):
        if not self.use_wandb or batch is None:
            return

        tgt = batch["sim_proj"]
        pred = batch["predictions"]
        texts = batch["text"]
        n_show = min(max_n, tgt.size(0))
        for i in range(n_show):
            caption = texts[i] if i < len(texts) else ""
            img_gt = wandb.Image(
                TF.to_pil_image((tgt[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | GT mask",
            )
            img_pred = wandb.Image(
                TF.to_pil_image((pred[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | Pred mask",
            )
            wandb.log({
                f"samples/{i+1}/gt": img_gt,
                f"samples/{i+1}/pred": img_pred,
            })
    
    def save_ckpt(self, name):
        path = self.out_dir / "ckpts" / f"{name}.pth"
        torch.save({
                "model": self.model.state_dict(),
                "config": self.cfg,
                "source_ckpt": self.args.init_ckpt},
            path,
        )
        print(f"Saved checkpoint: {path}")

    def fit(self):
        best = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_log, vis_batch = self.run_epoch(self.train_loader, train=True)
            val_log, vis_batch = self.run_epoch(self.val_loader, train=False)

            log = {
                "epoch": epoch,
                "train_loss": train_log["loss"],
                "train_bce": train_log["bce"],
                "train_tv": train_log["tv"],
                "val_loss": val_log["loss"],
                "val_bce": val_log["bce"],
                "val_tv": val_log["tv"],
                "lr": self.optim.param_groups[0]["lr"],
            }

            if val_log["loss"] < best:
                best = val_log["loss"]
                self.save_ckpt("best")

            self._wandb_vis_samples(vis_batch)
            if self.use_wandb:
                wandb.log(log)

            print(
                f"[{epoch}/{self.args.epochs}] "
                f"train loss={train_log['loss']:.4f} "
                f"bce={train_log['bce']:.4f} "
                f"tv={train_log['tv']:.4f} | "
                f"val loss={val_log['loss']:.4f} "
                f"bce={val_log['bce']:.4f} "
                f"tv={val_log['tv']:.4f}"
            )

        self.save_ckpt("final")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/oai_vitl_cot.yaml')
    parser.add_argument("--data", default=['dataset/h5/pt/embeddings_oai_proposal_sql.pt'], nargs="+")
    parser.add_argument("--init-ckpt", default='logs/20260413/oai_vitl_cot/ckpts/final.pth')
    parser.add_argument("--run-name", default='finetune')
    parser.add_argument("--output-dir", default="logs/finetune_1")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)

    args = parser.parse_args()
    Finetuner(args).fit()


if __name__ == "__main__":
    main()