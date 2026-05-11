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
from model.early_film import EarlyFiLMDINOv2
from utils.file_utils import load_config


def register_legacy_pickle_classes():
    if not hasattr(__main__, "RegionSimDataset"):
        __main__.RegionSimDataset = RegionSimDataset


def tv_loss(logits):
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    prob = torch.sigmoid(logits)
    tv_h = (prob[:, 1:, :] - prob[:, :-1, :]).abs().mean()
    tv_w = (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def trainable_state_dict(model):
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if name in trainable_names
    }


class EarlyFiLMTrainer:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_data()
        self.build_model()
        self.build_optimizer()

        date = datetime.datetime.now().strftime("%Y%m%d")
        self.out_dir = Path(args.output_dir) / date / args.run_name
        (self.out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.lambda_tv = args.lambda_tv
        self.global_step = 0

        self.use_wandb = not args.no_wandb
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
                    "data": args.data,
                    "config": args.config,
                    "amp": args.amp,
                },
            )

    def build_data(self):
        register_legacy_pickle_classes()
        ds_list = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in self.args.data
        ]
        print("Loaded", len(ds_list), "datasets with sizes:", [len(ds) for ds in ds_list])

        dataset_bg_dir = self.cfg.get("dataset_bg_dir", None)
        trainer_cfg = self.cfg.get("trainer", {})
        for ds in ds_list:
            if dataset_bg_dir is not None:
                ds._bg_dir = dataset_bg_dir
                ds._build_bg_bank()
            ds._thresh = self.args.thresh

            expected_random_pad = trainer_cfg.get("random_pad", getattr(ds, "_rand_pad", None))
            if expected_random_pad != getattr(ds, "_rand_pad", None):
                print(
                    "Trainer random_pad",
                    expected_random_pad,
                    "!=",
                    "dataset setting",
                    getattr(ds, "_rand_pad", None),
                )

        dataset = torch.utils.data.ConcatDataset(ds_list)
        val_ratio = self.args.val_split
        n_val = max(1, int(len(dataset) * val_ratio))
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _model_cfg(self):
        if "model" in self.cfg:
            return self.cfg["model"]
        return self.cfg.get("local_inferer", {}).get("model", {})

    def build_model(self):
        model_cfg = self._model_cfg()
        self.model = EarlyFiLMDINOv2(**model_cfg).to(self.device)

        if self.args.resume_ckpt:
            ckpt = torch.load(self.args.resume_ckpt, map_location=self.device)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            strict = not ckpt.get("trainable_only", False)
            result = self.model.load_state_dict(state, strict=strict)
            if not strict:
                print(
                    "Loaded adapter checkpoint: "
                    f"{len(result.missing_keys)} missing keys, "
                    f"{len(result.unexpected_keys)} unexpected keys."
                )

        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for _, p in trainable)
        print(f"Trainable parameter tensors: {len(trainable)}")
        print(f"Trainable parameters: {total_trainable:,}")
        for name, _ in trainable[:20]:
            print("  trainable:", name)
        if len(trainable) > 20:
            print(f"  ... {len(trainable) - 20} more")

    def build_optimizer(self):
        self.optim = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def set_train_mode(self, train: bool):
        self.model.train(train)
        # Frozen foundation models must stay deterministic even during adapter training.
        self.model.text_model.eval()
        self.model.vision_model.trunk.eval()

    def _resize_target_if_needed(self, target, logits):
        if logits.shape[-2:] == target.shape[-2:]:
            return target
        return F.interpolate(
            target.unsqueeze(1),
            size=logits.shape[-2:],
            mode="area",
        ).squeeze(1)

    def run_epoch(self, loader, train=True):
        self.set_train_mode(train)
        total_loss = 0.0
        total_bce = 0.0
        total_tv = 0.0
        vis_batch = None

        for batch in loader:
            rgb = batch["processed_img"].to(self.device, non_blocking=True)
            tgt = batch["sim_proj"].to(self.device, non_blocking=True)
            texts = list(batch["text"])

            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.args.amp and self.device.type == "cuda",
                    dtype=torch.bfloat16,
                ):
                    logits = self.model.get_heatmap_logits(
                        rgb,
                        texts=texts,
                        interpolate=False,
                    ).squeeze(1)
                    tgt_for_loss = self._resize_target_if_needed(tgt, logits)
                    loss_bce = self.bce(logits.float(), tgt_for_loss.float())
                    loss_tv = tv_loss(logits.float())
                    loss = loss_bce + self.lambda_tv * loss_tv

                if vis_batch is None:
                    with torch.no_grad():
                        vis_batch = {
                            "sim_proj": tgt_for_loss.detach().cpu(),
                            "predictions": torch.sigmoid(logits.float()).detach().cpu(),
                            "text": texts,
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
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/bce": loss_bce.item(),
                                "train/tv": loss_tv.item(),
                                "train/grad_norm": grad_norm,
                                "step": self.global_step,
                            },
                            commit=False,
                        )
                    self.global_step += 1

            bs = rgb.size(0)
            total_loss += loss.item() * bs
            total_bce += loss_bce.item() * bs
            total_tv += loss_tv.item() * bs

        n = len(loader.dataset)
        return {
            "loss": total_loss / n,
            "bce": total_bce / n,
            "tv": total_tv / n,
        }, vis_batch

    def wandb_vis_samples(self, batch, max_n=3):
        if not self.use_wandb or batch is None:
            return

        tgt = batch["sim_proj"]
        pred = batch["predictions"]
        texts = batch["text"]
        n_show = min(max_n, tgt.size(0))
        for i in range(n_show):
            caption = texts[i] if i < len(texts) else ""
            gt_img = wandb.Image(
                TF.to_pil_image((tgt[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | GT mask",
            )
            pred_img = wandb.Image(
                TF.to_pil_image((pred[i] * 255).byte(), mode="L").convert("RGB"),
                caption=f"{caption} | Pred mask",
            )
            wandb.log({
                f"samples/{i + 1}/gt": gt_img,
                f"samples/{i + 1}/pred": pred_img,
            })

    def save_ckpt(self, name):
        path = self.out_dir / "ckpts" / f"{name}.pth"
        state = (
            self.model.state_dict()
            if self.args.save_full_model
            else trainable_state_dict(self.model)
        )
        torch.save(
            {
                "model": state,
                "config": self.cfg,
                "epoch": name,
                "trainable_only": not self.args.save_full_model,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def fit(self):
        best = float("inf")
        for epoch in range(1, self.args.epochs + 1):
            train_log, _ = self.run_epoch(self.train_loader, train=True)
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

            if epoch % self.args.save_every == 0:
                self.save_ckpt(f"epoch_{epoch}")

            self.wandb_vis_samples(vis_batch)
            if self.use_wandb:
                wandb.log(log)

            print(
                f"[{epoch}/{self.args.epochs}] "
                f"train loss={train_log['loss']:.4f} "
                f"bce={train_log['bce']:.4f} "
                f"tv={train_log['tv']:.4f} | "
                f"val loss={val_log['loss']:.4f} "
                f"bce={val_log['bce']:.4f} "
                f"tv={val_log['tv']:.4f} | "
                f"lr={log['lr']:.2e}"
            )

        self.save_ckpt("final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/earlyfilm_vitl.yaml")
    parser.add_argument("--data", default=["dataset/h5/pt/embeddings_oai_proposal_sql.pt"], nargs="+")
    parser.add_argument("--run-name", default="earlyfilm_vitl")
    parser.add_argument("--output-dir", default="logs/earlyfilm")
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--no_wandb", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-full-model", action="store_true")

    args = parser.parse_args()
    EarlyFiLMTrainer(args).fit()


if __name__ == "__main__":
    main()
