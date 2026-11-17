import os
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score

from Config import Config
from SlotModel import build_model
from data.clevr import CLEVRDataset, clevr_collate
from data.clevr_precomputed import PrecomputedCLEVRStream, precomputed_clevr_collate
from visualize_slots import visualize_batch


def compute_fg_ari(masks, label_map):
    pred_labels = masks.squeeze(2).argmax(dim=1)
    scores = []
    for b in range(masks.shape[0]):
        gt = label_map[b].cpu().numpy().reshape(-1)
        pred = pred_labels[b].cpu().numpy().reshape(-1)
        fg = gt != 0
        scores.append(adjusted_rand_score(gt[fg], pred[fg]))
    return sum(scores) / len(scores)

import sys
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
        self.step = 0
        self._last_log_time = None
        self._last_log_step = 0
        self.best_val_mse = float("inf")   # NEW: tracks best checkpoint seen so far
        self.train_dataset_size = len(train_loader.dataset)   # NEW: needed to compute epoch from step count

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)   # NEW

    def lr_at_step(self, step):
        warmup = min(1.0, step / self.cfg.warmup_steps)
        decay = self.cfg.decay_rate ** (step / self.cfg.decay_steps)
        return self.cfg.base_lr * warmup * decay

    # ---------------------------------------------------------------
    # NEW: checkpoint save / load
    # ---------------------------------------------------------------
    def save_checkpoint(self, path):
        # includes everything needed to exactly resume: model weights,
        # optimizer state (Adam's running moment estimates -- resuming
        # without these causes a jump in the loss curve), and the step
        # counter (so lr_at_step picks up the schedule where it left off)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "best_val_mse": self.best_val_mse,
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"resumed from {path} at step {self.step}", flush=True)

    def maybe_resume(self):
        # NEW: looks for the most recent "latest.pt" and picks up from there
        # automatically -- call this once before fit() starts
        latest_path = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            self.load_checkpoint(latest_path)
        else:
            print("no checkpoint found, starting from scratch", flush=True)

    def train_step(self, batch):
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        lr = self.lr_at_step(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        recon, masks, _ = self.model(batch)
        loss = F.mse_loss(recon, batch["image"].to(self.device))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.step += 1
        return loss.item(), lr

    def epoch_at_step(self, step):   # NEW
        return (step * self.cfg.batch_size) / self.train_dataset_size

    def maybe_log(self, loss, lr):
        if self.step % self.cfg.log_every != 0:
            return
        epoch = self.epoch_at_step(self.step)
        now = time.time()
        if self._last_log_time is not None:
            elapsed = now - self._last_log_time
            steps_done = self.step - self._last_log_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
            imgs_per_sec = steps_per_sec * self.cfg.batch_size
            eta_hours = (
                (self.cfg.total_steps - self.step) / steps_per_sec / 3600
                if steps_per_sec > 0 else float("inf")
            )

            steps_per_epoch = self.train_dataset_size / self.cfg.batch_size   # NEW
            epoch_time_hours = (                                              # NEW
                steps_per_epoch / steps_per_sec / 3600 if steps_per_sec > 0 else float("inf")
            )

            print(
                f"step {self.step}/{self.cfg.total_steps} | epoch {epoch:.2f} | loss={loss:.4f} | lr={lr:.2e} "
                f"| {steps_per_sec:.2f} it/s | {imgs_per_sec:.1f} img/s "
                f"| {epoch_time_hours:.2f} h/epoch | ETA {eta_hours:.1f}h",   # NEW: added h/epoch
                flush=True,
            )
        else:
            print(f"step {self.step} | epoch {epoch:.2f}: loss={loss:.4f} (starting throughput measurement...)", flush=True)
        self._last_log_time = now
        self._last_log_step = self.step

    def evaluate(self, max_batches: int = 20):
        self.model.eval()
        mse_total, ari_total, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                recon, masks, _ = self.model(batch)
                mse_total += F.mse_loss(recon, batch["image"]).item()
                ari_total += compute_fg_ari(masks, batch["label_map"])
                n += 1
                if i == 0:
                    visualize_batch(self.model, batch, self.device, num_samples=2, out_dir="viz", step=self.step)
        self.model.train()
        return mse_total / n, ari_total / n

    def fit(self):
        self.model.train()
        self._last_log_time = time.time()
        pbar = tqdm(total=self.cfg.total_steps, desc="train", initial=self.step)
        while self.step < self.cfg.total_steps:
            for batch in self.train_loader:
                if self.step >= self.cfg.total_steps:
                    break
                loss, lr = self.train_step(batch)
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss:.4f}")
                self.maybe_log(loss, lr)

                if self.step % self.cfg.checkpoint_every == 0:   # NEW
                    self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "latest.pt"))

                if self.step % self.cfg.eval_every == 0:
                    val_mse, val_ari = self.evaluate()
                    print(f"[eval] step {self.step}: val_mse={val_mse:.4f} val_ari={val_ari:.4f}", flush=True)

                    if val_mse < self.best_val_mse:   # NEW: separate "best" checkpoint
                        self.best_val_mse = val_mse
                        self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "best.pt"))
                        print(f"[checkpoint] new best val_mse={val_mse:.4f}, saved best.pt", flush=True)

        self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "latest.pt"))  # NEW: final save at the end
        pbar.close()


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, device)

    if cfg.use_precomputed_clip:
        train_loader = DataLoader(
            PrecomputedCLEVRStream(cfg.clip_train_cache_dir, shuffle_buffer=cfg.clip_cache_shuffle_buffer),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=precomputed_clevr_collate,
        )
        val_loader = DataLoader(
            PrecomputedCLEVRStream(cfg.clip_val_cache_dir, shuffle_buffer=0),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=precomputed_clevr_collate,
        )
    else:
        train_loader = DataLoader(
            CLEVRDataset(cfg.train_dir, image_size=cfg.image_size, max_objects=cfg.max_objects),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=clevr_collate,
        )
        val_loader = DataLoader(
            CLEVRDataset(cfg.val_dir, image_size=cfg.image_size, max_objects=cfg.max_objects),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=clevr_collate,
        )

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.maybe_resume()   # NEW: picks up from checkpoints/latest.pt if it exists
    trainer.fit()


if __name__ == "__main__":
    main()