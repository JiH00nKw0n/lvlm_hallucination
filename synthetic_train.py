"""
Synthetic SAE training with 2D visualization video.

Trains TopKSAE(hidden=2, latent=5, k=1) on synthetic feature data and
records an mp4 showing dictionary atoms converging to ground truth features.
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from transformers import TrainerCallback, TrainingArguments

from src.datasets import SyntheticFeatureDatasetBuilder
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE
from src.runners.trainer import SAETrainer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Collator                                                           #
# ------------------------------------------------------------------ #

def synthetic_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    """Convert dataset samples to SAETrainer-compatible batch."""
    reps = torch.tensor(
        [f["representation"] for f in features], dtype=torch.float32,
    )
    return {"hidden_states": reps.unsqueeze(1)}  # (B, 1, hidden_size)


# ------------------------------------------------------------------ #
#  Trainer                                                            #
# ------------------------------------------------------------------ #

class SyntheticFeatureTrainer(SAETrainer):
    """SAETrainer + decoder norm/gradient projection per step."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        # gradient projection: remove component parallel to decoder directions
        # (skip when weight_tie=True since grad is shared with encoder)
        if not model.cfg.weight_tie and model.W_dec.grad is not None:
            model.remove_gradient_parallel_to_decoder_directions()
        return loss


# ------------------------------------------------------------------ #
#  DecoderNormCallback                                                #
# ------------------------------------------------------------------ #

class DecoderNormCallback(TrainerCallback):
    """Re-normalize decoder rows to unit norm after each optimizer step."""

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "set_decoder_norm_to_unit_norm"):
            model.set_decoder_norm_to_unit_norm()


# ------------------------------------------------------------------ #
#  Visualization Callback                                             #
# ------------------------------------------------------------------ #

class VisualizationCallback(TrainerCallback):
    """Capture 2D frames every N steps and save as mp4 on train end."""

    def __init__(
        self,
        gt_features: np.ndarray,
        train_data: np.ndarray,
        viz_every: int = 5,
        output_path: str = "outputs/synthetic_sae_training.mp4",
        fps: int = 10,
    ):
        super().__init__()
        self.gt_features = gt_features        # (n_features, 2)
        self.train_data = train_data           # (N, 2)
        self.viz_every = viz_every
        self.output_path = output_path
        self.fps = fps
        self._last_loss = None
        self.fig = None
        self.ax = None
        self.writer = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.writer = FFMpegWriter(fps=self.fps, metadata={"title": "SAE Training"})
        self.writer.setup(self.fig, self.output_path, dpi=150)
        self._capture_frame(model, step=0, loss=None)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.viz_every != 0:
            return
        self._capture_frame(model, state.global_step, self._last_loss)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self._last_loss = logs.get("loss", self._last_loss)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._capture_frame(model, state.global_step, self._last_loss)
        self.writer.finish()
        plt.close(self.fig)
        logger.info(f"Video saved to {self.output_path}")

    def _capture_frame(self, model, step: int, loss):
        ax = self.ax
        ax.clear()

        n_features = self.gt_features.shape[0]
        gt_colors = plt.cm.tab10(np.linspace(0, 1, n_features))

        # unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=0.5, alpha=0.3)

        # data scatter (background, very transparent)
        ax.scatter(
            self.train_data[:, 0], self.train_data[:, 1],
            s=6, c="silver", alpha=0.15, zorder=1,
        )

        # GT feature arrows
        for i in range(n_features):
            x, y = self.gt_features[i]
            ax.annotate(
                "", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=gt_colors[i], lw=2.5),
                zorder=3,
            )
            ax.text(
                x * 1.15, y * 1.15, f"GT{i}",
                fontsize=9, fontweight="bold", color=gt_colors[i],
                ha="center", va="center", zorder=4,
            )

        # SAE dictionary atoms (W_dec rows)
        if model is not None:
            W_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, 2)
            sae_colors = plt.cm.Set2(np.linspace(0, 1, W_dec.shape[0]))
            for j in range(W_dec.shape[0]):
                dx, dy = W_dec[j]
                ax.annotate(
                    "", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=sae_colors[j],
                        lw=2.0, linestyle="--",
                    ),
                    zorder=5,
                )
                ax.text(
                    dx * 1.15, dy * 1.15, f"D{j}",
                    fontsize=8, color=sae_colors[j],
                    ha="center", va="center", zorder=6,
                )

        # title
        title = f"Step {step}"
        if loss is not None:
            title += f"  |  loss = {loss:.4f}"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

        self.writer.grab_frame()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Synthetic SAE training with 2D video")
    p.add_argument("--feature-latent-dim", type=int, default=5)
    p.add_argument("--representation-dim", type=int, default=2)
    p.add_argument("--num-train", type=int, default=1000)
    p.add_argument("--sparsity", type=float, default=0.999)
    p.add_argument("--min-active", type=int, default=1)
    p.add_argument("--max-interference", type=float, default=0.3)
    p.add_argument("--latent-size", type=int, default=None,
                   help="SAE latent dim (default: same as feature-latent-dim)")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--viz-every", type=int, default=5)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/synthetic_sae")
    p.add_argument("--video-path", type=str, default="outputs/synthetic_sae_training.mp4")
    p.add_argument("--weight-tie", action="store_true",
                   help="Tie encoder and decoder weights (W_dec = W_enc)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Device: auto (detect), cuda, mps, cpu")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ---- Device ----
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # ---- Data ----
    logger.info("Building synthetic dataset...")
    builder = SyntheticFeatureDatasetBuilder(
        feature_latent_dim=args.feature_latent_dim,
        representation_dim=args.representation_dim,
        num_train=args.num_train,
        num_val=100,
        num_test=100,
        sparsity=args.sparsity,
        min_active=args.min_active,
        max_interference=args.max_interference,
        strategy="sdp",
        sdp_restarts=10,
        sdp_refine_steps=2000,
        seed=args.seed,
        verbose=True,
    )
    ds = builder.build_dataset()
    train_ds = ds["train"]

    gt_features = builder.wp.T          # (n_features, 2)
    train_data = np.array(train_ds["representation"])  # (N, 2)

    logger.info(f"GT features shape: {gt_features.shape}")
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Max interference: {builder.actual_max_interference:.4f}")

    # ---- Model ----
    latent_size = args.latent_size or args.feature_latent_dim
    logger.info(f"SAE latent_size: {latent_size} (GT features: {args.feature_latent_dim}, ratio: {latent_size/args.feature_latent_dim:.1f}x)")

    config = TopKSAEConfig(
        hidden_size=args.representation_dim,
        latent_size=latent_size,
        k=args.k,
        normalize_decoder=True,
        weight_tie=args.weight_tie,
    )
    sae = TopKSAE(config)
    logger.info(f"TopKSAE: hidden={config.hidden_size}, latent={config.latent_size}, k={config.k}, weight_tie={config.weight_tie}")

    # ---- Callbacks ----
    viz_cb = VisualizationCallback(
        gt_features=gt_features,
        train_data=train_data,
        viz_every=args.viz_every,
        output_path=args.video_path,
        fps=args.fps,
    )
    norm_cb = DecoderNormCallback()

    # ---- Trainer ----
    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_grad_norm=0.0,
        seed=args.seed,
        remove_unused_columns=False,
    )
    if device == "cpu":
        training_kwargs["use_cpu"] = True
    elif device == "mps":
        training_kwargs["use_cpu"] = False  # Trainer auto-detects MPS
    # cuda is the default when available, no extra flag needed
    training_args = TrainingArguments(**training_kwargs)

    trainer = SyntheticFeatureTrainer(
        model=sae,
        args=training_args,
        train_dataset=train_ds,
        data_collator=synthetic_collator,
        callbacks=[viz_cb, norm_cb],
    )

    # ---- Train ----
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # ---- Open video ----
    video_path = Path(args.video_path)
    if video_path.exists():
        logger.info(f"Opening {video_path}")
        subprocess.run(["open", str(video_path)])


if __name__ == "__main__":
    main()
