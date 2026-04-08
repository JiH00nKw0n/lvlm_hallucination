"""
Synthetic Theory Simplified: 2D Multimodal SAE Experiment.

Validates hypotheses about SAE behavior on a minimal 2D toy problem:
  - H1: Capacity bottleneck (m vs distinct directions)
  - H2: Split dictionary (standard SAE splits shared features)
  - H3: Group-sparse support merging (L_{2,1} loss)
  - H4: Trace alignment energy matching
  - H5: Over-regularization collapse
  - H6: Bias effect

Geometry (hardcoded):
  phi_1 (image-only)  = [cos(120°), sin(120°)]
  psi_3 (text-only)   = [cos(-120°), sin(-120°)]
  phi_2 (shared, img)  = [cos(-θ), sin(-θ)]
  psi_2 (shared, txt)  = [cos(θ), sin(θ)]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FFMpegWriter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Geometry                                                           #
# ------------------------------------------------------------------ #


def make_gt_atoms(theta_deg: float) -> dict[str, np.ndarray]:
    """Return GT dictionary atoms as unit vectors in R^2.

    Args:
        theta_deg: angle (degrees) of shared features from [1,0].

    Returns:
        Dict with keys phi_1, phi_2, psi_2, psi_3 each shape (2,).
    """
    theta = math.radians(theta_deg)
    return {
        "phi_1": np.array([math.cos(math.radians(120)), math.sin(math.radians(120))]),
        "psi_3": np.array([math.cos(math.radians(-120)), math.sin(math.radians(-120))]),
        "phi_2": np.array([math.cos(-theta), math.sin(-theta)]),
        "psi_2": np.array([math.cos(theta), math.sin(theta)]),
    }


# ------------------------------------------------------------------ #
#  Data Generation                                                    #
# ------------------------------------------------------------------ #


def make_paired_data(
    theta_deg: float,
    num_samples: int,
    seed: int,
    sparsity: float = 0.99,
    min_active: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate paired (image, text) data with Bernoulli per-concept activation.

    Each concept is independently active with probability (1 - sparsity).
    At least min_active concepts are enforced per sample.

      Concept 0 (image-only): x += c_0 * phi_1,  y += 0
      Concept 1 (shared):     x += c_1 * phi_2,  y += c_1 * psi_2
      Concept 2 (text-only):  x += 0,             y += c_2 * psi_3

    Returns:
        (img_data, txt_data) each shape (num_samples, 2).
    """
    atoms = make_gt_atoms(theta_deg)
    p = 1.0 - sparsity  # per-concept activation probability

    # Enumerate all 2^3 = 8 binary z-vectors, exclude all-zero
    z_combos = []
    probs = []
    for z0 in range(2):
        for z1 in range(2):
            for z2 in range(2):
                if z0 == 0 and z1 == 0 and z2 == 0:
                    continue
                z_combos.append((z0, z1, z2))
                # Binomial probability: p^(#active) * (1-p)^(#inactive)
                n_active = z0 + z1 + z2
                probs.append(p ** n_active * (1 - p) ** (3 - n_active))

    # Normalize probabilities (condition on at least one active)
    probs = np.array(probs)
    probs /= probs.sum()

    # Compute exact counts per combination
    counts = np.round(probs * num_samples).astype(int)
    # Adjust rounding to match num_samples exactly
    diff = num_samples - counts.sum()
    if diff > 0:
        # Add to largest probability combos
        order = np.argsort(-probs)
        for i in range(diff):
            counts[order[i % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(probs)
        for i in range(-diff):
            if counts[order[i % len(order)]] > 0:
                counts[order[i % len(order)]] -= 1

    # Generate data deterministically
    img_parts: list[np.ndarray] = []
    txt_parts: list[np.ndarray] = []

    for (z0, z1, z2), cnt in zip(z_combos, counts):
        if cnt == 0:
            continue
        x = z0 * atoms["phi_1"] + z1 * atoms["phi_2"]
        y = z1 * atoms["psi_2"] + z2 * atoms["psi_3"]
        img_parts.append(np.tile(x, (cnt, 1)))
        txt_parts.append(np.tile(y, (cnt, 1)))

    img_all = np.concatenate(img_parts, axis=0)
    txt_all = np.concatenate(txt_parts, axis=0)

    # Shuffle with seed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(img_all))
    img_all = img_all[perm]
    txt_all = txt_all[perm]

    return (
        torch.from_numpy(img_all).float(),
        torch.from_numpy(txt_all).float(),
    )


# ------------------------------------------------------------------ #
#  Simple SAE                                                         #
# ------------------------------------------------------------------ #


def make_optimal_decoder_init(theta_deg: float, latent_size: int) -> np.ndarray:
    """Return theoretically optimal W_dec init based on GT geometry.

    m=1: [1,0] (shared midpoint)
    m=2: [1,0] + phi_1
    m=3: [1,0] + phi_1 + psi_3
    m=4: phi_1 + phi_2 + psi_2 + psi_3 (all 4 GT atoms)
    m=5: all 4 GT + [1,0]

    Returns: (latent_size, 2) array of unit vectors.
    """
    atoms = make_gt_atoms(theta_deg)
    shared_mid = np.array([1.0, 0.0])

    candidates = {
        1: [shared_mid],
        2: [shared_mid, atoms["phi_1"]],
        3: [shared_mid, atoms["phi_1"], atoms["psi_3"]],
        4: [atoms["phi_1"], atoms["phi_2"], atoms["psi_2"], atoms["psi_3"]],
        5: [atoms["phi_1"], atoms["phi_2"], atoms["psi_2"], atoms["psi_3"], shared_mid],
    }
    vecs = candidates.get(latent_size, candidates[4][:latent_size])
    result = np.stack(vecs)
    result = result / (np.linalg.norm(result, axis=1, keepdims=True) + 1e-12)
    return result


def make_optimal_decoder_inits_expB(theta_deg: float) -> list[np.ndarray]:
    """Return multiple candidate optimal inits for Exp B (m=4 fixed).

    Different λ values favor different effective capacity levels:
    1. eff-m=4 (split): [phi_1, phi_2, psi_2, psi_3] — optimal for λ≈0
    2. eff-m=3 (merged): [phi_1, [1,0], psi_3, [1,0]] — optimal for medium λ
    3. eff-m=1 (collapsed): [[1,0], phi_1, psi_3, [1,0]] — optimal for large λ

    Returns: list of (4, 2) arrays.
    """
    atoms = make_gt_atoms(theta_deg)
    shared_mid = np.array([1.0, 0.0])

    def _norm(vecs: list[np.ndarray]) -> np.ndarray:
        r = np.stack(vecs)
        return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)

    return [
        # eff-m=4: all 4 GT atoms (split dictionary)
        _norm([atoms["phi_1"], atoms["phi_2"], atoms["psi_2"], atoms["psi_3"]]),
        # eff-m=3: shared merged to [1,0]
        _norm([atoms["phi_1"], shared_mid, atoms["psi_3"], shared_mid]),
        # eff-m=1: fully collapsed to shared direction
        _norm([shared_mid, shared_mid, shared_mid, shared_mid]),
    ]


class SimpleSAE(nn.Module):
    """Minimal Top-K SAE for 2D toy experiments."""

    def __init__(
        self,
        hidden_size: int = 2,
        latent_size: int = 4,
        k: int = 1,
        use_bias: bool = True,
        init_W_dec: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        self.use_bias = use_bias

        if init_W_dec is not None:
            init_t = torch.from_numpy(init_W_dec).float()
            self.W_enc = nn.Parameter(init_t.clone())  # encoder = decoder direction
            self.W_dec = nn.Parameter(init_t.clone())
        else:
            self.W_enc = nn.Parameter(torch.randn(latent_size, hidden_size) * 0.1)
            self.W_dec = nn.Parameter(torch.randn(latent_size, hidden_size) * 0.1)

        if use_bias:
            self.b_enc = nn.Parameter(torch.zeros(latent_size))
            self.b_dec = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("b_enc", torch.zeros(latent_size))
            self.register_buffer("b_dec", torch.zeros(hidden_size))

        # Normalize decoder to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x → sparse latent z (post-topk, dense vector)."""
        sae_in = x - self.b_dec
        pre_acts = F.relu(sae_in @ self.W_enc.T + self.b_enc)
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        sparse_z = torch.zeros_like(pre_acts)
        sparse_z.scatter_(-1, top_idx, top_vals)
        return sparse_z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sparse_z, x_hat, recon_loss)."""
        z = self.encode_sparse(x)
        x_hat = self.decode(z)
        recon_loss = (x - x_hat).pow(2).mean()
        return z, x_hat, recon_loss

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self) -> None:
        if self.W_dec.grad is None:
            return
        # Project out component parallel to decoder direction
        dec_normed = F.normalize(self.W_dec.data, dim=1)
        parallel = (self.W_dec.grad * dec_normed).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel * dec_normed


# ------------------------------------------------------------------ #
#  Alignment Losses                                                   #
# ------------------------------------------------------------------ #


def group_sparse_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """L_{2,1} norm over paired sparse codes.

    L_gs = (1/B) * sum_i sqrt(z_img_i^2 + z_txt_i^2)   summed over latent dims.

    Encourages joint sparsity: same support for paired samples.
    """
    # z_img, z_txt: (B, latent_size)
    return torch.sqrt(z_img.pow(2) + z_txt.pow(2) + 1e-12).sum(dim=-1).mean()


def trace_alignment_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Trace-based alignment loss: -1/b * Tr(Z_img_norm @ Z_txt_norm^T).

    Per the paper (Eq.1), codes are ℓ₂-normalized before computing trace.
    This bounds the loss and prevents activation magnitude explosion.
    """
    b = z_img.shape[0]
    # ℓ₂-normalize each sample's code vector
    z_img_norm = F.normalize(z_img, p=2, dim=-1)
    z_txt_norm = F.normalize(z_txt, p=2, dim=-1)
    # Tr(Z_img @ Z_txt^T) = sum of row-wise dot products
    trace_val = (z_img_norm * z_txt_norm).sum()
    return -trace_val / b


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #


@dataclass
class RunMetrics:
    theta_deg: float
    latent_size: int
    aux_loss_type: str  # "none", "group_sparse", "trace"
    aux_lambda: float
    use_bias: bool
    seed: int
    final_recon_loss: float = 0.0
    final_aux_loss: float = 0.0
    final_total_loss: float = 0.0
    # GT recovery: fraction of GT atoms matched by a decoder atom (cosine > threshold)
    gt_recovery: float = 0.0
    # Mean inner product with GT
    mip: float = 0.0
    # Per-atom best cosine with each GT
    atom_gt_cosines: dict[str, float] = field(default_factory=dict)
    # Shared alignment: cosine between decoder atoms matched to phi_2 and psi_2
    shared_alignment_cos: float = 0.0
    num_dead_atoms: int = 0


def compute_metrics(
    model: SimpleSAE,
    theta_deg: float,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute GT recovery metrics from trained SAE decoder."""
    atoms = make_gt_atoms(theta_deg)
    W_dec = model.W_dec.detach().cpu().numpy()  # (latent_size, 2)

    # Normalize
    dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-12)

    gt_names = ["phi_1", "phi_2", "psi_2", "psi_3"]
    gt_vecs = np.stack([atoms[n] for n in gt_names])  # (4, 2)
    gt_norm = gt_vecs / (np.linalg.norm(gt_vecs, axis=1, keepdims=True) + 1e-12)

    # Cosine similarity matrix: (latent_size, 4)
    cos_matrix = np.abs(dec_norm @ gt_norm.T)

    # Per GT: best matching decoder atom cosine
    best_per_gt = cos_matrix.max(axis=0)  # (4,)
    atom_gt_cosines = {gt_names[i]: float(best_per_gt[i]) for i in range(4)}

    gt_recovery = float((best_per_gt > threshold).mean())
    mip = float(best_per_gt.mean())

    # Shared alignment: cosine between atoms best-matching phi_2 and psi_2
    best_idx_phi2 = int(cos_matrix[:, 1].argmax())
    best_idx_psi2 = int(cos_matrix[:, 2].argmax())
    if best_idx_phi2 == best_idx_psi2:
        shared_cos = 1.0
    else:
        shared_cos = float(np.abs(dec_norm[best_idx_phi2] @ dec_norm[best_idx_psi2]))

    # Dead atoms: atoms not close to any GT (max cosine < 0.3)
    max_per_atom = cos_matrix.max(axis=1)  # (latent_size,)
    num_dead = int((max_per_atom < 0.3).sum())

    return {
        "gt_recovery": gt_recovery,
        "mip": mip,
        "atom_gt_cosines": atom_gt_cosines,
        "shared_alignment_cos": shared_cos,
        "num_dead_atoms": num_dead,
    }


# ------------------------------------------------------------------ #
#  Video Recorder                                                     #
# ------------------------------------------------------------------ #


class VideoRecorder:
    """Records training frames to mp4 + saves final frame as png."""

    # Colors for GT atoms
    GT_COLORS = {
        "phi_1": "#2196F3",   # blue (image-only)
        "phi_2": "#4CAF50",   # green (shared, img side)
        "psi_2": "#8BC34A",   # light green (shared, txt side)
        "psi_3": "#F44336",   # red (text-only)
    }
    GT_LABELS = {
        "phi_1": "φ₁ (img)",
        "phi_2": "φ₂ (shared·img)",
        "psi_2": "ψ₂ (shared·txt)",
        "psi_3": "ψ₃ (txt)",
    }

    def __init__(
        self,
        gt_atoms: dict[str, np.ndarray],
        output_dir: str,
        viz_every: int = 5,
        fps: int = 10,
        title_extra: str = "",
    ):
        self.gt_atoms = gt_atoms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_every = viz_every
        self.fps = fps
        self.title_extra = title_extra

        self.video_path = self.output_dir / "video.mp4"
        self.png_path = self.output_dir / "final_frame.png"

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.writer = FFMpegWriter(fps=self.fps, metadata={"title": "SAE 2D"})
        self.writer.setup(self.fig, str(self.video_path), dpi=150)

    def capture(self, model: SimpleSAE, step: int, loss: Optional[float] = None) -> None:
        if step > 0 and step % self.viz_every != 0:
            return
        self._draw_frame(model, step, loss)
        self.writer.grab_frame()

    def finalize(self, model: SimpleSAE, step: int, loss: Optional[float] = None) -> None:
        # Final frame → video + png
        self._draw_frame(model, step, loss)
        self.writer.grab_frame()
        self.writer.finish()
        self.fig.savefig(str(self.png_path), dpi=150, bbox_inches="tight")
        plt.close(self.fig)
        logger.info("Video: %s | PNG: %s", self.video_path, self.png_path)

    def _draw_frame(self, model: SimpleSAE, step: int, loss: Optional[float]) -> None:
        ax = self.ax
        ax.clear()

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.5, alpha=0.3)

        # Dotted reference line at [1,0]
        ax.plot([0, 1.3], [0, 0], "--", color="gray", lw=1, alpha=0.4)
        ax.text(1.35, 0, "[1,0]", fontsize=8, color="gray", va="center")

        # GT feature arrows
        for name, vec in self.gt_atoms.items():
            color = self.GT_COLORS[name]
            label = self.GT_LABELS[name]
            ax.annotate(
                "", xy=(vec[0], vec[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5),
                zorder=3,
            )
            ax.text(
                vec[0] * 1.2, vec[1] * 1.2, label,
                fontsize=8, fontweight="bold", color=color,
                ha="center", va="center", zorder=4,
            )

        # SAE decoder atoms (clamp norm to 1 for display)
        if model is not None:
            W_dec = model.W_dec.detach().cpu().numpy()
            norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
            W_dec_vis = W_dec / np.maximum(norms, 1.0)  # cap to unit norm
            sae_colors = plt.get_cmap("Set2")(np.linspace(0, 1, W_dec_vis.shape[0]))
            for j in range(W_dec_vis.shape[0]):
                dx, dy = W_dec_vis[j]
                ax.annotate(
                    "", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=sae_colors[j], lw=2.0, ls="--"),
                    zorder=5,
                )
                ax.text(
                    dx * 1.15, dy * 1.15, f"D{j}",
                    fontsize=8, color=sae_colors[j],
                    ha="center", va="center", zorder=6,
                )

        title = f"Step {step}"
        if loss is not None:
            title += f"  |  loss={loss:.5f}"
        if self.title_extra:
            title += f"  |  {self.title_extra}"
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #


def train_sae(
    img_data: torch.Tensor,
    txt_data: torch.Tensor,
    theta_deg: float,
    latent_size: int,
    eval_img: Optional[torch.Tensor] = None,
    eval_txt: Optional[torch.Tensor] = None,
    k: int = 1,
    use_bias: bool = True,
    aux_loss_type: str = "none",
    aux_lambda: float = 0.0,
    normalize_decoder: bool = False,
    init_W_dec: Optional[np.ndarray] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    viz_every: int = 5,
    fps: int = 10,
) -> tuple[SimpleSAE, RunMetrics]:
    """Train a SimpleSAE and return (model, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    gt_atoms = make_gt_atoms(theta_deg)

    model = SimpleSAE(
        hidden_size=2,
        latent_size=latent_size,
        k=k,
        use_bias=use_bias,
        init_W_dec=init_W_dec,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # Compute total steps for scheduler
    n = img_data.shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1, total_steps // 10)  # 10% warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # For exp B: we need paired batches. For exp A: we can mix.
    # Unified approach: always iterate paired data. For recon loss, concat and compute on both.
    img_data = img_data.to(device)
    txt_data = txt_data.to(device)

    # Video recorder
    title_extra = f"θ={theta_deg}° m={latent_size}"
    if aux_loss_type != "none":
        title_extra += f" {aux_loss_type} λ={aux_lambda}"
    title_extra += " bias" if use_bias else " no-bias"

    recorder = None
    if output_dir is not None:
        recorder = VideoRecorder(
            gt_atoms=gt_atoms,
            output_dir=output_dir,
            viz_every=viz_every,
            fps=fps,
            title_extra=title_extra,
        )
        recorder.capture(model, step=0, loss=None)

    global_step = 0
    last_loss = None

    for _epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_img = img_data[idx]
            x_txt = txt_data[idx]

            # Concatenated reconstruction loss (on both modalities)
            x_all = torch.cat([x_img, x_txt], dim=0)  # (2B, 2)
            _z_all, _x_hat_all, recon_loss = model(x_all)

            # Auxiliary loss on paired data
            aux_loss = torch.tensor(0.0, device=device)
            if aux_loss_type != "none" and aux_lambda > 0:
                z_img = model.encode_sparse(x_img)
                z_txt = model.encode_sparse(x_txt)
                if aux_loss_type == "group_sparse":
                    aux_loss = group_sparse_loss(z_img, z_txt)
                elif aux_loss_type == "trace":
                    aux_loss = trace_alignment_loss(z_img, z_txt)

            total_loss = recon_loss + aux_lambda * aux_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if normalize_decoder:
                model.remove_gradient_parallel_to_decoder()
            optimizer.step()
            scheduler.step()
            if normalize_decoder:
                model.normalize_decoder()

            global_step += 1
            last_loss = float(total_loss.item())

            if recorder is not None:
                recorder.capture(model, global_step, last_loss)

    # Final metrics on eval set (or train if no eval provided)
    e_img = eval_img.to(device) if eval_img is not None else img_data
    e_txt = eval_txt.to(device) if eval_txt is not None else txt_data
    model.eval()
    with torch.no_grad():
        e_all = torch.cat([e_img, e_txt], dim=0)
        _, _, final_recon = model(e_all)
        final_aux = torch.tensor(0.0)
        if aux_loss_type != "none" and aux_lambda > 0:
            z_img_e = model.encode_sparse(e_img)
            z_txt_e = model.encode_sparse(e_txt)
            if aux_loss_type == "group_sparse":
                final_aux = group_sparse_loss(z_img_e, z_txt_e)
            elif aux_loss_type == "trace":
                final_aux = trace_alignment_loss(z_img_e, z_txt_e)

    met_dict = compute_metrics(model, theta_deg)
    metrics = RunMetrics(
        theta_deg=theta_deg,
        latent_size=latent_size,
        aux_loss_type=aux_loss_type,
        aux_lambda=aux_lambda,
        use_bias=use_bias,
        seed=seed,
        final_recon_loss=float(final_recon.item()),
        final_aux_loss=float(final_aux.item()),
        final_total_loss=float(final_recon.item()) + aux_lambda * float(final_aux.item()),
        gt_recovery=float(met_dict["gt_recovery"]),
        mip=float(met_dict["mip"]),
        atom_gt_cosines=dict(met_dict["atom_gt_cosines"]),  # type: ignore[arg-type]
        shared_alignment_cos=float(met_dict["shared_alignment_cos"]),
        num_dead_atoms=int(met_dict["num_dead_atoms"]),
    )

    # Save video + metrics
    if recorder is not None:
        recorder.finalize(model, global_step, last_loss)

    if output_dir is not None:
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    return model, metrics


# ------------------------------------------------------------------ #
#  Experiment Runners                                                 #
# ------------------------------------------------------------------ #


def _train_best_of_seeds(
    img_data: torch.Tensor,
    txt_data: torch.Tensor,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    train_kwargs: dict,
    run_dir: str,
    num_seeds: int = 3,
    base_seed: int = 0,
    theta_deg: float = 0.0,
    latent_size: int = 4,
    is_exp_b: bool = False,
) -> RunMetrics:
    """Train with multiple init seeds + optimal init, keep lowest recon loss.

    Tries num_seeds random inits + 1 theoretically optimal init.
    Best one is retrained with video.
    """
    candidates: list[tuple[float, int, Optional[np.ndarray]]] = []  # (loss, seed, init_W_dec)

    eval_kwargs = {"eval_img": eval_img, "eval_txt": eval_txt}

    # Random seeds
    for i in range(num_seeds):
        s = base_seed + i
        _, met = train_sae(
            img_data, txt_data,
            **{**train_kwargs, **eval_kwargs, "seed": s, "output_dir": None, "init_W_dec": None},
        )
        logger.info("    seed=%d recon=%.5f", s, met.final_recon_loss)
        candidates.append((met.final_recon_loss, s, None))

    # Optimal inits
    if is_exp_b:
        opt_inits = make_optimal_decoder_inits_expB(theta_deg)
        for idx, opt_dec in enumerate(opt_inits):
            _, met = train_sae(
                img_data, txt_data,
                **{**train_kwargs, **eval_kwargs, "seed": base_seed, "output_dir": None, "init_W_dec": opt_dec},
            )
            labels = ["eff-m=4(split)", "eff-m=3(merged)", "eff-m=1(collapsed)"]
            logger.info("    optimal[%s] recon=%.5f", labels[idx], met.final_recon_loss)
            candidates.append((met.final_recon_loss, base_seed, opt_dec))
    else:
        opt_dec = make_optimal_decoder_init(theta_deg, latent_size)
        _, met = train_sae(
            img_data, txt_data,
            **{**train_kwargs, **eval_kwargs, "seed": base_seed, "output_dir": None, "init_W_dec": opt_dec},
        )
        logger.info("    optimal_init recon=%.5f", met.final_recon_loss)
        candidates.append((met.final_recon_loss, base_seed, opt_dec))

    # Pick best
    candidates.sort(key=lambda x: x[0])
    best_loss, best_seed, best_init = candidates[0]
    tag = "optimal_init" if best_init is not None else f"seed={best_seed}"
    logger.info("    → best: %s (recon=%.5f), retraining with video", tag, best_loss)

    _, met = train_sae(
        img_data, txt_data,
        **{**train_kwargs, **eval_kwargs, "seed": best_seed, "output_dir": run_dir, "init_W_dec": best_init},
    )
    return met


def run_experiment_A(args: argparse.Namespace) -> list[RunMetrics]:
    """Exp A: SAE latent size (m) sweep."""
    results = []
    theta_values = [float(x) for x in args.theta_values.split(",")]
    m_values = [int(x) for x in args.m_values.split(",")]

    for bias in ([True, False] if not args.single_bias else [not args.no_bias]):
        bias_tag = "bias" if bias else "nobias"
        for theta in theta_values:
            img_data, txt_data = make_paired_data(theta, args.num_train, args.seed, sparsity=args.sparsity)
            eval_img, eval_txt = make_paired_data(theta, args.num_eval, args.seed + 99999, sparsity=args.sparsity)
            for m in m_values:
                run_dir = Path(args.output_dir) / f"expA_{bias_tag}" / f"theta{int(theta)}_m{m}"
                logger.info("Exp A: θ=%s° m=%d bias=%s", theta, m, bias)
                train_kwargs = dict(
                    theta_deg=theta,
                    latent_size=m,
                    k=args.k,
                    use_bias=bias,
                    aux_loss_type="none",
                    aux_lambda=0.0,
                    normalize_decoder=not args.no_normalize_decoder,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    viz_every=args.viz_every,
                    fps=args.fps,
                    device=args.device,
                )
                met = _train_best_of_seeds(
                    img_data, txt_data,
                    eval_img=eval_img,
                    eval_txt=eval_txt,
                    train_kwargs=train_kwargs,
                    run_dir=str(run_dir),
                    num_seeds=args.num_init_seeds,
                    base_seed=args.seed,
                    theta_deg=theta,
                    latent_size=m,
                )
                results.append(met)
                logger.info("  → recon=%.5f mip=%.3f recovery=%.3f shared_align=%.3f",
                            met.final_recon_loss, met.mip, met.gt_recovery, met.shared_alignment_cos)
    return results


def run_experiment_B(args: argparse.Namespace, loss_type: str) -> list[RunMetrics]:
    """Exp B1/B2: alignment loss λ sweep (m=4 fixed)."""
    results = []
    theta_values = [float(x) for x in args.theta_values.split(",")]
    lambda_values = [float(x) for x in args.lambda_values.split(",")]
    tag = "B1" if loss_type == "group_sparse" else "B2"

    for bias in ([True, False] if not args.single_bias else [not args.no_bias]):
        bias_tag = "bias" if bias else "nobias"
        for theta in theta_values:
            img_data, txt_data = make_paired_data(theta, args.num_train, args.seed, sparsity=args.sparsity)
            eval_img, eval_txt = make_paired_data(theta, args.num_eval, args.seed + 99999, sparsity=args.sparsity)
            for lam in lambda_values:
                run_dir = (
                    Path(args.output_dir)
                    / f"exp{tag}_{bias_tag}"
                    / f"theta{int(theta)}_lambda{int(lam) if lam == int(lam) else lam}"
                )
                logger.info("Exp %s: θ=%s° λ=%s bias=%s", tag, theta, lam, bias)
                train_kwargs = dict(
                    theta_deg=theta,
                    latent_size=4,
                    k=args.k,
                    use_bias=bias,
                    aux_loss_type=loss_type,
                    aux_lambda=lam,
                    normalize_decoder=not args.no_normalize_decoder,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    viz_every=args.viz_every,
                    fps=args.fps,
                    device=args.device,
                )
                met = _train_best_of_seeds(
                    img_data, txt_data,
                    eval_img=eval_img,
                    eval_txt=eval_txt,
                    train_kwargs=train_kwargs,
                    run_dir=str(run_dir),
                    num_seeds=args.num_init_seeds,
                    base_seed=args.seed,
                    theta_deg=theta,
                    latent_size=4,
                    is_exp_b=True,
                )
                results.append(met)
                logger.info("  → recon=%.5f aux=%.5f total=%.5f mip=%.3f shared_align=%.3f",
                            met.final_recon_loss, met.final_aux_loss, met.final_total_loss,
                            met.mip, met.shared_alignment_cos)
    return results


# ------------------------------------------------------------------ #
#  Report Generation                                                  #
# ------------------------------------------------------------------ #


def generate_report(
    results_A: list[RunMetrics],
    results_B1: list[RunMetrics],
    results_B2: list[RunMetrics],
    output_dir: str,
) -> None:
    """Generate experiment_report.md with ALL run images inline."""
    out = Path(output_dir)
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    def img(rel_path: str, caption: str = "") -> None:
        if (out / rel_path).exists():
            w(f"![{caption or rel_path}]({rel_path})")
            w()

    w("# Synthetic Theory Simplified: 2D 실험 보고서")
    w()
    w(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w()

    # ---- Section 1: Hypotheses ----
    w("## 1. 가설")
    w()
    w("| ID | 가설 | 신뢰도 |")
    w("|-----|------|--------|")
    w("| H1 | Capacity Bottleneck: m < distinct directions → 완전 복원 불가 | ★★★ |")
    w("| H2 | Split Dictionary: standard SAE는 θ>0일 때 shared feature를 분리 | ★★★ |")
    w("| H3 | Group-Sparse Merging: L₂,₁ loss → support 공유로 atom 병합 | ★★★ |")
    w("| H4 | Trace Alignment: energy matching → 간접적 bimodal atom 유도 | ★★☆ |")
    w("| H5 | Over-Regularization: λ 과대 → reconstruction collapse | ★★★ |")
    w("| H6 | Bias Effect: bias가 평균 성분 흡수 → atom 방향 개선 | ★☆☆ |")
    w()

    # ---- Section 2: Geometry ----
    w("## 2. 실험 세팅")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w("| Embedding dim | 2 |")
    w("| φ₁ (image-only) | +120° from [1,0] |")
    w("| ψ₃ (text-only) | -120° from [1,0] |")
    w("| φ₂ (shared, img) | -θ from [1,0] |")
    w("| ψ₂ (shared, txt) | +θ from [1,0] |")
    w("| SAE activation | ReLU + Top-1 |")
    w("| Epochs | 10 (cosine annealing, 10% warmup) |")
    w()

    # ---- Helper: metrics table ----
    def metrics_table(results: list[RunMetrics], sweep_key: str) -> None:
        if not results:
            w("*No results.*")
            return
        w(f"| θ | {sweep_key} | bias | recon_loss | mip | gt_recovery | shared_align | dead |")
        w("|---|---|---|---|---|---|---|---|")
        for r in results:
            sweep_val = getattr(r, "latent_size" if sweep_key == "m" else "aux_lambda")
            w(f"| {r.theta_deg:.0f}° | {sweep_val} | {'Y' if r.use_bias else 'N'} "
              f"| {r.final_recon_loss:.5f} | {r.mip:.3f} | {r.gt_recovery:.2f} "
              f"| {r.shared_alignment_cos:.3f} | {r.num_dead_atoms} |")
        w()

    # ---- Collect sweep values from results ----
    theta_vals_A = sorted({r.theta_deg for r in results_A}) if results_A else []
    m_vals = sorted({r.latent_size for r in results_A}) if results_A else []
    theta_vals_B1 = sorted({r.theta_deg for r in results_B1}) if results_B1 else []
    lambda_vals_B1 = sorted({r.aux_lambda for r in results_B1}) if results_B1 else []
    theta_vals_B2 = sorted({r.theta_deg for r in results_B2}) if results_B2 else []
    lambda_vals_B2 = sorted({r.aux_lambda for r in results_B2}) if results_B2 else []

    # ---- Section 3: Exp A ----
    w("## 3. Exp A: Capacity Sweep (m)")
    w()
    for bias_tag in ["bias", "nobias"]:
        w(f"### Exp A — {bias_tag}")
        w()
        for theta in theta_vals_A:
            w(f"#### θ = {theta:.0f}°")
            w()
            for m in m_vals:
                img_path = f"expA_{bias_tag}/theta{int(theta)}_m{m}/final_frame.png"
                img(img_path, f"θ={theta:.0f}° m={m} {bias_tag}")
            w()
    w("### Exp A 전체 결과 테이블")
    w()
    metrics_table(results_A, "m")

    # ---- Section 4: Exp B1 ----
    w("## 4. Exp B1: Group-Sparse λ Sweep (m=4)")
    w()
    for bias_tag in ["bias", "nobias"]:
        w(f"### Exp B1 — {bias_tag}")
        w()
        for theta in theta_vals_B1:
            w(f"#### θ = {theta:.0f}°")
            w()
            for lam in lambda_vals_B1:
                lam_str = f"{int(lam)}" if lam == int(lam) else f"{lam}"
                img_path = f"expB1_{bias_tag}/theta{int(theta)}_lambda{lam_str}/final_frame.png"
                img(img_path, f"θ={theta:.0f}° λ={lam_str} {bias_tag}")
            w()
    w("### Exp B1 전체 결과 테이블")
    w()
    metrics_table(results_B1, "λ")

    # ---- Section 5: Exp B2 ----
    w("## 5. Exp B2: Trace Alignment λ Sweep (m=4)")
    w()
    for bias_tag in ["bias", "nobias"]:
        w(f"### Exp B2 — {bias_tag}")
        w()
        for theta in theta_vals_B2:
            w(f"#### θ = {theta:.0f}°")
            w()
            for lam in lambda_vals_B2:
                lam_str = f"{int(lam)}" if lam == int(lam) else f"{lam}"
                img_path = f"expB2_{bias_tag}/theta{int(theta)}_lambda{lam_str}/final_frame.png"
                img(img_path, f"θ={theta:.0f}° λ={lam_str} {bias_tag}")
            w()
    w("### Exp B2 전체 결과 테이블")
    w()
    metrics_table(results_B2, "λ")

    # ---- Section 6: Summary ----
    w("## 6. 종합 결론")
    w()
    w("| 가설 | 판정 | 핵심 증거 |")
    w("|------|------|----------|")
    w("| H1 | (실험 후 기입) | |")
    w("| H2 | (실험 후 기입) | |")
    w("| H3 | (실험 후 기입) | |")
    w("| H4 | (실험 후 기입) | |")
    w("| H5 | (실험 후 기입) | |")
    w("| H6 | (실험 후 기입) | |")
    w()

    report_path = out / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", report_path)


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Theory Simplified: 2D Multimodal SAE")

    p.add_argument("--experiment", type=str, default="all",
                   choices=["A", "B1", "B2", "all", "report"],
                   help="Which experiment to run ('report' = regenerate report from existing metrics)")

    # Sweep values
    p.add_argument("--theta-values", type=str, default="0,10,20,30,90")
    p.add_argument("--m-values", type=str, default="1,2,3,4,5")
    p.add_argument("--lambda-values", type=str,
                   default="0,0.25,0.5,1,2,4,8,16,32,64,128,256,512")

    # SAE config
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--no-bias", action="store_true", help="Disable learnable bias")
    p.add_argument("--no-normalize-decoder", action="store_true",
                   help="Disable decoder unit-norm constraint + gradient projection")
    p.add_argument("--single-bias", action="store_true",
                   help="Run only one bias variant (controlled by --no-bias)")

    # Data & training
    p.add_argument("--sparsity", type=float, default=0.99,
                   help="Per-concept deactivation probability (0.99 = 1%% chance each concept is active)")
    p.add_argument("--num-train", type=int, default=10000)
    p.add_argument("--num-eval", type=int, default=10000)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-init-seeds", type=int, default=3,
                   help="Number of weight init seeds to try per config (best recon loss wins)")

    # Video
    p.add_argument("--viz-every", type=int, default=5)
    p.add_argument("--fps", type=int, default=10)

    # Output
    p.add_argument("--output-dir", type=str, default="outputs/synthetic_theory_simplified")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])

    return p.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def collect_all_metrics(output_dir: str) -> tuple[list[RunMetrics], list[RunMetrics], list[RunMetrics]]:
    """Scan output_dir for all metrics.json and collect into A/B1/B2 lists."""
    out = Path(output_dir)
    results_A: list[RunMetrics] = []
    results_B1: list[RunMetrics] = []
    results_B2: list[RunMetrics] = []

    for metrics_file in sorted(out.rglob("metrics.json")):
        with open(metrics_file) as f:
            d = json.load(f)
        met = RunMetrics(**{k: v for k, v in d.items() if k in RunMetrics.__dataclass_fields__})
        rel = str(metrics_file.relative_to(out))
        if rel.startswith("expA"):
            results_A.append(met)
        elif rel.startswith("expB1"):
            results_B1.append(met)
        elif rel.startswith("expB2"):
            results_B2.append(met)

    logger.info("Collected: %d A, %d B1, %d B2 runs", len(results_A), len(results_B1), len(results_B2))
    return results_A, results_B1, results_B2


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.device = resolve_device(args.device)
    logger.info("Device: %s", args.device)

    # Report-only mode: collect existing metrics and regenerate report
    if args.experiment == "report":
        results_A, results_B1, results_B2 = collect_all_metrics(args.output_dir)
        generate_report(results_A, results_B1, results_B2, args.output_dir)
        return

    results_A: list[RunMetrics] = []
    results_B1: list[RunMetrics] = []
    results_B2: list[RunMetrics] = []

    if args.experiment in ("A", "all"):
        logger.info("=" * 60)
        logger.info("Running Experiment A: Capacity Sweep")
        logger.info("=" * 60)
        results_A = run_experiment_A(args)

    if args.experiment in ("B1", "all"):
        logger.info("=" * 60)
        logger.info("Running Experiment B1: Group-Sparse λ Sweep")
        logger.info("=" * 60)
        results_B1 = run_experiment_B(args, "group_sparse")

    if args.experiment in ("B2", "all"):
        logger.info("=" * 60)
        logger.info("Running Experiment B2: Trace Alignment λ Sweep")
        logger.info("=" * 60)
        results_B2 = run_experiment_B(args, "trace")

    # Save per-invocation results JSON
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
        },
        "expA": [asdict(r) for r in results_A],
        "expB1": [asdict(r) for r in results_B1],
        "expB2": [asdict(r) for r in results_B2],
    }
    results_path = Path(args.output_dir) / "all_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All results: %s", results_path)


if __name__ == "__main__":
    main()
