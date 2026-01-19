"""
VTI: Visual-Textual Intervention

PCA-derived steering vectors for both textual (LLM) and visual (vision encoder)
hidden states.

Reference:
    - VTI/vti_utils/llm_layers.py:9-32 (VTILayer with fixed 0.1 scaling)
    - VTI/vti_utils/utils.py:203-230 (Textual direction: last-token per layer)
    - VTI/vti_utils/utils.py:309-325 (Visual direction: per-token PCA)

Key Implementation Notes:
    1. Textual VTI: Applied to LLM decoder MLP outputs via hooks
    2. Visual VTI: Applied to vision encoder via hooks (per-token direction)
    3. Fixed 0.1 scaling factor for steering (Reference: llm_layers.py:28)
    4. Alpha is applied inside the hook AFTER normalization, not before

Formula (Reference: VTI/vti_utils/llm_layers.py:28):
    y = lam * F.normalize(vti_direction, dim=-1)
    x_new = F.normalize(F.normalize(x) + 0.1 * y, dim=-1) * ||x||

Supports: LLaVA, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMitigator, MitigatorConfig, ModelHelper


class VTIMitigator(BaseMitigator):
    """
    VTI: Visual-Textual Intervention.

    Reference: VTI/vti_utils/llm_layers.py

    Args:
        model: The VLM model
        model_type: llava, llava_next, qwen2_vl, qwen2_5_vl
        textual_vti: Textual intervention tensor (num_layers, hidden_dim)
        visual_vti: Visual intervention tensor (num_layers, num_tokens, hidden_dim)
        target_layers: Layers to apply textual intervention
        alpha_text: Strength for textual intervention (scalar or list, default: 0.8)
        alpha_image: Strength for visual intervention (scalar or list, default: 0.9)
    """

    name: str = "vti"
    FIXED_SCALE = 0.1  # VTI uses fixed 0.1 scaling (Reference: llm_layers.py:28)

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llava",
        textual_vti: Optional[torch.Tensor] = None,
        visual_vti: Optional[torch.Tensor] = None,
        target_layers: Optional[List[int]] = None,
        alpha_text: Union[float, List[float]] = 0.8,
        alpha_image: Union[float, List[float]] = 0.9,
        **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)

        num_layers = self._get_num_layers()
        self.target_layers = target_layers or list(range(num_layers))
        self.alpha_text = alpha_text
        self.alpha_image = alpha_image

        # Store VTI vectors (moved to device in setup)
        self.textual_vti: Dict[int, torch.Tensor] = {}
        self.visual_vti: Optional[torch.Tensor] = None

        if textual_vti is not None:
            self._set_textual_vti(textual_vti)
        if visual_vti is not None:
            self.visual_vti = visual_vti

        # Hook handles
        self._textual_hooks: List = []
        self._visual_hooks: List = []

    def _set_textual_vti(self, vti: torch.Tensor) -> None:
        """
        Set textual VTI directions.

        Note: Alpha is applied in the hook, not here.
        """
        for i in self.target_layers:
            if i < vti.shape[0]:
                self.textual_vti[i] = vti[i].clone()

    @staticmethod
    def _expand_lam(lam: Union[float, List[float]], num_dirs: int) -> List[float]:
        if isinstance(lam, (int, float)):
            return [float(lam)] * num_dirs
        lam_list = list(lam)
        if len(lam_list) == 1 and num_dirs > 1:
            lam_list = lam_list * num_dirs
        if len(lam_list) != num_dirs:
            raise ValueError(f"alpha length mismatch: expected {num_dirs}, got {len(lam_list)}")
        return lam_list

    def _create_textual_hook(self, layer_idx: int):
        """
        Create forward hook for textual VTI steering.

        Reference: VTI/vti_utils/llm_layers.py:16-30

        Formula (matches reference exactly):
            y = alpha * F.normalize(vti_direction, dim=-1)
            x_new = F.normalize(F.normalize(x) + 0.1 * y, dim=-1) * ||x||
        """
        def hook_fn(module, input, output):
            if layer_idx not in self.textual_vti:
                return output

            x = output
            if isinstance(output, tuple):
                x = output[0]

            device = x.device
            vti = self.textual_vti[layer_idx].to(device)
            x_float = x.float()

            # Compute original norm (Reference: line 18)
            norm = torch.norm(x_float, p=2, dim=-1, keepdim=True)

            # Compute y = lam * normalize(vti) (Reference: lines 19-27)
            if vti.dim() == 1:
                vti_list = [vti.unsqueeze(0)]
            elif vti.dim() == 2:
                vti_list = [vti_dir for vti_dir in vti]
            else:
                raise ValueError(f"Unexpected textual_vti shape: {vti.shape}")

            lam_list = self._expand_lam(self.alpha_text, len(vti_list))
            y = 0
            for i, vti_dir in enumerate(vti_list):
                vti_dir = vti_dir.to(device)
                if vti_dir.dim() == 1:
                    vti_dir = vti_dir.unsqueeze(0)
                if x.size(1) < 2:
                    y = y + lam_list[i] * F.normalize(vti_dir, dim=-1).repeat(1, x.shape[1], 1)
                else:
                    y = y + lam_list[i] * F.normalize(vti_dir, dim=-1)
            y = y / len(vti_list)

            # Apply VTI: x_new = normalize(normalize(x) + 0.1 * y) * ||x||
            # Reference: llm_layers.py:28
            x_norm = F.normalize(x_float, p=2, dim=-1)
            x_new = F.normalize(x_norm + self.FIXED_SCALE * y, p=2, dim=-1) * norm

            x_new = x_new.half()

            if isinstance(output, tuple):
                return (x_new,) + output[1:]
            return x_new

        return hook_fn

    @staticmethod
    def _get_nested_attr(obj: nn.Module, attr_path: str) -> nn.Module:
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    @staticmethod
    def _find_longest_modulelist(model: nn.Module, path: str = "") -> Tuple[str, int]:
        longest_path = path
        longest_len = 0
        for name, child in model.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > longest_len:
                longest_len = len(child)
                longest_path = f"{path}.{name}" if path else name
            child_path, child_len = VTIMitigator._find_longest_modulelist(
                child, f"{path}.{name}" if path else name
            )
            if child_len > longest_len:
                longest_len = child_len
                longest_path = child_path
        return longest_path, longest_len

    def _get_vision_layers(self, vision_model: nn.Module) -> nn.ModuleList:
        longest_path, longest_len = self._find_longest_modulelist(vision_model)
        if not longest_path or longest_len == 0:
            raise ValueError("Cannot find vision layers ModuleList")
        return self._get_nested_attr(vision_model, longest_path)

    def _create_visual_hook(self, layer_idx: int):
        """
        Create forward hook for visual VTI steering.

        Applied to vision encoder MLP outputs per layer.

        Reference: VTI/vti_utils/llm_layers.py:16-30
        Visual VTI has shape (num_layers, num_tokens, hidden_dim) for per-token direction.
        """
        def hook_fn(module, input, output):
            if self.visual_vti is None:
                return output

            # Output is typically [B, num_patches, hidden_dim]
            x = output
            if isinstance(output, tuple):
                x = output[0]
            if hasattr(output, 'last_hidden_state'):
                x = output.last_hidden_state

            device = x.device
            # visual_vti shape: [num_layers, num_tokens, hidden_dim] or [num_layers, num_dirs, num_tokens, hidden_dim]
            vti_layer = self.visual_vti[layer_idx].to(device)
            x_float = x.float()

            # Compute original norm
            norm = torch.norm(x_float, p=2, dim=-1, keepdim=True)

            if vti_layer.dim() == 2:
                vti_list = [vti_layer]
            elif vti_layer.dim() == 3:
                vti_list = [vti_dir for vti_dir in vti_layer]
            else:
                raise ValueError(f"Unexpected visual_vti shape: {vti_layer.shape}")

            lam_list = self._expand_lam(self.alpha_image, len(vti_list))
            y = 0
            for i, vti_dir in enumerate(vti_list):
                vti_dir = vti_dir.to(device)
                if vti_dir.dim() == 1:
                    vti_dir = vti_dir.unsqueeze(0)
                if x.size(1) < 2:
                    y = y + lam_list[i] * F.normalize(vti_dir, dim=-1).repeat(1, x.shape[1], 1)
                else:
                    y = y + lam_list[i] * F.normalize(vti_dir, dim=-1)
            y = y / len(vti_list)

            x_norm = F.normalize(x_float, p=2, dim=-1)
            x_new = F.normalize(x_norm + self.FIXED_SCALE * y, p=2, dim=-1) * norm

            x_new = x_new.half()

            if isinstance(output, tuple):
                return (x_new,) + output[1:]
            if hasattr(output, 'last_hidden_state'):
                output.last_hidden_state = x_new
                return output
            return x_new

        return hook_fn

    def setup(self) -> None:
        """Register forward hooks on MLP layers and vision encoder."""
        device = next(self.model.parameters()).device
        layers = self._get_layers()

        # Move VTI tensors to device
        for i in list(self.textual_vti.keys()):
            self.textual_vti[i] = self.textual_vti[i].to(device)
        if self.visual_vti is not None:
            self.visual_vti = self.visual_vti.to(device)

        # Textual hooks on MLP outputs
        for layer_idx in self.target_layers:
            if layer_idx in self.textual_vti and layer_idx < len(layers):
                mlp = ModelHelper.get_mlp_module(layers[layer_idx])
                handle = mlp.register_forward_hook(self._create_textual_hook(layer_idx))
                self._textual_hooks.append(handle)

        # Visual hooks on vision encoder MLP outputs
        if self.visual_vti is not None:
            try:
                vision_encoder = self._get_vision_encoder()
                vision_model = vision_encoder.vision_model if hasattr(vision_encoder, 'vision_model') else vision_encoder
                vision_layers = self._get_vision_layers(vision_model)
                max_layers = min(len(vision_layers), self.visual_vti.shape[0])
                for layer_idx in range(max_layers):
                    mlp = ModelHelper.get_mlp_module(vision_layers[layer_idx])
                    handle = mlp.register_forward_hook(self._create_visual_hook(layer_idx))
                    self._visual_hooks.append(handle)
            except ValueError:
                pass

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self._textual_hooks:
            handle.remove()
        self._textual_hooks.clear()

        for handle in self._visual_hooks:
            handle.remove()
        self._visual_hooks.clear()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with VTI steering using standard model.generate()."""
        gen_kwargs = {
            'max_new_tokens': self.config.max_new_tokens,
            'do_sample': self.config.do_sample,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
        }
        gen_kwargs.update(kwargs)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    @classmethod
    def compute_textual_vti(
        cls,
        model: nn.Module,
        hallucinating_inputs: List[Dict],
        correct_inputs: List[Dict],
        model_type: str = "llava",
        rank: int = 1,
    ) -> torch.Tensor:
        """
        Compute textual VTI direction from hallucinating/correct pairs.

        Reference: VTI/vti_utils/utils.py:203-230

        Direction: h_correct - h_hallucinating (for each layer, last token)

        Args:
            model: The VLM model
            hallucinating_inputs: List of inputs for hallucinating responses
            correct_inputs: List of inputs for correct responses
            model_type: Model type string
            rank: PCA rank (default: 1)

        Returns:
            vti: Tensor of shape (num_layers, hidden_dim)
        """
        def _svd_flip(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            max_abs_cols = torch.argmax(torch.abs(u), 1)
            i = torch.arange(u.shape[2], device=u.device)
            max_abs_cols = max_abs_cols.unsqueeze(-1)
            signs = torch.sign(torch.gather(u, 1, max_abs_cols))
            u = u * signs
            v = v * signs.view(v.shape[0], -1, 1)
            return u, v

        def _fit_pca(x: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim != 3:
                raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
            _, n, d = x.size()
            d = min(n_components, d)
            mean = x.mean(1, keepdim=True)
            z = x - mean
            u, _, vh = torch.linalg.svd(z, full_matrices=False)
            vt = vh
            u, vt = _svd_flip(u, vt)
            components = vt[:, :d]
            return components, mean

        all_diffs = []
        num_layers = None
        hidden_dim = None

        with torch.no_grad():
            for hall, correct in zip(hallucinating_inputs, correct_inputs):
                hall_out = model(**hall, output_hidden_states=True)
                correct_out = model(**correct, output_hidden_states=True)

                # Skip embedding layer (index 0)
                hall_hidden = hall_out.hidden_states[1:]
                correct_hidden = correct_out.hidden_states[1:]

                if num_layers is None:
                    num_layers = len(hall_hidden)
                    hidden_dim = hall_hidden[0].shape[-1]

                # Extract LAST TOKEN per layer (Reference: utils.py:205)
                # Shape: [num_layers, hidden_dim]
                hall_last = torch.stack([h[:, -1].squeeze(0) for h in hall_hidden], dim=0)
                correct_last = torch.stack([h[:, -1].squeeze(0) for h in correct_hidden], dim=0)

                # Flatten: [num_layers * hidden_dim]
                hall_flat = hall_last.view(-1).cpu()
                correct_flat = correct_last.view(-1).cpu()

                # Direction: correct - hallucinating (Reference: utils.py:219)
                diff = correct_flat - hall_flat
                all_diffs.append(diff)

        # Stack and fit PCA (Reference: utils.py:223-228)
        stacked = torch.stack(all_diffs).float()
        components, mean = _fit_pca(stacked, rank)
        direction = (components.sum(dim=1, keepdim=True) + mean).mean(0)
        vti = direction.view(num_layers, hidden_dim).float()

        return vti

    @classmethod
    def compute_visual_vti(
        cls,
        model: nn.Module,
        clean_images: List[Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]],
        model_type: str = "llava",
        mask_ratio: float = 0.99,
        rank: int = 1,
        patch_size: int = 14,
        image_grid_thw: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute visual VTI direction from clean vs masked images.

        Reference: VTI/vti_utils/utils.py:309-325

        Uses PER-TOKEN PCA (not global) - each token position has its own direction.

        Args:
            model: The VLM model
            clean_images: List of clean tensors or (masked_trials, clean) pairs
            model_type: Model type string
            mask_ratio: Fraction of patches to mask (default: 0.99)
            rank: PCA rank (default: 1)
            patch_size: Vision encoder patch size (default: 14)
            image_grid_thw: Qwen2-VL grid metadata (required for Qwen2-VL/2.5-VL)

        Returns:
            visual_vti: Tensor of shape (num_layers, num_tokens, hidden_dim)
        """
        def _svd_flip(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            max_abs_cols = torch.argmax(torch.abs(u), 1)
            i = torch.arange(u.shape[2], device=u.device)
            max_abs_cols = max_abs_cols.unsqueeze(-1)
            signs = torch.sign(torch.gather(u, 1, max_abs_cols))
            u = u * signs
            v = v * signs.view(v.shape[0], -1, 1)
            return u, v

        def _fit_pca(x: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim != 3:
                raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
            _, n, d = x.size()
            d = min(n_components, d)
            mean = x.mean(1, keepdim=True)
            z = x - mean
            u, _, vh = torch.linalg.svd(z, full_matrices=False)
            vt = vh
            u, vt = _svd_flip(u, vt)
            components = vt[:, :d]
            return components, mean

        model_type = ModelHelper.normalize_model_type(model_type)

        try:
            vision_encoder = ModelHelper.get_vision_encoder(model, model_type)
        except ValueError:
            raise ValueError(f"Cannot get vision encoder for model_type: {model_type}")

        # Get the actual vision model
        if hasattr(vision_encoder, 'vision_model'):
            vision_model = vision_encoder.vision_model
        else:
            vision_model = vision_encoder

        def mask_patches(
            img: torch.Tensor,
            indices: List[int],
            ps: int,
            mean_val: torch.Tensor,
        ) -> torch.Tensor:
            """Mask patches by setting to per-channel mean value."""
            masked = img.clone()
            h_patches = img.shape[1] // ps
            w_patches = img.shape[2] // ps

            for idx in indices:
                row = idx // w_patches
                col = idx % w_patches
                sy, sx = row * ps, col * ps
                masked[:, sy:sy+ps, sx:sx+ps] = mean_val.expand(-1, ps, ps)
            return masked

        def mask_flat_patches(img: torch.Tensor, indices: List[int]) -> torch.Tensor:
            """Mask patchified inputs (Qwen-style) by replacing rows with mean patch."""
            flat = img.view(img.shape[0], -1)
            mean_patch = flat.mean(dim=0, keepdim=True)
            masked = flat.clone()
            masked[indices] = mean_patch
            return masked.view_as(img)

        def _get_grid_thw(idx: int) -> Optional[torch.Tensor]:
            if image_grid_thw is None:
                return None
            if isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.ndim == 1:
                    return image_grid_thw.unsqueeze(0)
                return image_grid_thw[idx:idx+1]
            grid = image_grid_thw[idx]
            if isinstance(grid, torch.Tensor):
                return grid.unsqueeze(0) if grid.ndim == 1 else grid
            return torch.tensor(grid).unsqueeze(0)

        def _supports_output_hidden_states(module: nn.Module) -> bool:
            return hasattr(module, "config") and hasattr(module.config, "output_hidden_states")

        def _collect_hidden_states(
            img: torch.Tensor,
            grid_thw: Optional[torch.Tensor],
        ) -> List[torch.Tensor]:
            if _supports_output_hidden_states(vision_model):
                out = vision_model(
                    img,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return list(out.hidden_states)

            hidden_states: List[torch.Tensor] = []
            hooks = []

            def _capture(_module, _inp, out):
                out_tensor = out
                if isinstance(out_tensor, tuple):
                    out_tensor = out_tensor[0]
                if hasattr(out_tensor, "last_hidden_state"):
                    out_tensor = out_tensor.last_hidden_state
                if out_tensor.dim() == 2:
                    out_tensor = out_tensor.unsqueeze(0)
                hidden_states.append(out_tensor.detach().cpu())

            if hasattr(vision_model, "patch_embed"):
                hooks.append(vision_model.patch_embed.register_forward_hook(_capture))

            layers = None
            if hasattr(vision_model, "blocks") and isinstance(vision_model.blocks, nn.ModuleList):
                layers = vision_model.blocks
            else:
                longest_path, longest_len = cls._find_longest_modulelist(vision_model)
                if longest_path and longest_len > 0:
                    layers = cls._get_nested_attr(vision_model, longest_path)

            if layers is None:
                raise ValueError("Cannot locate vision layers for hidden state capture.")

            for layer in layers:
                hooks.append(layer.register_forward_hook(_capture))

            if grid_thw is not None:
                _ = vision_model(img, grid_thw=grid_thw)
            else:
                _ = vision_model(img)

            for hook in hooks:
                hook.remove()

            if not hidden_states:
                raise ValueError("No vision hidden states captured.")
            return hidden_states

        def _average_hidden_states(states_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
            if len(states_list) == 1:
                return states_list[0]
            num_layers = len(states_list[0])
            averaged: List[torch.Tensor] = []
            for layer_idx in range(num_layers):
                stacked = torch.stack([states[layer_idx] for states in states_list], dim=0)
                averaged.append(stacked.mean(dim=0))
            return averaged

        def _make_masked(
            img: torch.Tensor,
            total_tokens: int,
            to_mask: int,
        ) -> torch.Tensor:
            mask_indices = torch.randperm(total_tokens)[:to_mask].tolist()
            if img.dim() == 3:
                mean_val = img.mean(dim=(1, 2), keepdim=True)
                return mask_patches(img, mask_indices, patch_size, mean_val)
            return mask_flat_patches(img, mask_indices)

        # Collect hidden states for each image pair
        all_diffs = []  # Will be [num_samples, num_tokens, num_layers * hidden_dim]
        n_layers = None
        n_tokens = None
        feat_dim = None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if model_type in ("qwen2_vl", "qwen2_5_vl") and image_grid_thw is None:
            raise ValueError("Qwen2-VL/2.5-VL requires image_grid_thw for visual VTI.")

        with torch.no_grad():
            for idx, item in enumerate(clean_images):
                grid_thw = _get_grid_thw(idx)
                if grid_thw is not None:
                    grid_thw = grid_thw.to(device)

                masked_trials: Optional[List[torch.Tensor]] = None
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], list):
                    masked_trials = item[0]
                    img = item[1]
                else:
                    img = item

                img = img.to(device, dtype)
                if img.dim() == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)

                if img.dim() == 3:
                    # img: [C, H, W]
                    h, w = img.shape[1], img.shape[2]
                    h_patches = h // patch_size
                    w_patches = w // patch_size
                    total_patches = h_patches * w_patches
                    num_to_mask = int(mask_ratio * total_patches)
                    img_for_model = img.unsqueeze(0)
                elif img.dim() in (2, 5):
                    total_patches = img.shape[0]
                    num_to_mask = int(mask_ratio * total_patches)
                    img_for_model = img
                else:
                    raise ValueError(f"Unsupported image tensor shape: {tuple(img.shape)}")

                clean_hs = _collect_hidden_states(img_for_model, grid_thw)

                if masked_trials is not None:
                    masked_states = []
                    for masked in masked_trials:
                        masked = masked.to(device, dtype)
                        if masked.dim() == 4 and masked.shape[0] == 1:
                            masked = masked.squeeze(0)
                        masked_for_model = masked.unsqueeze(0) if masked.dim() == 3 else masked
                        masked_states.append(_collect_hidden_states(masked_for_model, grid_thw))
                    masked_hs = _average_hidden_states(masked_states)
                else:
                    masked_img = _make_masked(img, total_patches, num_to_mask)
                    masked_for_model = masked_img.unsqueeze(0) if masked_img.dim() == 3 else masked_img
                    masked_hs = _collect_hidden_states(masked_for_model, grid_thw)

                if n_layers is None:
                    n_layers = len(clean_hs)
                    # Skip CLS token (Reference: utils.py:298 uses [:,:])
                    n_tokens = clean_hs[0].shape[1]
                    feat_dim = clean_hs[0].shape[-1]

                # Stack all layers: [num_layers, 1, num_tokens, hidden_dim]
                clean_stack = torch.stack([h.cpu() for h in clean_hs], dim=0)
                masked_stack = torch.stack([h.cpu() for h in masked_hs], dim=0)

                # Reshape to [num_tokens, num_layers * hidden_dim]
                # Reference: utils.py:318 h.reshape(n_tokens, -1)
                clean_flat = clean_stack.squeeze(1).permute(1, 0, 2).reshape(n_tokens, -1)
                masked_flat = masked_stack.squeeze(1).permute(1, 0, 2).reshape(n_tokens, -1)

                # Direction: masked - clean (Reference: utils.py:318)
                diff = masked_flat - clean_flat  # [num_tokens, num_layers * hidden_dim]
                all_diffs.append(diff)

        # Stack: [num_tokens, num_samples, num_layers * hidden_dim]
        # Reference: utils.py:321 torch.stack(hidden_states_all, dim=1)
        stacked = torch.stack(all_diffs, dim=1).float()

        components, mean = _fit_pca(stacked, rank)
        direction = (components.sum(dim=1, keepdim=True) + mean).mean(1)
        visual_vti = direction.view(n_layers, n_tokens, feat_dim).float()

        return visual_vti
