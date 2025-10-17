from typing import Optional, List, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from src.models.reweighting_module.configuration_module import ReweightAttentionConfig


class ReweightAttentionModule(nn.Module):

    def __init__(self, config: ReweightAttentionConfig):
        self.config = config
        super().__init__()

        self.head_dim = config.head_dim
        self.rank_dim = config.rank_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_proj_a = nn.Linear(
            config.num_attention_heads * self.head_dim, config.num_attention_heads * self.rank_dim,
            bias=config.attention_bias
        )
        self.q_proj_b = nn.Linear(
            config.num_attention_heads * self.rank_dim, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj_a = nn.Linear(
            config.num_attention_heads * self.head_dim, config.num_attention_heads * self.rank_dim,
            bias=config.attention_bias
        )
        self.k_proj_b = nn.Linear(
            config.num_attention_heads * self.rank_dim, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.image_token_id = self.config.image_token_id
        self.assistant_token_ids = self.config.assistant_token_ids
        self.implementation_type = config.implementation_type

        # Learnable scaling parameter for reweighting strength
        self.alpha = nn.Parameter(torch.randn(1) * self.config.alpha_std)

    def _get_block_boundaries(self, input_ids: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """
        Determine block boundaries based on image and assistant token positions for each sample.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            List of block lists, one per batch sample. Each block list contains (start, end) tuples.
        """
        batch_size, seq_len = input_ids.shape
        all_blocks = []

        for batch_idx in range(batch_size):
            boundaries = []
            ids = input_ids[batch_idx]

            # Find image tokens
            image_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                img_start = int(image_positions[0])
                img_end = int(image_positions[-1])

                # Add blocks: [0, img_start-1], [img_start, img_end], [img_end+1, ...]
                if img_start > 0:
                    boundaries.append((0, img_start))
                boundaries.append((img_start, img_end + 1))
                start_after_img = img_end + 1
            else:
                start_after_img = 0

            # Find assistant token sequence
            ids_list = ids.tolist()
            assistant_seq = self.assistant_token_ids
            asst_end = None
            for i in range(len(ids_list) - len(assistant_seq) + 1):
                if ids_list[i:i + len(assistant_seq)] == assistant_seq:
                    asst_end = i + len(assistant_seq) - 1  # Last assistant token position
                    break

            if asst_end is not None:
                # Add block from after image to assistant_end (inclusive)
                if start_after_img < asst_end + 1:
                    boundaries.append((start_after_img, asst_end + 1))
                # Add block from after assistant to end
                if asst_end + 1 < seq_len:
                    boundaries.append((asst_end + 1, seq_len))
            else:
                # No assistant tokens, add remaining as one block
                if start_after_img < seq_len:
                    boundaries.append((start_after_img, seq_len))

            # If no special tokens at all, just use entire sequence
            if len(boundaries) == 0:
                boundaries.append((0, seq_len))

            all_blocks.append(boundaries)

        return all_blocks

    def _pool_blocks(
            self,
            attn_weights: torch.Tensor,
            all_block_boundaries: List[List[Tuple[int, int]]],
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool attention weights for each block along the key dimension using matrix operations.

        Args:
            attn_weights: (batch_size, num_heads, query_length, key_length)
            all_block_boundaries: List of block lists, one per batch sample
            attention_mask: (batch_size, 1, query_length, key_length) - causal mask with 0=attend, -inf=mask

        Returns:
            Pooled scores per block: (batch_size, num_heads, query_length, max_num_blocks)
        """
        batch_size, num_heads, query_length, key_length = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype

        # Find max number of blocks across batch
        max_num_blocks = max(len(blocks) for blocks in all_block_boundaries)

        # Process each batch sample
        all_block_scores = []
        for batch_idx, block_boundaries in enumerate(all_block_boundaries):
            num_blocks = len(block_boundaries)

            # Create block mask: (num_blocks, key_length)
            block_indices = torch.arange(key_length, device=device).unsqueeze(0)  # (1, key_length)
            block_starts = torch.tensor([start for start, _ in block_boundaries], device=device).unsqueeze(
                1
                )  # (num_blocks, 1)
            block_ends = torch.tensor([end for _, end in block_boundaries], device=device).unsqueeze(
                1
                )  # (num_blocks, 1)
            block_mask = ((block_indices >= block_starts) & (block_indices < block_ends)).to(
                dtype
                )  # (num_blocks, key_length)

            # Extract attention for this batch sample: (num_heads, query_length, key_length)
            sample_attn = attn_weights[batch_idx]

            # Extract causal mask for this batch sample: (1, query_length, key_length)
            sample_causal_mask = None
            if attention_mask is not None:
                sample_causal_mask = attention_mask[batch_idx, 0]  # (query_length, key_length)

            if self.implementation_type == "mean_pool":
                # For mean pooling: exclude -inf positions from the average
                if sample_causal_mask is not None:
                    # Create valid mask: True where we can attend (not -inf)
                    valid_mask = ~torch.isinf(sample_causal_mask)  # (query_length, key_length)

                    # For each query position and block, compute valid count
                    # valid_mask: (Lq, Lk), block_mask: (num_blocks, Lk)
                    # Result: (Lq, num_blocks) - count of valid positions per block per query
                    valid_counts = torch.matmul(valid_mask.to(dtype), block_mask.T)  # (Lq, num_blocks)

                    # Mask out -inf positions in attention weights
                    masked_attn = sample_attn.masked_fill(~valid_mask.unsqueeze(0), 0.0)  # (H, Lq, Lk)

                    # Sum attention weights per block
                    block_sums = torch.matmul(masked_attn, block_mask.T)  # (H, Lq, num_blocks)

                    # Compute mean by dividing by valid counts (avoid division by zero)
                    valid_counts = valid_counts.clamp(min=1.0)  # Avoid division by zero
                    block_scores = block_sums / valid_counts.unsqueeze(0)  # (H, Lq, num_blocks)
                else:
                    # No causal mask: use simple mean pooling
                    block_sizes = (block_ends - block_starts).to(dtype).clamp(min=1.0)  # (num_blocks, 1) - prevent division by zero
                    normalized_block_mask = block_mask / block_sizes  # (num_blocks, key_length)
                    block_scores = torch.matmul(sample_attn, normalized_block_mask.T)  # (H, Lq, num_blocks)

            elif self.implementation_type == "max_pool":
                # For max pooling: safe masked max with empty block handling
                # Expand: (H, Lq, Lk) -> (H, Lq, 1, Lk)
                attn_expanded = sample_attn.unsqueeze(-2)
                # Expand: (num_blocks, Lk) -> (1, 1, num_blocks, Lk)
                mask_expanded = block_mask.unsqueeze(0).unsqueeze(0)

                # Use dtype-safe negative infinity
                neg_inf = torch.finfo(sample_attn.dtype).min

                # Combine block mask with causal mask
                if sample_causal_mask is not None:
                    # causal_mask: (Lq, Lk), expand to (1, Lq, 1, Lk)
                    causal_expanded = sample_causal_mask.unsqueeze(0).unsqueeze(-2)
                    # Only keep positions that are both in block and causally valid
                    valid_mask = mask_expanded.bool() & ~torch.isinf(causal_expanded)
                else:
                    valid_mask = mask_expanded.bool()

                # Mask out invalid positions: (H, Lq, num_blocks, Lk)
                masked_attn = attn_expanded.masked_fill(~valid_mask, neg_inf)

                # Max pool along key dimension: (H, Lq, num_blocks)
                block_scores = masked_attn.amax(dim=-1)

                # Handle empty blocks: replace -inf with 0 (no reweighting effect)
                has_any = valid_mask.any(dim=-1)  # (1, Lq, num_blocks)
                block_scores = torch.where(has_any.squeeze(0), block_scores, torch.zeros_like(block_scores))

            else:
                raise ValueError(f"Unknown implementation_type: {self.implementation_type}")

            # Pad to max_num_blocks if needed: (H, Lq, max_num_blocks)
            if num_blocks < max_num_blocks:
                padding = torch.zeros(num_heads, query_length, max_num_blocks - num_blocks, device=device, dtype=dtype)
                block_scores = torch.cat([block_scores, padding], dim=-1)

            all_block_scores.append(block_scores)

        # Stack along batch dimension: (batch_size, num_heads, query_length, max_num_blocks)
        return torch.stack(all_block_scores, dim=0)

    def _expand_to_full_shape(
            self,
            block_weights: torch.Tensor,
            all_block_boundaries: List[List[Tuple[int, int]]],
            shape: Tuple[int, int, int, int],
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expand block pooled scores to full attention shape using matrix multiplication.

        Args:
            block_weights: (batch_size, num_heads, query_length, max_num_blocks)
            all_block_boundaries: List of block lists, one per batch sample
            shape: Target shape (batch_size, num_heads, query_length, key_length)
            attention_mask: (batch_size, 1, query_length, key_length) - causal mask with 0=attend, -inf=mask

        Returns:
            Reweight mask: (batch_size, num_heads, query_length, key_length)
        """
        batch_size, num_heads, query_length, key_length = shape
        device = block_weights.device
        dtype = block_weights.dtype

        # Process each batch sample
        all_reweight_masks = []
        for batch_idx, block_boundaries in enumerate(all_block_boundaries):
            num_blocks = len(block_boundaries)

            # Create block mapping matrix: (num_blocks, key_length)
            block_indices = torch.arange(key_length, device=device).unsqueeze(0)  # (1, key_length)
            block_starts = torch.tensor([start for start, _ in block_boundaries], device=device).unsqueeze(
                1
                )  # (num_blocks, 1)
            block_ends = torch.tensor([end for _, end in block_boundaries], device=device).unsqueeze(
                1
                )  # (num_blocks, 1)
            block_mapping = ((block_indices >= block_starts) & (block_indices < block_ends)).to(
                dtype
                )  # (num_blocks, key_length)

            # Extract block weights for this sample: (num_heads, query_length, max_num_blocks)
            # Take only valid blocks: (num_heads, query_length, num_blocks)
            sample_weights = block_weights[batch_idx, :, :, :num_blocks]

            # Expand to full shape via matrix multiplication
            # (H, Lq, num_blocks) @ (num_blocks, Lk) -> (H, Lq, Lk)
            reweight_mask = torch.matmul(sample_weights, block_mapping)

            # Mask out causally invalid positions (set to 0 = no reweighting)
            if attention_mask is not None:
                sample_causal_mask = attention_mask[batch_idx, 0]  # (query_length, key_length)
                # Where causal_mask is -inf, set reweight_mask to 0
                reweight_mask = reweight_mask.masked_fill(torch.isinf(sample_causal_mask).unsqueeze(0), 0.0)

            all_reweight_masks.append(reweight_mask)

        # Stack along batch dimension: (batch_size, num_heads, query_length, key_length)
        return torch.stack(all_reweight_masks, dim=0)

    def forward(
            self,
            input_ids: torch.Tensor,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
    ):

        batch_size, num_head, query_length, dim = query_states.shape
        _, _, key_length, _ = key_states.shape

        # Q
        query_states = query_states.transpose(1, 2).reshape(batch_size, query_length, -1)
        query_states = self.q_proj_b(self.q_proj_a(query_states))
        query_states = query_states.view(batch_size, query_length, num_head, dim).transpose(1, 2)

        # K
        key_states = key_states.transpose(1, 2).reshape(batch_size, key_length, -1)
        key_states = self.k_proj_b(self.k_proj_a(key_states))
        key_states = key_states.view(batch_size, key_length, num_head, dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)

        # Compute raw attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Get block boundaries
        block_boundaries = self._get_block_boundaries(input_ids)

        # Pool attention weights per block (with causal masking)
        block_weights = self._pool_blocks(attn_weights, block_boundaries, attention_mask)

        # Expand to full attention shape (with causal masking)
        reweight_mask = self._expand_to_full_shape(
            block_weights,
            block_boundaries,
            (batch_size, num_head, query_length, key_length),
            attention_mask
        )

        # Center reweight_mask by subtracting mean (no std normalization)
        if attention_mask is not None:
            valid_mask = ~torch.isinf(attention_mask[:, 0, :, :key_length])  # (batch_size, query_length, key_length)
            valid_mask = valid_mask.unsqueeze(1)  # (batch_size, 1, query_length, key_length)

            # Mask out invalid positions once
            reweight_mask = reweight_mask.masked_fill(~valid_mask, 0.0)

            # Count valid positions per query
            valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1, Lq, 1)

            # Compute mean over valid positions only
            mean = reweight_mask.sum(dim=-1, keepdim=True) / valid_counts  # (B, H, Lq, 1)

            # Center by subtracting mean
            reweight_mask = reweight_mask - mean

            # Re-mask invalid positions to 0 (since mean subtraction affects them)
            reweight_mask = reweight_mask.masked_fill(~valid_mask, 0.0)
        else:
            # Simple mean centering without masking
            mean = reweight_mask.mean(dim=-1, keepdim=True)
            reweight_mask = reweight_mask - mean

        # Clip reweight_mask to prevent extreme values
        reweight_mask = torch.clamp(reweight_mask, min=-10.0, max=10.0)

        # Scale by learnable alpha parameter
        reweight_mask = self.alpha * reweight_mask

        # Final clipping after scaling
        reweight_mask = torch.clamp(reweight_mask, min=-5.0, max=5.0)

        return reweight_mask
