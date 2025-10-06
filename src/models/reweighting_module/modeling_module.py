from typing import Optional, List, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from src.models.reweighting_module.configuration_module import ReweightAttentionConfig


class ReweightAttentionModule(nn.Module):

    def __init__(self, config: ReweightAttentionConfig):
        self.config = config
        super().__init__()

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

    def _find_image_token_positions(self, input_ids: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        Find the start and end positions of image tokens in the sequence.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            (start, end) tuple or None if no image token found
        """
        # Work with first sample in batch (assume same structure across batch)
        ids = input_ids[0]
        image_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            return None

        return int(image_positions[0]), int(image_positions[-1]) + 1

    def _find_assistant_end_position(self, input_ids: torch.Tensor) -> Optional[int]:
        """
        Find the end position of assistant token sequence.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            End position (exclusive) or None if not found
        """
        ids = input_ids[0].tolist()
        assistant_seq = self.assistant_token_ids

        # Search for the assistant token sequence
        for i in range(len(ids) - len(assistant_seq) + 1):
            if ids[i:i+len(assistant_seq)] == assistant_seq:
                return i + len(assistant_seq)

        return None

    def _get_block_boundaries(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Determine block boundaries based on image and assistant token positions.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            List of (start, end) tuples for each block
        """
        seq_len = input_ids.shape[1]
        boundaries = [0]

        # Add image token boundaries
        image_pos = self._find_image_token_positions(input_ids)
        if image_pos is not None:
            img_start, img_end = image_pos
            boundaries.extend([img_start, img_end])

        # Add assistant token boundary
        asst_end = self._find_assistant_end_position(input_ids)
        if asst_end is not None:
            boundaries.append(asst_end)

        # Add sequence end
        boundaries.append(seq_len)

        # Sort and remove duplicates
        boundaries = sorted(set(boundaries))

        # Convert to (start, end) pairs
        blocks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries) - 1)]

        return blocks

    def _pool_blocks(
            self,
            attn_weights: torch.Tensor,
            block_boundaries: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Pool attention weights for each block along the key dimension.

        Args:
            attn_weights: (batch_size, num_heads, query_length, key_length)
            block_boundaries: List of (start, end) tuples

        Returns:
            Pooled scores per block: (batch_size, num_heads, query_length, num_blocks)
        """
        batch_size, num_heads, query_length, _ = attn_weights.shape
        num_blocks = len(block_boundaries)

        block_scores = []
        for start, end in block_boundaries:
            # Extract block: (B, H, Lq, block_len)
            block_attn = attn_weights[:, :, :, start:end]

            # Pool along key dimension
            if self.implementation_type == "mean_pool":
                score = block_attn.mean(dim=-1)  # (B, H, Lq)
            elif self.implementation_type == "max_pool":
                score = block_attn.max(dim=-1)[0]  # (B, H, Lq)
            else:
                raise ValueError(f"Unknown implementation_type: {self.implementation_type}")

            block_scores.append(score)

        # Stack to (B, H, Lq, num_blocks)
        block_scores = torch.stack(block_scores, dim=-1)

        return block_scores

    def _expand_to_full_shape(
            self,
            block_weights: torch.Tensor,
            block_boundaries: List[Tuple[int, int]],
            shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Expand block pooled scores to full attention shape using matrix multiplication.

        Args:
            block_weights: (batch_size, num_heads, query_length, num_blocks)
            block_boundaries: List of (start, end) tuples
            shape: Target shape (batch_size, num_heads, query_length, key_length)

        Returns:
            Reweight mask: (batch_size, num_heads, query_length, key_length)
        """
        batch_size, num_heads, query_length, key_length = shape
        device = block_weights.device
        dtype = block_weights.dtype
        num_blocks = len(block_boundaries)

        # Create block mapping matrix: (num_blocks, key_length)
        # Each row indicates which keys belong to that block
        block_indices = torch.arange(key_length, device=device).unsqueeze(0)  # (1, key_length)
        block_starts = torch.tensor([start for start, _ in block_boundaries], device=device).unsqueeze(1)  # (num_blocks, 1)
        block_ends = torch.tensor([end for _, end in block_boundaries], device=device).unsqueeze(1)  # (num_blocks, 1)

        block_mapping = ((block_indices >= block_starts) & (block_indices < block_ends)).to(dtype)  # (num_blocks, key_length)

        # Expand block weights to full shape via matrix multiplication
        # (B, H, Lq, num_blocks) @ (num_blocks, Lk) -> (B, H, Lq, Lk)
        reweight_mask = torch.matmul(block_weights, block_mapping)

        return reweight_mask

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

        # Pool attention weights per block
        block_weights = self._pool_blocks(attn_weights, block_boundaries)

        # Expand to full attention shape (use pooled scores directly)
        reweight_mask = self._expand_to_full_shape(
            block_weights,
            block_boundaries,
            (batch_size, num_head, query_length, key_length)
        )

        return reweight_mask
