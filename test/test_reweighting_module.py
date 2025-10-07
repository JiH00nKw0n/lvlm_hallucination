"""
Test script for ReweightAttentionModule
"""
import torch
from src.models.reweighting_module.configuration_module import ReweightAttentionConfig
from src.models.reweighting_module.modeling_module import ReweightAttentionModule


def test_block_detection():
    """Test block boundary detection with different input scenarios"""
    config = ReweightAttentionConfig(
        num_attention_heads=32,
        hidden_dim=4096,
        image_token_id=32000,
        assistant_token_ids=[22933, 9047, 13566, 29901],
        implementation_type="mean_pool"
    )

    module = ReweightAttentionModule(config)

    # Test case 1: Image + Instruction + Generated
    # Format: [instruction tokens] [image tokens] [ASSISTANT:] [generated tokens]
    input_ids_1 = torch.tensor([[
        1, 2, 3, 4,  # instruction
        32000, 32000, 32000,  # image tokens
        22933, 9047, 13566, 29901,  # ASSISTANT:
        100, 101, 102  # generated
    ]])

    boundaries_1 = module._get_block_boundaries(input_ids_1)
    print("Test 1 - Image + Instruction + Generated:")
    print(f"  Input length: {input_ids_1.shape[1]}")
    print(f"  Boundaries: {boundaries_1}")
    print(f"  Expected: [(0, 4), (4, 7), (7, 11), (11, 14)]")
    print()

    # Test case 2: No image, only instruction + generated
    input_ids_2 = torch.tensor([[
        1, 2, 3, 4, 5,  # instruction
        22933, 9047, 13566, 29901,  # ASSISTANT:
        100, 101, 102  # generated
    ]])

    boundaries_2 = module._get_block_boundaries(input_ids_2)
    print("Test 2 - No image, Instruction + Generated:")
    print(f"  Input length: {input_ids_2.shape[1]}")
    print(f"  Boundaries: {boundaries_2}")
    print(f"  Expected: [(0, 9), (9, 12)]")
    print()

    # Test case 3: Only generated (no image, no instruction)
    input_ids_3 = torch.tensor([[100, 101, 102, 103, 104]])

    boundaries_3 = module._get_block_boundaries(input_ids_3)
    print("Test 3 - Only generated:")
    print(f"  Input length: {input_ids_3.shape[1]}")
    print(f"  Boundaries: {boundaries_3}")
    print(f"  Expected: [(0, 5)]")
    print()


def test_full_forward():
    """Test full forward pass with mock attention tensors"""
    config = ReweightAttentionConfig(
        num_attention_heads=4,  # Smaller for testing
        hidden_dim=128,
        num_key_value_heads=4,
        head_dim=32,
        rank_dim=8,
        image_token_id=32000,
        assistant_token_ids=[22933, 9047, 13566, 29901],
        implementation_type="mean_pool"
    )

    module = ReweightAttentionModule(config)

    # Create mock inputs
    batch_size = 2
    seq_len = 14
    num_heads = 4
    head_dim = 32

    input_ids = torch.tensor([[
        1, 2, 3, 4,  # instruction (4 tokens)
        32000, 32000, 32000,  # image (3 tokens)
        22933, 9047, 13566, 29901,  # ASSISTANT: (4 tokens)
        100, 101, 102  # generated (3 tokens)
    ]] * batch_size)

    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Create causal attention mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')),
        diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # Forward pass
    reweight_mask = module(input_ids, query_states, key_states, attention_mask)

    print("Test 4 - Full forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Query shape: {query_states.shape}")
    print(f"  Key shape: {key_states.shape}")
    print(f"  Reweight mask shape: {reweight_mask.shape}")
    print(f"  Expected shape: ({batch_size}, {num_heads}, {seq_len}, {seq_len})")
    print(f"  Reweight mask dtype: {reweight_mask.dtype}")
    print(f"  Reweight mask range: [{reweight_mask.min():.4f}, {reweight_mask.max():.4f}]")
    print()

    # Check if values are in log space (should be negative or zero)
    print(f"  Mean value: {reweight_mask.mean():.4f}")
    print(f"  Std value: {reweight_mask.std():.4f}")
    print()

    # Visualize block weights for first query position
    print("  Block weights (softmax probabilities) for first sample, first head, first query:")
    boundaries = module._get_block_boundaries(input_ids)
    print(f"  Boundaries: {boundaries}")

    # Extract attention weights for first query position
    query_states_proj = module.q_proj_b(module.q_proj_a(
        query_states[0].transpose(0, 1).reshape(seq_len, -1)
    )).view(seq_len, num_heads, head_dim).transpose(0, 1)

    key_states_proj = module.k_proj_b(module.k_proj_a(
        key_states[0].transpose(0, 1).reshape(seq_len, -1)
    )).view(seq_len, num_heads, head_dim).transpose(0, 1)

    attn_w = torch.matmul(query_states_proj, key_states_proj.transpose(1, 2)) * module.scaling
    attn_w = attn_w + attention_mask[0, 0]

    block_scores = module._pool_blocks(attn_w.unsqueeze(0), boundaries)
    block_probs = torch.softmax(block_scores, dim=-1)

    for q_pos in [0, 7, 13]:  # instruction, assistant end, last token
        print(f"  Query position {q_pos}:")
        for i, (s, e) in enumerate(boundaries):
            prob = block_probs[0, 0, q_pos, i].item()
            print(f"    Block {i} [{s}:{e}): {prob:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ReweightAttentionModule")
    print("=" * 60)
    print()

    test_block_detection()
    test_full_forward()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
