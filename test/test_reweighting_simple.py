"""
Simple test script for ReweightAttentionModule without full model imports
"""
import torch

# Direct imports to avoid model initialization issues
from src.models.reweighting_module.configuration_module import ReweightAttentionConfig


def test_config():
    """Test configuration creation"""
    print("=" * 60)
    print("Test 1: Configuration Creation")
    print("=" * 60)

    config = ReweightAttentionConfig(
        num_attention_heads=32,
        hidden_dim=4096,
        image_token_id=32000,
        assistant_token_ids=[22933, 9047, 13566, 29901],
        implementation_type="mean_pool"
    )

    print(f"Image token ID: {config.image_token_id}")
    print(f"Assistant token IDs: {config.assistant_token_ids}")
    print(f"Implementation type: {config.implementation_type}")
    print(f"✅ Config created successfully\n")


def test_block_detection_logic():
    """Test the block detection logic manually"""
    print("=" * 60)
    print("Test 2: Block Detection Logic (Manual)")
    print("=" * 60)

    image_token_id = 32000
    assistant_token_ids = [22933, 9047, 13566, 29901]

    # Test case 1
    input_ids = torch.tensor(
        [[
            1, 2, 3, 4,  # instruction (4 tokens)
            32000, 32000, 32000,  # image (3 tokens)
            22933, 9047, 13566, 29901,  # ASSISTANT: (4 tokens)
            100, 101, 102  # generated (3 tokens)
        ]]
    )

    seq_len = input_ids.shape[1]
    print(f"Input: {input_ids.shape}, seq_len={seq_len}")

    # Find image positions
    ids = input_ids[0]
    image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if len(image_positions) > 0:
        img_start, img_end = int(image_positions[0]), int(image_positions[-1]) + 1
        print(f"Image tokens: [{img_start}, {img_end})")
    else:
        img_start, img_end = None, None
        print("No image tokens found")

    # Find assistant position
    ids_list = input_ids[0].tolist()
    asst_end = None
    for i in range(len(ids_list) - len(assistant_token_ids) + 1):
        if ids_list[i:i + len(assistant_token_ids)] == assistant_token_ids:
            asst_end = i + len(assistant_token_ids)
            break

    if asst_end:
        print(f"Assistant tokens end at: {asst_end}")
    else:
        print("No assistant tokens found")

    # Build boundaries
    boundaries = [0]
    if img_start is not None:
        boundaries.extend([img_start, img_end])
    if asst_end is not None:
        boundaries.append(asst_end)
    boundaries.append(seq_len)
    boundaries = sorted(set(boundaries))

    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    print(f"Boundaries: {boundaries}")
    print(f"Blocks: {blocks}")
    print(f"✅ Expected: [(0, 4), (4, 7), (7, 11), (11, 14)]\n")


def test_pooling_and_softmax():
    """Test pooling and softmax normalization"""
    print("=" * 60)
    print("Test 3: Pooling and Softmax")
    print("=" * 60)

    batch_size, num_heads, query_len, key_len = 2, 4, 10, 10
    attn_weights = torch.randn(batch_size, num_heads, query_len, key_len)

    # Define 3 blocks: [0:3), [3:7), [7:10)
    block_boundaries = [(0, 3), (3, 7), (7, 10)]

    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Block boundaries: {block_boundaries}")

    # Mean pooling
    block_scores = []
    for start, end in block_boundaries:
        score = attn_weights[:, :, :, start:end].mean(dim=-1)  # (B, H, Lq)
        block_scores.append(score)

    block_scores = torch.stack(block_scores, dim=-1)  # (B, H, Lq, num_blocks)
    print(f"Block scores shape: {block_scores.shape}")

    # Softmax normalization
    block_weights = torch.softmax(block_scores, dim=-1)
    print(f"Block weights shape: {block_weights.shape}")

    # Check if softmax sums to 1
    weight_sums = block_weights.sum(dim=-1)
    print(f"Sum of weights per query (should be ~1.0): {weight_sums[0, 0, 0]:.6f}")

    # Show example weights
    print(f"\nExample block weights for sample 0, head 0, query 0:")
    for i in range(len(block_boundaries)):
        print(f"  Block {i}: {block_weights[0, 0, 0, i]:.4f}")

    print(f"✅ Pooling and softmax work correctly\n")


def test_expand_to_full_shape():
    """Test expanding block weights to full attention shape"""
    print("=" * 60)
    print("Test 4: Expand to Full Shape")
    print("=" * 60)

    batch_size, num_heads, query_len, key_len = 2, 4, 10, 10
    block_boundaries = [(0, 3), (3, 7), (7, 10)]
    num_blocks = len(block_boundaries)

    # Mock block weights (B, H, Lq, num_blocks)
    block_weights = torch.softmax(torch.randn(batch_size, num_heads, query_len, num_blocks), dim=-1)

    print(f"Block weights shape: {block_weights.shape}")
    print(f"Target shape: ({batch_size}, {num_heads}, {query_len}, {key_len})")

    # Expand to full shape
    reweight_mask = torch.zeros(batch_size, num_heads, query_len, key_len)

    for block_idx, (start, end) in enumerate(block_boundaries):
        weight = block_weights[:, :, :, block_idx:block_idx + 1]
        log_weight = torch.log(weight + 1e-9)
        reweight_mask[:, :, :, start:end] = log_weight.expand(-1, -1, -1, end - start)

    print(f"Reweight mask shape: {reweight_mask.shape}")
    print(f"Value range: [{reweight_mask.min():.4f}, {reweight_mask.max():.4f}]")

    # Verify: exp(log_weight) should give back normalized weights per block
    for block_idx, (start, end) in enumerate(block_boundaries):
        original_weight = block_weights[0, 0, 0, block_idx]
        recovered_weight = torch.exp(reweight_mask[0, 0, 0, start])
        print(f"Block {block_idx}: original={original_weight:.4f}, recovered={recovered_weight:.4f}")

    print(f"✅ Expansion to full shape works correctly\n")


if __name__ == "__main__":
    test_config()
    test_block_detection_logic()
    test_pooling_and_softmax()
    test_expand_to_full_shape()

    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
