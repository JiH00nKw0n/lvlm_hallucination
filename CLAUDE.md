# LVLM Hallucination Mitigation

## Coding Style

1. **Functional**: `map()`, list comprehension over for-loops
2. **Async**: `asyncio.Semaphore(32)` + `tqdm_asyncio.gather()`
3. **Types**: Always annotate with `Optional[T]`, `Union[T1, T2]`
4. **Dataclass**: Use `__post_init__` for dependent fields
5. **Errors**: Validate early, clear messages with type info
6. **Efficiency**: Broadcasting > loops, `.clamp(min=1.0)` for div-by-zero
7. **Registry**: `@registry.register_*('Name')` pattern
8. **Naming**: `_private`, `CONSTANT`, `snake_case`, `PascalCase`

## Environment

Python 3.12, PyTorch 2.8.0 (CUDA 12.4), Transformers 4.57.6, TRL 0.23.1

---

## Decoding Methods (src/decoding/)

### Overview

| Method | Intervention | Formula/Key |
|--------|-------------|-------------|
| **VCD** | Logits | `(1+α)*logits - α*logits_noised`, cutoff: `log(β) + max(logits)` |
| **AvisC** | Embeddings + Logits | Blind tokens: `attn < mean + λ*std`, mask embeddings |
| **VISTA** | MLP output | `x = norm(norm(x) + λ*norm(vsv)) * ||x||` |
| **VTI** | MLP output | Same as VISTA with fixed `0.1` scaling |
| **Middle Layers** | Attention (pre-softmax) | `attn[:,-1,img] += α * mean(abs(attn[:,-1,img]))` |
| **FarSight** | Attention (pre-softmax) | `W = QK^T/√d + P_upper_triangular`, no KV cache |
| **Deco** | Logits (early exit) | `logits + α * max_prob * early_logits`, top-k/p candidates |
| **OPERA** | Beam search + rollback | Attention penalty, rollback on summary token pattern |
| **Octopus** | Dynamic (VCD/AvisC/M3ID) | Classifier selects strategy per token, 4 KV caches |

### Quick Start

```python
from src.decoding import get_mitigator

with get_mitigator('vcd', model, model_type="llava", alpha=1.0) as m:
    output = m.generate(input_ids, pixel_values=pixel_values)
```

### Supported Models

| model_type | Layers | Norm/lm_head |
|------------|--------|--------------|
| `llava` / `llava_next` | `language_model.model.layers` | `language_model.model.norm`, `language_model.lm_head` |
| `qwen2_vl` / `qwen2_5_vl` | `model.language_model.layers` | `model.language_model.norm`, `lm_head` |

---

## Reference Files

| Method | File | Key Lines |
|--------|------|-----------|
| VCD | `VCD/vcd_utils/vcd_sample.py` | 141-159 |
| AvisC | `AvisC/avisc_utils/avisc_sample.py` | 160-179, 206-208 |
| VISTA | `VISTA/llm_layers.py` | 7-34, 132-150 |
| VTI | `VTI/vti_utils/utils.py` | 203-230, 309-325 |
| Middle Layers | `middle_layers.../modify_attention.py` | 77-88 |
| FarSight | `FarSight/Shell/farsight_patch.py` | 7-126 |
| Deco | `Deco/transformers/.../utils.py` | 2660-2682 |
| OPERA | `OPERA/transformers-4.29.2/.../utils.py` | 3419-3449, 3457-3545 |
| Octopus | `Octopus/.../train_token_amber.py` | 132-256, 286-317 |

---

## Key Implementation Notes

1. **VISTA/VTI**: Last-token per layer for PCA direction
2. **Qwen2-VL**: 3D multimodal RoPE, `cache_position` required, image tokens via `<|vision_start|>`/`<|vision_end|>`
3. **FarSight**: `use_cache=False` mandatory; LLaMA path uses num_heads QKV; apply only in eval; Qwen path supported
4. **Middle Layers**: LLaMA path uses num_heads QKV + attention_mask shape/clamp; `img_start_idx/img_end_idx` can be injected
5. **Deco**: pass `early_exit_layers` into model forward when supported; apply logits_processor/logits_warper before sampling
6. **OPERA**: Beam search only (`num_beams > 1`)
7. **Octopus**: 4 separate KV caches (base/AvisC/VCD/M3ID), DPO loss training
