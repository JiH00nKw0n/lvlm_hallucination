# Decoding Methods Validation

각 방법론의 Reference 구현과 현재 구현을 비교 검증한 문서입니다.

---

## SAVE (Sparse Autoencoder Steering)

### Reference
- **파일**: `SAVE/collect_sae_activations.py` (SAE 구조/weight 로딩)
- **파일**: `SAVE/identify_features.py` (feature indices 산출)
- **파일**: `SAVE/chair_steer.py` (steering hook)

```python
# chair_steer.py 핵심 로직
direction_i = sae.fc2.weight[:, idx]
faith_vec = F.normalize(direction_i, dim=0)

x_new = x_new + alpha * faith_vec_b
x_new = x_new - alpha * hal_vec_b
```

### 현재 구현
- **파일**: `src/decoding/save.py`
- **파일**: `src/decoding/save_utils/sae.py`
- **파일**: `src/decoding/save_utils/io.py`

```python
# save.py 핵심 로직
sae = SAE().to(self.model.device).half()
sae.load_state_dict(remove_module_prefix(state))

faith_indices, hal_indices = load_feature_indices(feature_path)
x_new = x_new + alpha * faith_vec
x_new = x_new - alpha * hal_vec
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| SAE 구조 | 2-layer MLP (fc1/relu/fc2) | 동일 | ✅ |
| 가중치 로딩 | `.pkl` load + prefix 제거 | 동일 | ✅ |
| Faith 벡터 | fc2 weight 방향 | 동일 | ✅ |
| Hall 벡터 | fc2 weight 방향 | 동일 | ✅ |
| Steering 방식 | add/subtract | 동일 | ✅ |

### 모델 호환성 검증

SAVE는 LLaVA-NeXT만 지원합니다.

| Model | Hook 대상 | SAVE 지원 |
|-------|-----------|-----------|
| **LLaVA-NeXT** | `model.layers.{layer}` | ✅ |
| **LLaVA** | `model.layers.{layer}` | ❌ |
| **Qwen2-VL** | - | ❌ |
| **Qwen2.5-VL** | - | ❌ |

### Inference / Training (모델별)

SAVE는 학습 없이 inference-only로 동작합니다.  

## 1. VCD (Visual Contrastive Decoding)

### Reference
- **파일**: `VCD/vcd_utils/vcd_sample.py`
- **핵심 라인**: 141-153

```python
# Reference 핵심 로직 (lines 141-153)
cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

# cutoff (version 2 - Adaptive Plausibility Constraints)
cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```

### 현재 구현
- **파일**: `src/decoding/vcd.py`
- **핵심 라인**: 126-134

```python
# 현재 구현 (lines 126-134)
cutoff = torch.log(torch.tensor(self.beta, device=logits_orig.device)) + \
         logits_orig.max(dim=-1, keepdim=True).values

cd_logits = (1 + self.alpha) * logits_orig - self.alpha * logits_noised
cd_logits = cd_logits.masked_fill(logits_orig < cutoff, -float("inf"))
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Cutoff 공식 | `log(beta) + max(logits)` | `log(beta) + max(logits)` | ✅ |
| CD 공식 | `(1+alpha)*logits - alpha*logits_cd` | `(1+alpha)*logits - alpha*logits_noised` | ✅ |
| Masking | `masked_fill(logits < cutoff, -inf)` | `masked_fill(logits < cutoff, -inf)` | ✅ |
| Sampling | `logits_warper` (top-k/top-p/temp) | `sample_top_p` (top-k/top-p/temp) | ✅ |
| Default alpha | 0.5 | 0.5 | ✅ |
| Default beta | 0.1 | 0.1 | ✅ |

### 모델 호환성 검증

VCD는 `pixel_values`, `cache_position`, `past_key_values`, `attention_mask`를 사용합니다.

| Model | Forward Signature | VCD 지원 |
|-------|------------------|---------|
| **LLaVA** | `pixel_values`, `cache_position`, `image_sizes`, `past_key_values`, `attention_mask` | ✅ |
| **LLaVA-NeXT** | `pixel_values`, `cache_position`, `image_sizes`, `past_key_values`, `attention_mask` | ✅ |
| **Qwen2-VL** | `pixel_values`, `cache_position`, `image_grid_thw`, `rope_deltas`, `past_key_values`, `attention_mask` | ✅ |
| **Qwen2.5-VL** | `pixel_values`, `cache_position`, `image_grid_thw`, `rope_deltas`, `past_key_values`, `attention_mask` | ✅ |

**호환성 근거** (transformers 기준):
- `transformers/src/transformers/models/llava/modeling_llava.py:367-381`, `transformers/src/transformers/models/llava/modeling_llava.py:453-481`
- `transformers/src/transformers/models/llava_next/modeling_llava_next.py:602-620`, `transformers/src/transformers/models/llava_next/modeling_llava_next.py:704-733`
- `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py:1289-1306`, `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py:1392-1464`
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1405-1424`, `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1516-1564`

### Inference / Training (모델별)

VCD는 학습 없이 inference-only로 사용합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | `cache_position`은 내부에서 갱신, 첫 step만 이미지 전달 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | `image_sizes`는 processor 출력 사용 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | 3D RoPE 위해 `cache_position` 필요 (VCD가 관리) |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | 비디오 미사용 시 `second_per_grid_ts` 불필요 |

```python
# 공통 패턴 (processor 출력에서 필요한 키만 전달)
inputs = processor(images=image, text=prompt, return_tensors="pt")
with VCDMitigator(model, model_type="llava") as mitigator:
    output = mitigator.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        image_sizes=inputs.get("image_sizes"),
        image_grid_thw=inputs.get("image_grid_thw"),
    )
```

### 불일치 사항
없음

---

## 2. AvisC (Attention-based Vision Calibration)

### Reference
- **파일**: `AvisC/avisc_utils/avisc_sample.py`
- **핵심 라인**: 160-179 (Blind token detection), 206-208 (Contrastive decoding)

```python
# Reference: Blind token detection (lines 160-179)
layer_img_att_portion = []
for logit in outputs.attentions:
    img_logit = logit.mean(dim=1)[:,-1, img_idx]
    layer_img_att_portion.append(img_logit.sum())

layer_img_att_portion = torch.stack(layer_img_att_portion, dim=0)
total_img_att_portion = layer_img_att_portion.sum()
layer_img_att_portion = layer_img_att_portion / total_img_att_portion
k = count_top_p(layer_img_att_portion.unsqueeze(0), top_p=float(layer_gamma))

_, top_k_lay_idx = torch.topk(layer_img_att_portion.float(), k, dim=0)

# Thresholding
att_logits = torch.stack([attention[i].mean(dim=1)[:,-1,img_idx] for i in top_k_lay_idx], dim=1)
img_att_logits = att_logits.mean(dim=1)

mask_idx = torch.where(img_att_logits < img_att_logits.mean() + img_att_logits.std() * lamb)[1]
```

```python
# Reference: Contrastive decoding (lines 206-208)
cutoff = torch.log(torch.tensor(avisc_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
diffs = (1+avisc_alpha)*next_token_logits - avisc_alpha*next_token_logits_method
avisc_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```

### 현재 구현
- **파일**: `src/decoding/avisc.py`
- **핵심 라인**: 132-193 (Blind detection), 195-218 (Contrastive decoding)

```python
# 현재 구현: Blind token detection (lines 157-186)
layer_img_att = []
for attn in attentions:
    img_attn = attn.mean(dim=1)[:, -1, img_start:img_end]
    layer_img_att.append(img_attn.sum())

layer_img_att = torch.stack(layer_img_att, dim=0)
layer_probs = layer_img_att / layer_img_att.sum()

sorted_probs = torch.sort(layer_probs, descending=True)[0]
cumsum = torch.cumsum(sorted_probs, dim=0)
k = (cumsum < self.layer_gamma).sum().item() + 1
_, top_k_layers = torch.topk(layer_probs.float(), k, dim=0)
top_k_layers = top_k_layers.tolist()

att_stack = torch.stack([attentions[i].mean(dim=1)[:, -1, img_start:img_end] for i in top_k_layers], dim=1)
img_att = att_stack.mean(dim=1)

threshold = img_att.mean() + self.lamb * img_att.std()
blind_mask = img_att < threshold
```

```python
# 현재 구현: Contrastive decoding (lines 210-218)
cutoff = torch.log(torch.tensor(self.beta, device=logits_orig.device)) + \
         logits_orig.max(dim=-1, keepdim=True).values
diffs = (1 + self.alpha) * logits_orig - self.alpha * logits_masked
return diffs.masked_fill(logits_orig < cutoff, -float("inf"))
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Layer attention 계산 | `logit.mean(dim=1)[:,-1,img_idx].sum()` | `attn.mean(dim=1)[:,-1,img_start:img_end].sum()` | ✅ |
| Layer selection | `count_top_p(layer_probs, top_p=layer_gamma)` | `cumsum < layer_gamma` | ✅ |
| Threshold 공식 | `mean() + std() * lamb` | `mean() + lamb * std()` | ✅ |
| Blind token 기준 | `< threshold` (낮은 attention) | `< threshold` | ✅ |
| Contrastive 공식 | `(1+alpha)*logits - alpha*logits_masked` | `(1+alpha)*logits - alpha*logits_masked` | ✅ |
| Cutoff 공식 | `log(beta) + max(logits)` | `log(beta) + max(logits)` | ✅ |
| Masking scheme | `zeros`, `ones`, `noise` | `zeros`, `ones`, `noise` | ✅ |
| Sampling | `logits_warper` (top-k/top-p/temp) | `sample_top_p` (top-k/top-p/temp) | ✅ |

**Note**: Reference는 `img_idx = list(range(35, 35+576))`로 LLaVA 하드코딩. 현재 구현은 `_get_image_token_indices()`로 동적 감지.

### 모델 호환성 검증

AvisC는 `output_attentions=True`가 필수입니다.

| Model | `output_attentions` 지원 | Forward Signature | AvisC 지원 |
|-------|------------------------|------------------|-----------|
| **LLaVA** | ✅ (TransformersKwargs 통해) | `modeling_llava.py:367` | ✅ |
| **LLaVA-NeXT** | ✅ `output_attentions: Optional[bool]` | `modeling_llava_next.py:615` | ✅ |
| **Qwen2-VL** | ✅ `output_attentions: Optional[bool]` | `modeling_qwen2_vl.py:1298` | ✅ |
| **Qwen2.5-VL** | ✅ `output_attentions: Optional[bool]` | `modeling_qwen2_5_vl.py:1414` | ✅ |

**호환성 근거** (transformers 기준):
- `transformers/src/transformers/models/llava_next/modeling_llava_next.py:462,615`
- `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py:497,1298`
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1414`

### 불일치 사항
없음

**추가 참고**:
- SLA는 모델 내부 `logits_aug` 플래그로 적용되며 `output_hidden_states=True`가 필요합니다.

### Inference / Training (모델별)

AvisC는 학습 없이 inference-only로 사용합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | 첫 step에서 `output_attentions=True` 필요 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | 첫 step에서 `output_attentions=True` 필요 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | 3D RoPE 위해 `cache_position` 필요 (AvisC가 관리) |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | 비디오 미사용 시 `second_per_grid_ts` 불필요 |

---

## 3. VISTA (Visual Information Steering with Attention)

### Reference
- **파일**: `VISTA/llm_layers.py`
- **핵심 라인**: 7-34 (VSVLayer), 132-150 (add_vsv_layers)

```python
# Reference: VSVLayer forward (lines 15-31)
class VSVLayer(nn.Module):
    def forward(self, x):
        x = x.float()
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y = 0
        if self.simple_mode:
            # Simple mode (line 20-24)
            vsv = self.vsv[0]
            lam_schedule = self.lam[0]
            y = lam_schedule * F.normalize(vsv, dim=-1).repeat(1,x.shape[1],1)
            x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        else:
            # Full mode (line 26-30)
            for i in range(len(self.vsv)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]), F.cosine_similarity(x, -self.vsv[i], dim=-1)).unsqueeze(-1)
                y += self.lam[i] * lambda_sim * F.normalize(self.vsv[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.vsv)
            x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        return x.half()
```

### 현재 구현
- **파일**: `src/decoding/vista.py`
- **핵심 라인**: 18-66 (VSVLayer), 120-178 (add_vsv_layers)

```python
# 현재 구현: VSVLayer (reference 동일)
class VSVLayer(nn.Module):
    def forward(self, x):
        x = x.float()
        original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y = 0
        if self.simple_mode:
            vsv = self.vsv[0]
            lam_schedule = self.lam[0]
            y = lam_schedule * F.normalize(vsv, dim=-1).repeat(1,x.shape[1],1)
            x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        else:
            for i in range(len(self.vsv)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x, -self.vsv[i], dim=-1)).unsqueeze(-1)
                y += self.lam[i] * lambda_sim * F.normalize(self.vsv[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.vsv)
            x = F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm
        return x.half()
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Original norm 보존 | `torch.norm(x, p=2, dim=-1, keepdim=True)` | `torch.norm(x_float, p=2, dim=-1, keepdim=True)` | ✅ |
| Simple mode 공식 | `lam * F.normalize(vsv)` | `lam * F.normalize(vsv)` | ✅ |
| Full mode lambda_sim | `1.0 + max(0, cos_sim(x, -vsv))` | `1.0 + clamp(cos_sim, min=0.0)` | ✅ |
| Full mode y 계산 | `lam * lambda_sim * normalize(vsv)` | `lam * lambda_sim * normalize(vsv)` | ✅ |
| Full mode 평균 | `y / len(vsv)` | `y / len(vsv)` | ✅ |
| 최종 공식 | `normalize(normalize(x) + y) * ||x||` | `normalize(normalize(x) + y) * ||x||` | ✅ |
| Output dtype | `return x.half()` | `return x.half()` | ✅ |
| MLP에 적용 | `nn.Sequential(original_mlp, VSVLayer)` | `nn.Sequential(original_mlp, VSVLayer)` | ✅ |

### 모델 호환성 검증

VISTA는 MLP에 `VSVLayer`를 래핑하므로 각 모델의 layer 구조가 MLP를 포함해야 합니다.

| Model | MLP 모듈 경로 | VISTA 지원 |
|-------|-------------|-----------|
| **LLaVA** | `language_model.model.layers[i].mlp` | ✅ |
| **LLaVA-NeXT** | `language_model.model.layers[i].mlp` | ✅ |
| **Qwen2-VL** | - | ⚠️ (Reference 미지원) |
| **Qwen2.5-VL** | - | ⚠️ (Reference 미지원) |

**호환성 근거** (transformers 기준):
- `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py:544,556` - `Qwen2VLDecoderLayer.mlp`
- LLaMA-based 모델은 `LlamaDecoderLayer.mlp` 구조 사용

### 불일치 사항
없음

### Inference / Training (모델별)

VISTA는 VSV를 사전에 계산하는 optional training 단계가 있으며, 추론 시에는 레이어 래핑으로 적용됩니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | VSV 계산 시 `output_hidden_states=True` 필요 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | VSV 계산 시 `output_hidden_states=True` 필요 |
| **Qwen2-VL** | - | Reference 미지원 |
| **Qwen2.5-VL** | - | Reference 미지원 |

**VSV 계산 요약**:
- 각 데모에서 layer별 last-token hidden state 추출
- `pos - neg` 차이 벡터를 PCA로 축약
- `vsv` 텐서로 저장 후 `VISTAMitigator(vsv=...)`에 전달

**현재 구현 메모**:
- `src/decoding/vista.py`의 `compute_vsv()`는 `src/decoding/vista_utils/steering_vector.py:obtain_vsv`를 사용하여 Reference의 `ForwardTracer` 기반 hidden state 추출을 그대로 따른다.
- LLaVA 계열은 `model.language_model`로 VSV를 계산한다 (Reference의 `model_loader.llm_model` 동작).

### VSV 계산 유틸리티 (`vista_utils/`)

VISTA의 on-the-fly VSV 계산 유틸리티입니다.

**Reference**: `VISTA/steering_vector.py`, `VISTA/model_loader.py:364-430`

**핵심 개념**: 이미지별로 VSV를 실시간 계산
- `pos_kwargs`: Full prompt with image
- `neg_kwargs`: Null prompt without image (text-only)
- Direction: `pos - neg` (with_image - without_image)

#### 파일 구조

```
src/decoding/vista_utils/
├── __init__.py
├── steering_vector.py   # obtain_vsv, get_hiddenstates, ForwardTracer
├── prompt_utils.py      # prepare_pos_prompt, prepare_neg_prompt
└── pca.py               # PCA implementation
```

#### 주요 함수

| 함수 | Reference | 설명 |
|------|-----------|------|
| `obtain_vsv()` | steering_vector.py:113-129 | VSV 계산 (pos - neg) |
| `get_hiddenstates()` | steering_vector.py:91-110 | Hidden states 추출 |
| `ForwardTracer` | steering_vector.py:22-88 | Hook-based activation 수집 |
| `prepare_neg_prompt()` | model_loader.py:368-430 | Null prompt 생성 |
| `prepare_pos_prompt()` | model_loader.py:364-365 | Original kwargs 반환 |

#### 사용 예시

```python
from src.decoding.vista_utils import obtain_vsv, prepare_vsv_kwargs_pair

# 1. 입력 준비
kwargs_pair = prepare_vsv_kwargs_pair(
    model=llm_model,
    tokenizer=tokenizer,
    questions=formatted_questions,
    original_kwargs={"input_ids": input_ids, "images": image_tensor},
    model_type="llava",
)

# 2. VSV 계산 (이미지별 on-the-fly)
vsv, neg_emb = obtain_vsv(llm_model, [kwargs_pair])

# 3. VSV 적용
from src.decoding.vista import add_vsv_layers
add_vsv_layers(llm_model, torch.stack([vsv], dim=1).cuda(), [0.01])
```

#### VISTA vs VTI 차이

| 항목 | VISTA | VTI |
|------|-------|-----|
| **데이터** | 이미지별 on-the-fly | 사전 정의된 JSONL 데모 |
| **Direction** | with_image - without_image | correct - hallucinating |
| **적용 시점** | Inference time (per image) | Pre-computed (shared) |

---

## 4. VTI (Visual-Textual Intervention)

### Reference
- **파일**: `VTI/vti_utils/llm_layers.py`
- **핵심 라인**: 9-32 (VTILayer with 0.1 scaling)

```python
# Reference: VTILayer forward (lines 16-30)
class VTILayer(nn.Module):
    def forward(self, x):
        norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
        y = 0
        for i in range(len(self.vti_direction)):
            lambda_sim = 1.0
            y += self.lam[i] * lambda_sim * F.normalize(self.vti_direction[i], dim=-1)
        y = y / len(self.vti_direction)
        # 핵심: 0.1 fixed scaling (line 28)
        x = F.normalize(F.normalize(x.float(), dim=-1) + 0.1 * y, dim=-1) * norm
        return x.half()
```

- **파일**: `VTI/vti_utils/utils.py`
- **핵심 라인**: 203-230 (Textual direction: last-token per layer)

```python
# Reference: Direction 계산 (lines 218-219)
for demonstration_id in range(num_demonstration):
    h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
    # direction = correct - hallucinating (각 layer의 last token)
```

### 현재 구현
- **파일**: `src/decoding/vti.py`
- **핵심 라인**: 15-44 (VTILayer), 97-154 (add_vti_layers)

```python
# 현재 구현: VTILayer (reference 동일)
class VTILayer(nn.Module):
    def forward(self, x):
        norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
        y = 0
        for i in range(len(self.vti_direction)):
            y += self.lam[i] * F.normalize(self.vti_direction[i], dim=-1)
        y = y / len(self.vti_direction)
        x = F.normalize(F.normalize(x.float(), dim=-1) + 0.1 * y, dim=-1) * norm
        return x.half()
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Fixed scale | `0.1` (line 28) | `FIXED_SCALE = 0.1` (line 53) | ✅ |
| Norm 보존 | `torch.norm(x.float(), dim=-1)` | `torch.norm(x_float, p=2, dim=-1)` | ✅ |
| y 계산 | `lam * F.normalize(vti_direction)` | `lam * F.normalize(vti)` | ✅ |
| 최종 공식 | `normalize(normalize(x) + 0.1 * y) * norm` | `normalize(normalize(x) + 0.1 * y) * norm` | ✅ |
| Direction 방향 | `correct - hallucinating` (line 219) | `correct - hallucinating` (line 343) | ✅ |
| Last token 사용 | `h[layer][:,-1]` (line 205) | `h[:, -1]` (line 335-336) | ✅ |
| Output dtype | `return x.half()` | `x_new.half()` | ✅ |

**VISTA vs VTI 핵심 차이점:**
- VISTA: `lam * lambda_sim * normalize(vsv)` (동적 lambda_sim)
- VTI: `0.1 * (alpha * normalize(vti))` (고정 0.1 scaling)

### 모델 호환성 검증

VTI는 LLM/vision MLP에 `VTILayer`를 래핑합니다.

| Model | MLP 모듈 경로 | VTI 지원 |
|-------|-------------|---------|
| **LLaVA** | `language_model.model.layers[i].mlp` | ✅ |
| **LLaVA-NeXT** | `language_model.model.layers[i].mlp` | ✅ |
| **Qwen2-VL** | `model.language_model.layers[i].mlp` | ✅ |
| **Qwen2.5-VL** | `model.language_model.layers[i].mlp` | ✅ |

### 불일치 사항
없음

### Inference / Training (모델별)

VTI는 textual/visual direction을 사전에 계산하는 optional training 단계가 있으며, 추론 시 레이어 래핑으로 적용됩니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | textual/visual 계산 시 `output_hidden_states=True` 필요 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | 동일 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | vision encoder hook 사용 |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | vision encoder hook 사용 |

**Direction 계산 요약**:
- Textual: 각 layer last-token `correct - hallucinating` → PCA
- Visual: masked vs clean 패치 차이 → token-wise PCA (masked trials이 여러 개면 평균)

**추가 참고**:
- Qwen2-VL/2.5-VL에서 visual VTI 계산 시 `image_grid_thw`가 필수입니다.
- Visual VTI는 `[masked_trials, clean]` 입력만 허용합니다 (Reference와 동일).
- layer 수가 맞지 않으면 `len(vti) == len(layers)` assertion이 발생합니다.

### VTI Direction 계산 유틸리티 (`vti_utils/`)

VTI의 direction 계산 유틸리티입니다.

**Reference**: `VTI/vti_utils/utils.py`

**핵심 개념**: 사전 정의된 JSONL 데모에서 direction 계산
- `value`: Correct (non-hallucinating) response
- `h_value`: Hallucinating response
- Direction: `correct - hallucinating` (각 layer의 last token)

#### 파일 구조

```
src/decoding/vti_utils/
├── __init__.py
├── utils.py                        # 모든 유틸리티 함수
├── pca.py                          # PCA implementation
└── hallucination_vti_demos.jsonl   # 데모 데이터셋
```

#### 주요 함수

| 함수 | Reference 라인 | 설명 |
|------|---------------|------|
| `process_image()` | 26-44 | 이미지 프로세서로 이미지 처리 |
| `mask_patches()` | 46-81 | 패치 마스킹 (Visual VTI용) |
| `get_demos()` | 150-181 | JSONL에서 데모 로드 |
| `get_prompts()` | 84-148 | Positive/negative 입력 쌍 생성 |
| `get_hiddenstates()` | 184-210 | Layer별 last-token hidden state 추출 |
| `obtain_textual_vti()` | 212-230 | Textual direction 계산 (correct - hallucinating) |
| `get_visual_hiddenstates()` | 257-307 | Vision encoder hidden states 추출 |
| `obtain_visual_vti()` | 309-325 | Visual direction 계산 (masked - clean) |

#### JSONL 포맷

```json
{
  "id": "000000103108",
  "image": "COCO_train2014_000000103108.jpg",
  "value": "correct response...",
  "h_value": "hallucinating response...",
  "co_objects": ["Baseball bat", "Hotdog"],
  "question": "Describe this image in detail."
}
```

#### 사용 예시

```python
from src.decoding.vti_utils import get_demos, obtain_textual_vti, obtain_visual_vti, VTIArgs

# 1. 데모 로드
args = VTIArgs(num_demos=50, data_file="/path/to/coco")
inputs_images, input_ids = get_demos(args, image_processor, model, tokenizer)

# 2. Textual VTI direction 계산
textual_direction, _ = obtain_textual_vti(model, input_ids, inputs_images, rank=1)

# 3. Visual VTI direction 계산 (optional)
visual_direction, _ = obtain_visual_vti(model, inputs_images, rank=1)

# 4. VTI 적용
with get_mitigator('VTIMitigator', model, model_type="llava",
                   textual_vti=textual_direction,
                   alpha_text=0.8) as m:
    output = m.generate(input_ids, images=image_tensor)
```

---

## 5. Middle Layers (Image Attention Boost)

### Reference
- **파일**: `middle_layers_indicating_hallucinations/modify_attention.py`
- **핵심 라인**: 77-88 (Image attention modification)

```python
# Reference: Image attention modification (lines 77-88)
### vision attention modification
if hasattr(self, "aggregation"):
    img_start_idx = self.img_start_idx
    img_end_idx = self.img_end_idx
    aggregation = self.aggregation

    if aggregation == "mean":
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx]
            + self.alpha * attn_weights[:, :, -1, img_start_idx:img_end_idx].abs().mean(dim=1, keepdim=True)
        )
### vision attention modification
```

### 현재 구현
- **파일**: `src/decoding/middle_layers.py`
- **핵심 라인**: 154-161 (LLaMA), 250-257 (Qwen)

```python
# 현재 구현: Image attention boost (lines 154-161)
# === IMAGE ATTENTION BOOST ===
img_start = mitigator._img_start
img_end = mitigator._img_end
if img_end > img_start and attn_weights.shape[-1] > img_start:
    actual_end = min(img_end, attn_weights.shape[-1])
    if mitigator.aggregation == "mean":
        boost = mitigator.alpha * attn_weights[:, :, -1, img_start:actual_end].abs().mean(dim=1, keepdim=True)
        attn_weights[:, :, -1, img_start:actual_end] = attn_weights[:, :, -1, img_start:actual_end] + boost
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| 적용 위치 | `attn_weights[:,:,-1,img_start:img_end]` | `attn_weights[:,:,-1,img_start:actual_end]` | ✅ |
| Boost 계산 | `alpha * abs().mean(dim=1, keepdim=True)` | `alpha * abs().mean(dim=1, keepdim=True)` | ✅ |
| Aggregation | `"mean"` | `"mean"` | ✅ |
| 적용 방식 | `attn + boost` (pre-softmax) | `attn + boost` (pre-softmax) | ✅ |
| forward 교체 | `types.MethodType()` | `types.MethodType()` | ✅ |

**추가 기능 (현재 구현):**
- `actual_end = min(img_end, attn_weights.shape[-1])` - boundary check 추가
- Qwen2-VL 전용 attention forward 구현 (3D RoPE 지원)

### 모델 호환성 검증

Middle Layers는 attention forward를 교체하므로 eager attention이 필요합니다.

| Model | Attention 모듈 | Middle Layers 지원 |
|-------|---------------|-------------------|
| **LLaVA** | `LlamaAttention` (language_model.model.layers[i].self_attn) | ✅ |
| **LLaVA-NeXT** | `LlamaAttention` | ✅ |
| **Qwen2-VL** | `Qwen2VLSdpaAttention` → `eager` 모드로 전환 | ✅ |
| **Qwen2.5-VL** | `Qwen2_5_VLAttention` → `eager` 모드로 전환 | ✅ |

**호환성 근거** (transformers 기준):
- LLaMA: `transformers/src/transformers/models/llama/modeling_llama.py` - `LlamaAttention`
- Qwen2-VL: `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py:491-541` - `Qwen2VLAttention`
- Qwen models require `_attn_implementation = 'eager'` for custom forward

### 불일치 사항
없음

### Inference / Training (모델별)

Middle Layers는 학습 없이 inference-only로 사용합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | attention forward 교체 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | attention forward 교체 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | eager attention 필요 |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | eager attention 필요 |

---

## 6. FarSight (Upper Triangular Penalty Register)

### Reference
- **파일**: `FarSight/Shell/farsight_patch.py`
- **핵심 라인**: 7-126

```python
# Reference: 핵심 공식 (lines 34-61, 112-113)
# Attention scores
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

# Causal mask
C = torch.tril(torch.ones((L, L), device=device, dtype=dtype)).view(1, 1, L, L)

# Upper triangular distance matrix
delta = (j_idx - i_idx).to(dtype)
upper = torch.triu(torch.ones((L, L), device=device, dtype=dtype), diagonal=1)

# Basic linear decay: P_basic = -(sigma * ReLU(j-i)) * upper
base_sigma = float(math.log(256) / math.log(1024))
P_basic = -(base_sigma * F.relu(delta)) * upper

# Progressive decay
prog_factor = 1.0 - (i_idx.to(dtype) / max(L - 1, 1)) * 0.5
P_prog = -(base_sigma * prog_factor) * F.relu(delta) * upper

# ALiBi slopes
slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)])
P_alibi = -(F.relu(delta) * upper) * slopes

# Final: W = attn_scores * C + P
W = attn_scores * C + P_total
attn_probs = torch.softmax(W, dim=-1) * C
```

### 현재 구현
- **파일**: `src/decoding/farsight.py`
- **핵심 라인**: 82-183 (LLaMA), 185-284 (Qwen)

```python
# 현재 구현: 핵심 공식 (lines 120-171)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
C = torch.tril(torch.ones((L, L), device=device, dtype=dtype)).view(1, 1, L, L)

delta = (j_idx - i_idx).to(dtype)
upper = torch.triu(torch.ones((L, L), device=device, dtype=dtype), diagonal=1)

sigma = mitigator.decay_factor
P_basic = -(sigma * F.relu(delta)) * upper

prog_factor = 1.0 - (i_idx.to(dtype) / max(L - 1, 1)) * 0.5
P_prog = -(sigma * prog_factor) * F.relu(delta) * upper

P_static = (0.5 * P_basic + 0.5 * P_prog).view(1, 1, L, L)

if mitigator.use_alibi:
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)])
    P_alibi = -(F.relu(delta) * upper).view(1, 1, L, L) * slopes

W = attn_scores * C + P_total
attn_probs = torch.softmax(W, dim=-1) * C
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| W 공식 | `attn_scores * C + P` | `attn_scores * C + P_total` | ✅ |
| Â 공식 | `softmax(W) * C` | `softmax(W) * C` | ✅ |
| P_basic | `-(sigma * ReLU(j-i)) * upper` | `-(sigma * F.relu(delta)) * upper` | ✅ |
| Progressive decay | `1.0 - (i/L) * 0.5` | `1.0 - (i/L) * 0.5` | ✅ |
| ALiBi slopes | `2^(-8*(h+1)/H)` | `2^(-8*(h+1)/H)` | ✅ |
| Default sigma | `log(256)/log(1024)` | `log(256)/log(1024)` | ✅ |
| Padding-safe | `valid_cols * valid_rows` | `valid_cols * valid_rows` | ✅ |
| KV cache | ❌ Not supported | ❌ `use_cache=False` enforced | ✅ |

### 모델 호환성 검증

FarSight는 attention forward를 완전히 교체하므로 eager attention이 필요합니다.

| Model | Attention 교체 | use_cache | FarSight 지원 |
|-------|--------------|-----------|--------------|
| **LLaVA** | `LlamaAttention` | False (enforced) | ✅ |
| **LLaVA-NeXT** | `LlamaAttention` | False | ✅ |
| **Qwen2-VL** | `Qwen2VLAttention` (returns 2 values) | False | ✅ |
| **Qwen2.5-VL** | `Qwen2_5_VLAttention` | False | ✅ |

**호환성 근거** (transformers 기준):
- LLaMA: 3-value return `(output, attn_weights, past_key_value)`
- Qwen2-VL: 2-value return `(output, attn_weights)` - 별도 forward 구현

### 불일치 사항
없음

### Inference / Training (모델별)

FarSight는 학습 없이 inference-only로 사용합니다. `use_cache=False`가 필수입니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | `use_cache=False` 강제 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | `use_cache=False` 강제 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | eager attention 필요 |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | eager attention 필요 |

---

## 7. Deco (Decoding with Early Exit Calibration)

### Reference
- **파일**: `Deco/transformers/generation/utils.py`
- **핵심 라인**: 2660-2682

```python
# Reference: Candidate selection & layer selection (lines 2660-2676)
last_layer_tokens_logits = outputs.logits[:, -1, :]
last_layer_tokens_probs = nn.functional.softmax(last_layer_tokens_logits, dim=-1)
candidate_tokens_probs, candidate_tokens_ids = torch.topk(last_layer_tokens_probs, k=threshold_top_k)
candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p)
candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))
candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]

stacked_early_exit_layers = torch.stack([dict_outputs[i][:, -1, :] for i in early_exit_layers], dim=0)
softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(1)
max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
premature_max_probs = candidate_tokens_early_exit_probs.max().item()
target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1)

# Reference: Blend formula (lines 2680-2682)
final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))
```

### 현재 구현
- **파일**: `src/decoding/deco.py`
- **핵심 라인**: 115-182

```python
# 현재 구현: Candidate selection (lines 132-157)
probs = F.softmax(final_logits, dim=-1).squeeze(0)
candidate_probs, candidate_ids = torch.topk(probs, k=self.threshold_top_k, dim=-1)
cumulative = candidate_probs.cumsum(dim=-1)
cutoff_idx = torch.searchsorted(cumulative, self.threshold_top_p, right=False)
cutoff_idx = torch.clamp(cutoff_idx + 1, max=self.threshold_top_k)
candidate_ids = candidate_ids[:cutoff_idx]

stacked = torch.stack([early_logits[i] for i in layer_indices], dim=0)
softmaxed = F.softmax(stacked, dim=-1)
candidate_probs_early = softmaxed[:, :, candidate_ids].squeeze(1)
max_prob_flat_idx = candidate_probs_early.argmax()
layer_idx_local = (max_prob_flat_idx // candidate_probs_early.size(1)).item()

# 현재 구현: Blend formula (lines 175-180)
blended = final_logits + self.alpha * max_prob * early_logits
mask = torch.ones_like(blended, dtype=torch.bool)
mask[:, candidate_ids] = False
blended = blended.masked_fill(mask, -float("inf"))
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Top-k selection | `torch.topk(probs, k=threshold_top_k)` | `torch.topk(probs, k=threshold_top_k)` | ✅ |
| Top-p filtering | `searchsorted(cumsum, top_p)` | `searchsorted(cumsum, top_p)` | ✅ |
| Layer stacking | `torch.stack([dict_outputs[i] for i...])` | `torch.stack([early_logits[i] for i...])` | ✅ |
| Layer selection | `argmax // num_candidates` | `argmax // size(1)` | ✅ |
| Blend formula | `logits + alpha * max_prob * early_logits` | `logits + alpha * max_prob * early_logits` | ✅ |
| Candidate masking | `masked_fill(not_in_candidates, -inf)` | `masked_fill(mask, -inf)` | ✅ |

### 모델 호환성 검증

Deco는 `output_hidden_states=True`로 중간 레이어 hidden states를 가져와서 norm + lm_head를 적용합니다.

| Model | `output_hidden_states` | norm/lm_head 경로 | Deco 지원 |
|-------|----------------------|------------------|----------|
| **LLaVA** | ✅ | `language_model.model.norm`, `language_model.lm_head` | ✅ |
| **LLaVA-NeXT** | ✅ | `language_model.model.norm`, `language_model.lm_head` | ✅ |
| **Qwen2-VL** | ✅ | `model.language_model.norm`, `lm_head` | ✅ |
| **Qwen2.5-VL** | ✅ | `model.language_model.norm`, `lm_head` | ✅ |

**호환성 근거** (transformers 기준):
- 모든 VLM은 `output_hidden_states=True` 지원
- norm/lm_head 경로는 base.py의 `_get_norm_and_lm_head()` 메서드에서 모델 타입별로 처리

### 불일치 사항
없음

### Inference / Training (모델별)

Deco는 학습 없이 inference-only로 사용합니다. `output_hidden_states=True`가 필요합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | cache_position 필요 (Deco가 관리) |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | cache_position 필요 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | 3D RoPE 위해 cache_position 필요 |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | 동일 |

---

## 8. OPERA (Over-trust Penalty and Retrospective Allocation)

### Reference
- **파일**: `OPERA/transformers-4.29.2/src/transformers/generation/utils.py`
- **핵심 라인**: 3419-3449 (Penalty), 3457-3557 (Rollback)

```python
# Reference: Attention penalty (lines 3419-3449)
attn_local = attn_last[:, :, response_start:, response_start:]
attn_local = scale_factor * attn_local  # scale_factor=50.0

for j in range(attn_local.shape[-1]):
    local_score = 1e-7 * attn_local[..., j:, j].prod(-1).data
    attn_local_scores[..., j] = local_score

# Early tokens (<=10): penalize based on image attention
attn_scores = attn_last[:, :, -1, img_start:img_end+1].sum(-1)
rollback_scores, rollback_locs = attn_local_scores.max(-1)

penalty_scores = -attn_scores if cur_response_lens <= 10 else rollback_scores

# Apply penalty
candidate_token_scores -= penalty_weights * penalty_scores
```

```python
# Reference: Rollback condition (lines 3457-3461)
if all((rollback_loc_gather == rollback_loc).long().sum() > threshold
       for _, rollback_loc_gather in enumerate(rollback_loc_gathers)):
    if rollback_loc < 10:
        assert False  # Don't rollback for early tokens
    rollback_pos = rollback_loc + 1
```

### 현재 구현
- **파일**: `src/decoding/opera.py`
- **핵심 라인**: 120-227 (Penalty), 228-389 (Rollback)

```python
# 현재 구현: Attention penalty (reference 동일)
attn_local = attn_last[:, :, attn_pos["response_start"]:, attn_pos["response_start"]:]
attn_local = scale_factor * attn_local

for j in range(attn_local.shape[-1]):
    local_score = 1e-7 * attn_local[..., j:, j].prod(-1)
    attn_local_scores[..., j] = local_score

attn_scores = attn_last[:, :, -1, attn_pos["image_start"]:attn_pos["image_end"]+1].sum(-1)
rollback_scores, rollback_locs = attn_local_scores.max(-1)

penalty_scores = -attn_scores if cur_response_lens <= 10 else rollback_scores
```

```python
# 현재 구현: Rollback condition (reference 동일)
if all((rollback_loc_gather == rollback_loc).long().sum() > int(threshold)
       for _, rollback_loc_gather in enumerate(rollback_loc_gathers)):
    if rollback_loc < 10:
        assert False
    rollback_pos = rollback_loc + 1
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Scale factor | `scale_factor * attn_local` | `scale_factor * attn_local` | ✅ |
| Local score | `1e-7 * attn_local[...,j:,j].prod(-1)` | `1e-7 * attn_local[...,j:,j].prod(-1)` | ✅ |
| Early penalty | `-attn_scores` (image attention) | `-attn_scores` | ✅ |
| Late penalty | `rollback_scores` | `rollback_scores` | ✅ |
| Penalty apply | `candidate_token_scores -= penalty_weights * penalty_scores` | 동일 | ✅ |
| Rollback trigger | `threshold` consecutive matches | `threshold` consecutive matches | ✅ |
| Rollback 제외 | `rollback_loc < 10` skip | `rollback_loc >= 10` required | ✅ |
| Beam search | ✅ `BeamSearchScorer` 기반 | 동일 | ✅ |

### 모델 호환성 검증

OPERA는 `output_attentions=True`와 beam search를 사용합니다.

| Model | `output_attentions` | beam search | OPERA 지원 |
|-------|-------------------|-------------|-----------|
| **LLaVA** | ✅ | ✅ (`opera_beam_search`) | ✅ |
| **LLaVA-NeXT** | ✅ | ✅ | ✅ |
| **Qwen2-VL** | ⚠️ | ⚠️ | ⚠️ (Reference 미지원) |
| **Qwen2.5-VL** | ⚠️ | ⚠️ | ⚠️ (Reference 미지원) |

**호환성 근거** (transformers 기준):
- Beam search는 모든 모델에서 `model.generate(num_beams=N)` 형태로 지원
- `output_attentions=True`는 모든 VLM forward에서 지원

### 불일치 사항
없음

### Inference / Training (모델별)

OPERA는 학습 없이 inference-only로 사용합니다. `num_beams > 1`, `output_attentions=True`가 필요합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | beam search 필수 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | beam search 필수 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | beam search 필수 |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | beam search 필수 |

---

## 9. Octopus (Dynamic Strategy Selection)

### Reference
- **파일**: `Octopus/eval_bench/train_token_amber.py`
- **핵심 라인**: 132-256 (MyModel classifier), 286-317 (M3ID formula), 611-631 (DPO loss)

```python
# Reference: MyModel classifier architecture (lines 132-175)
class MyModel(torch.nn.Module):
    def __init__(self, model, d_model=4096, nhead=2, num_decoder_layers=2, num_classes=4, n_query=4, bt=1):
        super(MyModel, self).__init__()
        self.n_query = n_query
        self.model = model
        self.Llama = model.model

        self.queries = torch.nn.Parameter(torch.randn(n_query, d_model).to(dtype=torch.float16))
        self.cls_token = torch.nn.Parameter(torch.randn(1, d_model).to(dtype=torch.float16))

        # TransformerEncoder (not Decoder in actual usage)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead).to(dtype=torch.float16)
        encoder_layer.apply(self.init_weights)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_decoder_layers).to(dtype=torch.float16)

        # MLP with Kaiming initialization
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model//4).to(dtype=torch.float16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model//4, num_classes).to(dtype=torch.float16)
        )
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                torch.nn.init.constant_(layer.bias, 0)
```

```python
# Reference: M3ID formula (Octopus/avisc_utils/avisc_sample.py:286-317)
gamma_t = math.exp(-lambda_decay * t)
# Log-softmax formulation
lc = F.log_softmax(logits_orig, dim=-1)
lu = F.log_softmax(logits_text, dim=-1)
m3id_logit = lc + ((1 - gamma_t) / gamma_t) * (lc - lu)
```

```python
# Reference: DPO loss (lines 611-621)
policy_chosen_logps = torch.gather(action_scores.log_softmax(-1), 2, policy_chosen_label[:,:,None]).squeeze(2).sum(-1)
policy_rejected_logps = torch.gather(action_scores.log_softmax(-1), 2, policy_rejected_label[:,:,None]).squeeze(2).sum(-1)

pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = 0  # reference_free
logits = (pi_logratios - ref_logratios).sum(-1)
losses = -F.logsigmoid(beta * logits)
```

```python
# Reference: Save format (lines 626-631)
torch.save({
    'query': policy.queries.data,
    'cls': policy.cls_token.data,
    'mlp_state_dict': policy.mlp.state_dict(),
    'transformer': policy.transformer.state_dict()
}, path)
```

### 현재 구현
- **파일**: `src/decoding/octopus.py`
- **핵심 라인**: 27-141 (OctopusClassifier/OctopusPolicy), 168-214 (DPO loss), 240-289 (generate), 291-331 (save/load)

```python
# 현재 구현: OctopusClassifier (float32, batch_first=False)
class OctopusClassifier(nn.Module):
    def __init__(self, d_model=4096, nhead=2, num_layers=2, num_classes=4, n_query=4, bt=1):
        self.cls_token = nn.Parameter(torch.randn(1, d_model).to(dtype=torch.float32))
        self.queries = nn.Parameter(torch.randn(n_query, d_model).to(dtype=torch.float32))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead).to(dtype=torch.float32)
        encoder_layer.apply(self.init_weights)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers).to(dtype=torch.float32)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4).to(dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(d_model // 4, num_classes).to(dtype=torch.float32),
        )
```

```python
# 현재 구현: OctopusPolicy.generate (avisc_sample.py 사용)
def generate(self, inputs=None, **kwargs):
    from Octopus.avisc_utils import avisc_sample
    self.model.sample = types.MethodType(avisc_sample.sample, self.model)
    return avisc_sample.generate(self.model, inputs=inputs, mymodel=self, **kwargs)
```

```python
# 현재 구현: DPO loss (lines 286-326)
def compute_loss(self, hidden_states, chosen_actions, rejected_actions, beta=1.0):
    action_scores = self.classifier(hidden_states).unsqueeze(1)  # [B, 1, C]
    chosen_logps = torch.gather(action_scores.log_softmax(-1), 2, chosen_actions[:, :, None]).squeeze(2).sum(-1)
    rejected_logps = torch.gather(action_scores.log_softmax(-1), 2, rejected_actions[:, :, None]).squeeze(2).sum(-1)

    logits = (chosen_logps - rejected_logps).sum(-1)
    loss = -F.logsigmoid(beta * logits)
    return loss
```

```python
# 현재 구현: Save/Load (lines 620-651)
def save_pretrained(self, path):
    torch.save({
        'query': self.classifier.queries.data,
        'cls': self.classifier.cls_token.data,
        'mlp_state_dict': self.classifier.mlp.state_dict(),
        'transformer': self.classifier.transformer.state_dict(),
    }, path)
```

### 검증 결과: ✅ 일치

| 항목 | Reference | 현재 구현 | 일치 |
|------|-----------|----------|------|
| Classifier CLS token | `nn.Parameter(torch.randn(1, d_model))` | `nn.Parameter(torch.randn(1, d_model))` | ✅ |
| Transformer | `TransformerEncoder(nhead=2, num_layers=2)` | `TransformerEncoder(nhead=2, num_layers=2)` | ✅ |
| MLP 구조 | `d_model -> d_model//4 -> num_classes` | `d_model -> d_model//4 -> num_classes` | ✅ |
| MLP activation | `LeakyReLU` | `LeakyReLU` | ✅ |
| Kaiming init | `kaiming_uniform_(weight, nonlinearity='relu')` | `kaiming_uniform_(weight, nonlinearity='relu')` | ✅ |
| Bias init | `constant_(bias, 0)` | `constant_(bias, 0)` | ✅ |
| Generation loop | `avisc_sample.py` | `avisc_sample.py` | ✅ |
| DPO loss | `-logsigmoid(beta * (chosen - rejected))` | `-logsigmoid(beta * logits)` | ✅ |
| Save format | `query`, `cls`, `mlp_state_dict`, `transformer` | `query`, `cls`, `mlp_state_dict`, `transformer` | ✅ |
| num_classes | 4 (NONE, AVISC, VCD, M3ID) | 4 (ACTION_NONE/AVISC/VCD/M3ID) | ✅ |

**Reference 동작 재현 포인트**:
- `OctopusPolicy`는 MyModel과 동일하게 `inputs_embeds`에서 이미지 토큰(`idx + 35`)을 마스킹합니다.
- `avisc_sample.py`의 `logits_processor/logits_warper` 경로를 그대로 사용합니다.

### 모델 호환성 검증

Octopus는 VCD, AvisC, M3ID 세 가지 전략을 동적으로 선택하므로 모든 전략의 요구사항을 충족해야 합니다.

| Model | VCD (pixel_values) | AvisC (output_attentions) | M3ID (text-only) | Octopus 지원 |
|-------|-------------------|--------------------------|-----------------|-------------|
| **LLaVA** | ✅ | ✅ | ✅ | ✅ |
| **LLaVA-NeXT** | ✅ | ✅ | ✅ | ✅ |
| **Qwen2-VL** | ⚠️ | ⚠️ | ⚠️ | ⚠️ (Reference 미지원) |
| **Qwen2.5-VL** | ⚠️ | ⚠️ | ⚠️ | ⚠️ (Reference 미지원) |

**추가 요구사항:**
- LLaVA 기반 `prepare_inputs_labels_for_multimodal`, `prepare_inputs_for_generation_*`가 필요
- 이미지 토큰 offset(35) 기반 마스킹이 고정값임 (Reference 동일)
- 4개의 별도 KV cache: base, AvisC, VCD, M3ID
- 배치 크기 > 1이면 sample별로 generate를 순차 실행하고 pad 처리

**호환성 근거** (transformers 기준):
- 모든 VLM은 `output_hidden_states`, `output_attentions` 지원
- KV cache는 `use_cache=True`로 개별 관리 가능

### 불일치 사항
없음

### Inference / Training (모델별)

Octopus는 classifier 학습이 필요합니다. 학습 시 `output_scores=True`, `return_dict_in_generate=True`로 action logits을 수집합니다.

| Model | 필수 입력 | 비고 |
|-------|-----------|------|
| **LLaVA** | `pixel_values` | action logits per-step 저장 필요 |
| **LLaVA-NeXT** | `pixel_values`, `image_sizes` | 동일 |
| **Qwen2-VL** | `pixel_values`, `image_grid_thw` | 3D RoPE 위해 `cache_position` 필요 (Octopus가 관리) |
| **Qwen2.5-VL** | `pixel_values`, `image_grid_thw` | 동일 |

**학습 요약**:
- generate 중 action logits을 step별로 수집
- DPO loss: chosen/rejected action logp 차이 합산 후 `-logsigmoid(beta * logits)`

---
