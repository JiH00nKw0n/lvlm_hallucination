# Modality Gap

## 분류

- (1) modality-specific 원인이 핵심shared misalignment가 본질이라기보다, 모달리티 고유 정보/구조가 남는 것이 핵심이라는 입장
- (2) shared misalignment가 핵심이지만, 줄일 수 없거나 줄이면 안 됨gap이 shared-space misalignment이긴 해도, 본질적이거나 유용해서 적극 축소에 반대하는 입장
- (3) shared misalignment가 핵심이고, 실제로 줄여야 함gap을 shared-space 정렬 실패로 보고, 방법을 써서라도 줄여야 한다는 입장

## Modality Gap 논문

1. [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://openreview.net/forum?id=S7Evzt9uit3) / [arXiv:2203.02053](https://arxiv.org/abs/2203.02053)
    
    관점: modality gap은 multimodal shared space의 기하 현상이다. | 원인: narrow cone initialization bias와 contrastive optimization, 특히 temperature가 함께 만든다고 본다. | 해석: gap은 단순 artifact가 아니라 downstream 성능과 fairness에 영향하는 구조적 현상이다.
    
2. [Towards understanding the modality gap in CLIP](https://openreview.net/forum?id=8W3KGzw7fNI)
    
    관점: CLIP gap은 단순 초기화 잔재보다 학습이 빠지기 쉬운 구조적 상태에 가깝다. | 원인: CLIP loss의 local minima가 gap을 유지한다고 본다. | 해석: 초기화만 바꿔서는 부족하고 손실/최적화 구조 자체를 봐야 한다고 주장한다.
    
3. [Understanding and Constructing Latent Modality Structures in Multi-modal Representation Learning](https://arxiv.org/abs/2303.05952)
    
    관점: 목표는 perfect alignment가 아니라 downstream에 유리한 latent modality structure다. | 원인: 정보이론적으로 exact alignment는 일반적으로 최적이 아니라고 본다. | 해석: intra/inter-modal regularization으로 구조를 설계해야 한다고 본다. | 분류: `(1) modality-specific 원인이 핵심`
    
4. [Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data](https://openreview.net/forum?id=ttXg3SKAg5) / [arXiv:2401.08567](https://arxiv.org/abs/2401.08567)
    
    관점: gap이 남아 있으면 서로 다른 modality embedding의 interchangeability가 깨진다. | 원인: contrastive space에 modality별 성분과 기하적 분리가 남는다고 본다. | 해석: `C3`로 gap을 줄이면 uni-modal data만으로도 cross-modal task를 더 잘 풀 수 있다고 본다. | 분류: `(1) modality-specific 원인이 핵심`
    
5. [Accept the Modality Gap: An Exploration in the Hyperbolic Space](https://openreview.net/forum?id=8PJroTRUSN)
    
    관점: gap은 제거해야 할 결함보다 받아들여야 할 구조에 가깝다. | 원인: text와 image의 표현 방식과 정보 내용이 본질적으로 달라 proximity를 강제하면 hierarchy가 깨진다고 본다. | 해석: gap을 허용한 채 similarity를 정의하는 편이 더 낫다고 본다. | 분류: `(1) modality-specific 원인이 핵심`
    
6. [Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models](https://openreview.net/forum?id=uAFHCZRmXk) / [arXiv:2404.07983](https://arxiv.org/abs/2404.07983)
    
    관점: modality gap과 object bias는 같은 뿌리에서 나온 두 현상이다. | 원인: image-caption 간 `information imbalance`가 공통 원인이라고 본다. | 해석: gap만 닫는 것보다 supervision 비대칭을 이해하는 게 더 중요하다고 본다. | 분류: `(1) modality-specific 원인이 핵심`
    
7. [Mitigate the Gap: Improving Cross-Modal Alignment in CLIP](https://openreview.net/forum?id=aPTGvFqile) / [arXiv:2406.17639](https://arxiv.org/abs/2406.17639)
    
    관점: gap은 hypersphere에서 modality subregion이 과도하게 분리된 상태다. | 원인: encoder들이 너무 독립적으로 자라며 cross-modal alignment가 부족해진다고 본다. | 해석: parameter sharing과 regularization으로 gap을 줄여야 한다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
8. [Explaining and Mitigating the Modality Gap in Contrastive Multimodal Learning](https://openreview.net/forum?id=2sThreW73a) / [arXiv:2412.07909](https://arxiv.org/abs/2412.07909)
    
    관점: gap은 정적 geometry보다 학습 dynamics의 산물이다. | 원인: mismatched pairs와 learnable temperature가 gap을 만들고 유지한다고 본다. | 해석: temperature scheduling과 modality swapping으로 완화할 수 있다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
9. [The Double-Ellipsoid Geometry of CLIP](https://openreview.net/forum?id=QGUju9B68Z) / [arXiv:2411.14517](https://arxiv.org/abs/2411.14517)
    
    관점: CLIP raw embedding은 image와 text가 separate off-center ellipsoid shell 위에 놓인다고 본다. | 원인: false negative와 uncertainty 차이가 modality별 conformity distribution을 만들고 gap이 이를 맞춘다고 본다. | 해석: gap은 단순 결함이 아니라 uncertainty-aware organization에 기여하는 geometry일 수 있다고 본다.
    
10. [Post-pre-training for Modality Alignment in Vision-Language Foundation Models](https://arxiv.org/abs/2504.12717)
    
    관점: pretrained CLIP에는 여전히 image-text feature cluster gap이 남아 downstream을 제한한다. | 원인: 기존 gap reduction은 재학습 비용이 크거나 zero-shot 성능 저하를 일으킨다고 본다. | 해석: pretrain과 finetune 사이의 post-pre-training 단계로 gap을 줄여야 한다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
11. [Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning](https://arxiv.org/abs/2507.09118)
    
    관점: continual learning에서는 modality gap이 pre-trained knowledge preservation 정도를 반영하는 신호일 수 있다. | 원인: fine-tuning 중 gap 변화가 곧 기존 CLIP 지식 보존 정도와 연결된다고 본다. | 해석: gap을 무조건 없애기보다 preserve+compensate 해야 한다고 주장한다.
    
12. [Closing The Modality Gap Enables Novel Multimodal Learning Applications](https://openreview.net/forum?id=P3Oba8Z3B6)
    
    관점: gap은 multimodal semantic alignment를 가로막는 핵심 병목이다. | 원인: contrastive loss가 true semantic alignment를 충분히 만들지 못해 latent space가 fragmented된다고 본다. | 해석: gap을 닫으면 semantic communication, medical multimodal tasks 같은 새 응용이 열린다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
13. [Closing the Modality Gap for Mixed Modality Search](https://openreview.net/forum?id=tJE6rcoMPL) / [arXiv:2507.19054](https://arxiv.org/abs/2507.19054)
    
    관점: gap은 mixed-modality retrieval에서 직접적 failure mode다. | 원인: image/text embedding이 distinct cluster를 유지해 ranking bias와 fusion failure가 생긴다고 본다. | 해석: post-hoc calibration으로 gap을 제거해야 mixed search가 개선된다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
14. [Closing the Modality Gap Aligns Group-Wise Semantics](https://openreview.net/forum?id=RHPqr2egJO)
    
    관점: gap의 악영향은 instance-wise retrieval보다 group-wise semantics에서 더 크다. | 원인: pair-level alignment는 되어도 group-level semantic geometry는 partially shared 상태로 남는다고 본다. | 해석: clustering 같은 task에는 gap reduction이 중요하다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
15. [The Modality Gap in Multimodal Semantic Communication](https://openreview.net/forum?id=ssWFdcTWY1)
    
    관점: gap은 하나의 semantic concept를 unified representation으로 묶지 못하게 하는 structural misalignment다. | 원인: contrastive latent space가 modality별 embedding을 하나의 cluster로 충분히 collapse시키지 못한다고 본다. | 해석: gap reduction이 multimodal compression의 전제라고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
16. [Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP](https://openreview.net/forum?id=NXLjaYdgaL) / [arXiv:2310.19752](https://arxiv.org/abs/2310.19752)
    
    관점: text proxy를 그대로 vision zero-shot 분류에 쓰는 것은 modality gap 때문에 suboptimal하다. | 원인: contrastive loss만으로는 text-vision gap을 충분히 줄일 수 없고, optimal proxy는 vision space에만 있을 수 있다고 본다. | 해석: vision proxy를 직접 학습하는 편이 낫다고 본다. | 분류: `(1) modality-specific 원인이 핵심`
    
17. [Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion](https://openreview.net/forum?id=VVVfuIcmKR) / [arXiv:2502.04263](https://arxiv.org/abs/2502.04263)
    
    관점: CLIP의 inter-modal training은 intra-modal misalignment도 만든다. | 원인: inter-modal contrastive loss가 intra-modal constraint를 주지 않기 때문이라고 본다. | 해석: modality inversion이나 gap reduction이 same-modality task 개선에 도움된다고 본다. | 분류: `(3) shared misalignment가 핵심이고, 실제로 줄여야 함`
    
18. [Beyond Cross-Modal Alignment: Measuring and Leveraging Modality Gap in Vision-Language Models](https://openreview.net/forum?id=oVSQIwRwqs)
    
    관점: modality gap은 잘 정렬된 모델에도 남고, 오히려 유용할 수 있다. | 원인: abstract는 원인 규명보다 vision-dominant / language-dominant / cross-modal feature의 공존 자체에 초점을 둔다. | 해석: gap을 없애기보다 측정하고 probing/editing/control에 활용하자는 입장이다. | 분류: `(1) modality-specific 원인이 핵심`

## Modality Split 논문

가장 중요한 포인트부터 말하면, decomposing... 과 redundancy ... 논문들은 shared concept의 feature가 perfectly align되어있을 때에만 유용한 방법론을 제안함

- [Decomposing multimodal embedding spaces with group-sparse autoencoders](https://arxiv.org/abs/2601.20028) / [ICLR 2026 poster](https://openreview.net/forum?id=ZJlVXZ5dmK), 2026-01-27. 표준 SAE가 멀티모달 임베딩에서 `split dictionaries`를 배우는 문제를 직접 정의하고, group sparsity + cross-modal masking으로 완화합니다.
- [Interpreting CLIP with Hierarchical Sparse Autoencoders](https://arxiv.org/abs/2502.20578), 2025-02-27, ICML 2025. `modality split` 자체를 주제로 삼진 않지만, CLIP/SigLIP에 SAE를 붙인 주요 선행작입니다.
- [Cross-Modal Redundancy and the Geometry of Vision–Language Embeddings](https://openreview.net/forum?id=VYQuICALXj), ICLR 2026. Aligned SAE로 `bimodal atom`과 `unimodal atom`을 분리해서, unimodal atom이 modality gap을 설명한다고 주장합니다.
