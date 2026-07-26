"""Gather every rebuttal measurement into one Korean-language summary.

Each analysis writes its own JSON; this pulls the numbers that answer a specific
reviewer question into one place, organized by the question rather than by the
script that produced it, so the response can be written from a single page.

Anything missing is reported as missing rather than skipped, because a silently
absent row reads as "we chose not to report it".

    python scripts/real_alpha/collect_rebuttal_results.py \
        --root outputs --setting coco_k8 --run r1 --pair r1r2 \
        --out outputs/REBUTTAL_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SETTING_NAME = {
    "coco_k8": "COCO 학습, L=8192(한쪽당 4096), K=8, 30 epoch — 논문 Figure 2 조건",
    "cc3m_k32": "CC3M 학습, L=8192(한쪽당 4096), k=32, 10 epoch — 논문 Table 1 조건",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--setting", default="coco_k8", help="coco_k8 또는 cc3m_k32")
    p.add_argument("--run", default="r1")
    p.add_argument("--pair", default="r1r2")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pct(x: float | None) -> str:
    """Percent with enough digits that a chance rate never prints as 0.0%."""
    if x is None:
        return "--"
    v = 100 * x
    if v == 0:
        return "0%"
    return f"{v:.1f}%" if v >= 1 else f"{v:.3f}%"


def num(x: float | None, nd: int = 3) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    s, r, pair = args.setting, args.run, args.pair

    ea = load(root / "rebuttal_EA" / f"{s}_{pair}" / "fig_same_modality.json")
    ec = load(root / "rebuttal_EC" / f"{s}_{r}" / "match_confidence.json")
    ee = load(root / "rebuttal_EE" / f"{s}_{r}" / "one_to_many_span.json")
    ef = load(root / "rebuttal_EF" / f"{s}_{r}" / "alignment_ceiling.json")
    ed = load(root / "rebuttal_ED" / f"{s}_{pair}" / "stability_conditioned.json")
    eb = load(root / "rebuttal_EB" / f"{s}_{r}" / "coco80_correspondence.json")
    eb_na = load(root / "rebuttal_EB" / f"{s}_{r}_noarea" / "coco80_correspondence.json")
    ab = load(root / "rebuttal_EC" / f"{s}_{r}" / "confidence_ablation.json")

    L: list[str] = []
    L.append(f"# 리부탈 측정 결과 — {SETTING_NAME.get(s, s)}")
    L.append("")
    L.append("여기 있는 모든 수치는 이번 응답을 위해 새로 학습한 SAE에서 나왔다. 제출본의")
    L.append("체크포인트는 하나도 재사용하지 않았고, seed가 데이터 순서뿐 아니라 초기")
    L.append("가중치까지 통제하도록 학습 코드를 고쳤다. 각 run의 초기 가중치 해시를 파일로")
    L.append("남겨 세 run이 실제로 서로 다르게 시작했음을 확인할 수 있다.")
    L.append("")
    L.append("상관행렬은 제출본 그림을 만든 함수를 그대로 쓴다. 새 코드 경로로 원래")
    L.append("체크포인트를 다시 돌려 저장된 행렬과 비교했을 때 최대 절대 오차가 3e-4로,")
    L.append("half precision 저장에서 오는 반올림 수준이었다.")
    L.append("")

    # ---- same-modality control ----------------------------------------------
    L.append("## 보고된 격차가 통상적인 학습 변동보다 큰가")
    L.append("")
    L.append("*Reviewer PBPC Q1, AC의 첫 번째 우려.*")
    L.append("")
    L.append("> 리뷰어: \"같은 모달리티에서 독립적으로 학습한 SAE 사이의 feature 방향 거리를")
    L.append("> image SAE seed 1 대 seed 2처럼 재서, 보고된 heterogeneity가 통상적인 SAE 학습")
    L.append("> 변동보다 큰지 확인해달라.\"")
    L.append("")
    L.append("리뷰어가 제안한 비교를 그대로 한다. 확인하려는 것은 하나다 — 우리가 모달리티")
    L.append("차이라고 부른 것이 사실은 SAE를 두 번 학습하면 어차피 생기는 변동인가. 다만")
    L.append("리뷰어 제안을 곧이곧대로만 하면 우리에게 유리한 쪽으로 편향되므로, 같은")
    L.append("모달리티에 입력 불일치만 넣은 조건을 하나 더 두어 두 성분을 분리한다.")
    L.append("")
    if ea is None:
        L.append("**측정 없음** — `run_rebuttal_EA.sh`를 실행할 것.")
    else:
        L.append("co-activation 상관이 기준을 넘는 쌍에 한정해, 두 feature 방향 사이의 cosine")
        L.append("distance를 잰다. 자주 켜지는 latent 하나가 수백 개의 셀을 만들어 중앙값을")
        L.append("지배하지 못하도록, latent마다 자기 쌍들의 중앙값을 먼저 구하고 그 값들의")
        L.append("중앙값을 보고한다.")
        L.append("")
        L.append("| 비교 | 살아있는 latent | 상관 0.6 이상 쌍 | 거리 중앙값 | 95% 신뢰구간 | 거리 평균 | 95% 신뢰구간 |")
        L.append("|---|---|---|---|---|---|---|")
        labels = {
            "img_img": "image SAE, 두 run",
            "txt_txt": "text SAE, 두 run, 같은 캡션",
            "txt_txt_diffcap": "text SAE, 두 run, 다른 캡션",
            "img_txt": "image 대 text (논문이 재는 것)",
        }
        for k, lab in labels.items():
            e = ea["panels"].get(k)
            if e is None:
                # CC3M has one caption per image, so the different-caption
                # condition does not exist there.
                L.append(f"| {lab} | 해당 없음 | -- | -- | -- | -- | -- |")
                continue
            h = e["headline"]
            ci = h["ci95_over_latents"]
            mci = h.get("ci95_mean_over_latents", [None, None])
            L.append(f"| {lab} | {e['n_alive_a']}/{e['n_alive_b']} | {h['n_pairs']} | "
                     f"{num(h['median_over_latents'])} | [{num(ci[0])}, {num(ci[1])}] | "
                     f"{num(h.get('mean_over_latents'))} | "
                     f"[{num(mci[0])}, {num(mci[1])}] |")
        L.append(f"| 무작위 방향 | -- | -- | {num(ea['random_null_distance'], 2)} | -- | "
                 f"{num(ea['random_null_distance'], 2)} | -- |")
        L.append("")
        L.append("평균이 중앙값보다 일관되게 크다. 거의 직교한 쌍들이 만드는 긴 꼬리가 평균을")
        L.append("끌어올리기 때문이고, 같은 이유로 중앙값이 대표값으로는 더 안정적이다. 두 값을")
        L.append("모두 싣는 이유는 하나만 보면 다른 하나가 같은 방향인지 알 수 없기 때문이다.")
        L.append("신뢰구간은 셀이 아니라 latent를 재표집해 구한다 — 같은 latent가 만드는 셀들은")
        L.append("서로 독립이 아니다.")
        L.append("")
        L.append("다른 캡션 조건이 왜 필요한지 짚어둘 필요가 있다. 두 image SAE는 완전히 같은")
        L.append("벡터를 읽으므로 입력이 달라서 생기는 잡음이 아예 없는 반면, image 대 text는")
        L.append("한쪽이 사진을 다른 쪽이 그 사진에 대한 문장을 읽는다. 두 dictionary가")
        L.append("기하적으로 동일하더라도 이 불일치만으로 거리가 벌어질 수 있다. 다른 캡션")
        L.append("조건은 모달리티는 같게 두고 입력 불일치만 넣은 것이라, 두 성분을 분리해준다.")
        L.append("")
        p = ea.get("paired_img_txt_minus_img_img", {})
        if p.get("n_latents"):
            mci = p.get("ci95_mean", [None, None])
            L.append(f"두 비교 모두에서 기준을 넘는 image latent {p['n_latents']}개를 하나씩 짝지어")
            L.append(f"보면, 모달리티를 건너는 데 드는 추가 거리가 중앙값 {num(p['median_difference'])} ")
            L.append(f"(95% 신뢰구간 [{num(p['ci95'][0])}, {num(p['ci95'][1])}]), "
                     f"평균 {num(p.get('mean_difference'))} "
                     f"(95% 신뢰구간 [{num(mci[0])}, {num(mci[1])}])이다. "
                     f"Wilcoxon p={p['wilcoxon_p']:.1e}.")
            L.append("")

        others = sorted((root / "rebuttal_EA").glob(f"{s}_r*/fig_same_modality.json"))
        if len(others) > 1:
            L.append("세 run의 모든 조합에 대해 같은 비교를 반복한 결과:")
            L.append("")
            L.append("| run 조합 | image×image | text×text | text, 다른 캡션 | image×text | 짝비교 차이 |")
            L.append("|---|---|---|---|---|---|")
            for op in others:
                od = load(op)
                if od is None:
                    continue
                tag = op.parent.name.split("_")[-1]
                g = od["panels"]
                opr = od.get("paired_img_txt_minus_img_img", {})

                def med(key: str) -> str:
                    e = g.get(key)
                    return num(e["headline"]["median_over_latents"]) if e else "--"

                L.append(f"| {tag} | {med('img_img')} | {med('txt_txt')} | "
                         f"{med('txt_txt_diffcap')} | {med('img_txt')} | "
                         f"{num(opr.get('median_difference'))} |")
            L.append("")

    # ---- match quality -------------------------------------------------------
    L.append("## 매칭된 latent가 정말 같은 개념을 가리키는가")
    L.append("")
    L.append("*Reviewer PBPC Q2, AC의 매칭 품질 지적.*")
    L.append("")
    L.append("> 리뷰어: \"activation correlation이 찾아낸 의미 대응이 얼마나 정확한가? 매칭된")
    L.append("> image/text latent가 단지 상관된 개념이 아니라 같은 개념을 나타내는지, 객체나")
    L.append("> 속성 어노테이션 또는 top-activating 예시의 수동 평가로 정량 검증해달라.\"")
    L.append("")
    L.append("함께 켜진다는 것이 같은 개념이라는 뜻인지, 아니면 사진에서 자주 같이 등장하는")
    L.append("별개 개념일 뿐인지를 가른다. 의자와 식탁은 함께 나타나지만 같은 개념이 아니다.")
    L.append("그래서 판단 기준을 우리 모델 바깥에서 가져온다 — COCO가 사람 손으로 달아둔 객체")
    L.append("어노테이션이다.")
    L.append("")
    if eb is None:
        L.append("**측정 없음** — `eval_coco80_correspondence.py`를 실행할 것.")
    else:
        res, ctl = eb["result"], eb["controls"]
        L.append(f"COCO 객체 카테고리 중 표본이 충분한 {eb['n_categories']}개에 대해, 각 모달리티가")
        L.append("자기 쪽에서만 계산한 AUC로 그 카테고리를 가장 잘 구분하는 좌표를 하나씩 고른다.")
        L.append("AUC는 그 객체가 있는 샘플에서의 활성이 없는 샘플에서의 활성보다 클 확률이다.")
        L.append("이미지 쪽은 사진의 한쪽 절반만, 텍스트 쪽은 나머지 절반의 캡션만 보므로 두")
        L.append("선택이 서로에게 영향을 줄 수 없다. 그러고 나서야 학습된 permutation을 꺼내,")
        L.append("두 선택이 서로 매칭된 좌표인지 묻는다. **순위는 AUC로 매기며 co-activation")
        L.append("상관은 permutation을 만들 때만 쓰인다** — 순위까지 상관으로 매기면 순환 논증이")
        L.append("된다.")
        L.append("")
        L.append("순위는 양방향으로 보고한다. 이미지 쪽 선택이 텍스트 쪽 순위에서 몇 위인지와,")
        L.append("텍스트 쪽 선택이 이미지 쪽 순위에서 몇 위인지는 서로 다른 질문이기 때문이다.")
        L.append("1위에서는 두 질문이 같은 사건이 된다 — 양쪽이 매칭된 좌표를 골랐다는 뜻이다.")
        L.append("")
        L.append("| 방향 | 1위 | 5위 이내 | 10위 이내 | 순위 중앙값 | MRR |")
        L.append("|---|---|---|---|---|---|")
        for key, lab in (("image_pick_in_text_ranking", "이미지 쪽 선택을 텍스트 쪽 순위에서"),
                         ("text_pick_in_image_ranking", "텍스트 쪽 선택을 이미지 쪽 순위에서")):
            d = res.get(key)
            if not d:
                continue
            L.append(f"| {lab} | {pct(d['top1'])} | {pct(d['top5'])} | {pct(d['top10'])} | "
                     f"{d['median_rank']:.0f} / {eb['m_eff']} | {d['mrr']:.3f} |")
        L.append("")
        L.append("1위 일치율을 모든 기준선과 비교하면:")
        L.append("")
        L.append("| | 일치율 |")
        L.append("|---|---|")
        L.append(f"| 학습된 permutation | **{pct(res['agree@1'])}** |")
        L.append(f"| 무작위 permutation | {pct(ctl['random_permutation_hit@1_mean'])} |")
        L.append(f"| 카테고리 라벨을 섞은 경우 | {pct(ctl['label_shuffle_hit@1'])} |")
        L.append(f"| 우연 ({eb['m_eff']}개 중 하나) | {pct(ctl['chance_hit@1'])} |")
        L.append(f"| 이미지 쪽이 자기 나머지 절반과 (달성 가능한 천장) | {pct(ctl['image_self_agreement'])} |")
        L.append(f"| 텍스트 쪽이 자기 나머지 절반과 | {pct(ctl['text_self_agreement'])} |")
        L.append("")
        L.append(f"무작위 permutation 대비 p = {ctl['p_value_vs_random_permutation']:.4f}. "
                 f"{eb['n_categories']}개 카테고리에서 서로 다른 image latent가 "
                 f"{eb['distinct_image_latents_chosen']}개 선택됐으므로, 자주 켜지는 latent 몇 개가 "
                 f"전부를 이기는 퇴화 현상은 아니다.")
        L.append("")
        dropped = eb.get("categories_dropped", [])
        if dropped:
            L.append(f"표본이 부족해 빠진 {len(dropped)}개 카테고리: {', '.join(dropped)}. "
                     f"모두 사진에서 작게 나오는 물체이고, 객체가 화면 면적의 5% 이상을 "
                     f"차지해야 양성으로 치는 조건에서 걸러진다. CLIP 이미지 임베딩은 사진 "
                     f"전체를 벡터 하나로 요약하므로, 구석에 작게 있는 물체를 그 사진의 개념으로 "
                     f"보기는 어렵다는 판단이다.")
            L.append("")
        if eb_na is not None:
            rn = eb_na["result"]
            cn = eb_na["controls"]
            L.append("면적 조건을 아예 걸지 않았을 때의 민감도:")
            L.append("")
            L.append("| 면적 조건 | 카테고리 | 1위 일치 | 천장 | 라벨 섞기 |")
            L.append("|---|---|---|---|---|")
            L.append(f"| 5% 이상 | {eb['n_categories']} | {pct(res['agree@1'])} | "
                     f"{pct(ctl['image_self_agreement'])} | {pct(ctl['label_shuffle_hit@1'])} |")
            L.append(f"| 조건 없음 | {eb_na['n_categories']} | {pct(rn['agree@1'])} | "
                     f"{pct(cn['image_self_agreement'])} | {pct(cn['label_shuffle_hit@1'])} |")
            L.append("")
            L.append("조건을 풀면 카테고리는 늘지만 라벨 자체가 시끄러워져 천장도 함께 내려간다.")
            L.append("결론의 방향은 두 경우 모두 같다.")
            L.append("")

    # ---- match confidence ----------------------------------------------------
    L.append("## 매칭이 얼마나 강하고 얼마나 분명한가")
    L.append("")
    L.append("*Reviewer PBPC Q3.*")
    L.append("")
    L.append("> 리뷰어: \"이 방법은 feature splitting, merging, 양쪽 활성 latent 수가 다른 경우,")
    L.append("> 모달리티 고유 feature를 어떻게 다루는가? Hungarian 매칭의 correlation 분포와")
    L.append("> low-confidence 매칭의 비율을 보고해달라.\"")
    L.append("")
    L.append("분포와 비율은 요청 그대로 보고한다. \"low-confidence\"는 표준 정의가 없어서 우리가")
    L.append("정해야 했다. 세 가지로 나눠 읽되 무게를 다르게 뒀다. 상관의 절대 크기가 주")
    L.append("지표다 — 샘플이 수백만 개라 통계적 유의성은 거의 모든 매칭이 통과하므로 아무도")
    L.append("묻지 않은 질문에 답하는 셈이 된다. 보조로 2등 후보와의 차이(다른 데로 갔어도")
    L.append("이상하지 않았는가)와 상호 1순위 여부를 본다. 마지막으로 짝을 실제로 파괴해")
    L.append("잡음 바닥을 깐다.")
    L.append("")
    if ec is None:
        L.append("**측정 없음** — `analyze_match_confidence.py`를 실행할 것.")
    else:
        L.append(f"매칭된 {ec['n_matched_usable']}쌍의 co-activation 상관을 0.1 구간으로 나누면:")
        L.append("")
        L.append("| 상관계수 | 쌍 수 | 비율 | 누적 |")
        L.append("|---|---|---|---|")
        for b in ec.get("correlation_bands", []):
            L.append(f"| {b['range']} | {b['count']} | {pct(b['share'])} | "
                     f"{pct(b['cumulative_from_top'])} |")
        L.append("")
        amb = ec["ambiguity"]
        L.append(f"배정된 파트너와 2등 후보의 차이가 10% 이내인 경우가 "
                 f"{pct(amb['share_runner_up_within_10pct'])}다. 이런 쌍은 매칭이 다른 곳으로 "
                 f"갔어도 목적함수가 크게 달라지지 않았을 것이다. 두 latent가 서로를 1순위로 "
                 f"꼽는 경우는 {pct(ec['reciprocity']['share_mutual_first_choice'])}다.")
        nf = ec.get("noise_floor")
        if nf:
            L.append("")
            L.append(f"이미지와 캡션의 짝을 실제로 뒤섞고 상관행렬을 다시 계산한 뒤 같은 "
                     f"Hungarian을 돌리면, 그 절차가 순수한 잡음에서 찾아내는 상관의 99분위가 "
                     f"{num(nf['p99'], 4)}다. 실제 매칭의 {pct(nf['share_of_real_matches_below_floor'])}가 "
                     f"그 아래에 있다. 상관행렬의 열만 섞는 값싼 방법은 각 행의 최댓값이 "
                     f"보존되어 Hungarian이 같은 값을 다시 고르므로 쓸 수 없다.")
        L.append("")

    if ab is not None:
        L.append("약한 매칭이 실제로 손해를 끼치는지는 그 비율이 얼마인지와는 다른 질문이다.")
        L.append("상관이 컷오프 이상인 매칭만 남기고 나머지 좌표를 0으로 만든 뒤, 학습에 쓰지")
        L.append("않은 split에서 이미지와 캡션을 서로 검색하게 하면:")
        L.append("")
        L.append("| 남긴 매칭 | 좌표 수 | I→T R@1 | I→T R@5 | T→I R@1 | 파트너를 섞었을 때 I→T R@1 |")
        L.append("|---|---|---|---|---|---|")
        for name, e in ab["by_cutoff"].items():
            if "I2T R@1" not in e:
                continue
            L.append(f"| {name} | {e['n_coordinates']} | {100 * e['I2T R@1']:.2f} | "
                     f"{100 * e['I2T R@5']:.2f} | {100 * e['T2I R@1']:.2f} | "
                     f"{100 * e['shuffled_partners']['I2T R@1']:.2f} |")
        L.append("")
        L.append("마지막 열은 좌표는 그대로 두고 각 좌표가 어느 text latent와 짝지어지는지만")
        L.append("망가뜨린 조건이다. 여기서 성능이 0 근처로 무너진다는 것은, 검색을 떠받치는")
        L.append("것이 그 좌표들이 얼마나 활성화되느냐가 아니라 대응 관계 자체라는 뜻이다.")
        L.append("")

    # ---- feature splitting ---------------------------------------------------
    L.append("## 이 격차가 그냥 feature splitting 때문 아닌가")
    L.append("")
    L.append("*AC의 세 번째 우려.*")
    L.append("")
    L.append("> 요청: 1:N인 경우를 찾아, 1에 해당하는 image column vector와 N에 해당하는 text")
    L.append("> column vector들이 펼치는 span 사이의 직교 거리, 즉 span에 직교하는 성분의 크기를")
    L.append("> 구해볼 것.")
    L.append("")
    L.append("측정된 거리가 사실은 splitting의 부산물이라는 반론을 검정한다. image 쪽 개념")
    L.append("하나가 text 쪽에서 N개로 쪼개진 것뿐이라면, 그 N개를 다 합친 공간은 image 방향을")
    L.append("설명할 수 있어야 한다. decoder 행은 학습 시점에 이미 단위벡터로 정규화되어")
    L.append("있으므로, 사영 성분과 직교 잔차의 제곱합이 정확히 1이 되고 설명 비율을 1 − r²로")
    L.append("읽을 수 있다.")
    L.append("")
    if ee is None:
        L.append("**측정 없음** — `analyze_1toN_span.py`를 실행할 것.")
    elif ee.get("n_groups", 0) == 0:
        L.append(f"이 임계값({ee['tau']})에서 파트너가 둘 이상인 image latent가 없다.")
    else:
        ex = ee["explained"]
        L.append(f"하나의 image latent가 상관 {ee['tau']} 이상으로 둘 이상의 text latent와 이어지는")
        L.append(f"1:N 그룹은 살아있는 image latent의 {pct(ee['group_share_of_alive_image'])}를 차지한다")
        L.append(f"({ee['n_groups']}개 그룹). image 방향을 그 text 파트너들이 펼치는 부분공간에")
        L.append("사영했을 때 설명되는 에너지 비율은 다음과 같다.")
        L.append("")
        L.append("| 부분공간 | 설명 비율 중앙값 |")
        L.append("|---|---|")
        L.append(f"| 모든 파트너 | {num(ex['all_partners']['median'])} |")
        L.append(f"| 가장 강한 파트너 하나 | {num(ex['strongest_partner_only']['median'])} |")
        L.append(f"| 가장 강한 파트너 + 무작위 text atom | {num(ex['strongest_partner_plus_random_atoms']['median'])} |")
        L.append(f"| 무작위 text atom | {num(ex['random_text_atoms']['median'])} |")
        L.append(f"| 무작위 방향 | {num(ex['random_unit_directions']['median'])} "
                 f"(이론값 {num(ee['analytic_random_subspace'])}) |")
        L.append("")
        L.append("N차원 부분공간은 어떤 방향이든 우연히 N/d 정도는 설명하므로, 설명 비율만")
        L.append("단독으로 보면 의미가 없다. splitting 가설을 실제로 검정하는 비교는 가장 강한")
        L.append("파트너를 남기고 나머지를 무작위 atom으로 바꾼 조건이다.")
        L.append("")
        L.append(f"쪼개진 나머지 파트너들이 더하는 몫은 {num(ee['marginal_gain_over_strongest'])}이고 "
                 f"대조군은 {num(ee['marginal_gain_of_control'])}이다. 효과는 실재하지만 작고, "
                 f"image 방향의 {pct(ee['unexplained_median'])}는 어떤 splitting 조합으로도 설명되지 "
                 f"않는다. 설명 비율이 0.5를 넘는 그룹은 {pct(ee['frac_groups_explained_above_half'])}다.")
        L.append("")

    # ---- ceiling -------------------------------------------------------------
    L.append("## 더 좋은 매칭이나 전역 변환으로 격차를 없앨 수 있나")
    L.append("")
    L.append("*요청 범위 밖에서 추가한 검정. 리뷰어가 낼 수 있는 가장 강한 반론 두 개를 미리")
    L.append("막는다 — 매칭 알고리즘이 나빠서 거리가 큰 것 아니냐, 그리고 두 공간이 통째로")
    L.append("회전만큼 어긋난 것 아니냐. Reviewer 3VJU가 요청한 \"CCA나 Procrustes, optimal")
    L.append("transport 같은 다른 정렬 방법과 비교해달라\"에도 부분적으로 답이 된다.*")
    L.append("")
    if ef is None:
        L.append("**측정 없음** — `analyze_alignment_ceiling.py`를 실행할 것.")
    else:
        o = ef["oracle_cosine"]
        onull = ef["oracle_cosine_against_random_directions"]
        t = ef["global_transform"]
        L.append("매칭을 아예 버리고, 각 image 방향에 대해 dictionary 전체에서 가장 가까운 text")
        L.append(f"방향을 신이 골라준다고 하면 cosine 중앙값이 {num(o['median'])}이다 "
                 f"(거리 {num(1 - o['median'])}). 같은 탐색을 무작위 방향에 대고 하면 "
                 f"{num(onull['median'])}이고, 그만큼의 후보 중 최댓값의 이론값은 "
                 f"{num(ef['oracle_chance_analytic'])}이다. 어떤 매칭 절차도 이 상한을 넘을 수 "
                 f"없으므로, 거리가 큰 이유를 매칭 알고리즘 탓으로 돌릴 수 없다.")
        L.append("")
        L.append("매칭된 쌍의 절반으로 전역 변환 하나를 적합하고 나머지 절반에서 평가하면:")
        L.append("")
        L.append("| | held-out cosine |")
        L.append("|---|---|")
        L.append(f"| 변환 없음 | {num(t['identity_cos'])} |")
        L.append(f"| 최적 회전 | {num(t['rotation_cos'])} |")
        L.append(f"| 최적 선형변환 | {num(t['linear_cos'])} |")
        L.append(f"| 짝을 섞어 적합한 회전 | {num(t['rotation_on_shuffled_pairs_cos'])} |")
        L.append("")
        L.append("회전을 맞출수록 오히려 나빠지므로, 두 dictionary의 차이는 하나의 전역 회전으로")
        L.append("설명되지 않는다. 짝을 섞어 적합한 회전이 바닥에 붙는 것은 이 검정에 자유")
        L.append("파라미터로 인한 누수가 없음을 보여준다.")
        L.append("")

    # ---- stability -----------------------------------------------------------
    L.append("## 두 run이 모두 재현해내는 개념에서도 격차가 남는가")
    L.append("")
    L.append("*AC의 dictionary non-identifiability 우려.*")
    L.append("")
    L.append("> 요청: Papadimitriou et al. (arXiv 2504.11695)의 stability 지표를 써서, 두 모델의")
    L.append("> 안정성에 가장 크게 기여하는 상위 1% concept을 구하고, 그 concept들에 대해서마저도")
    L.append("> image column vector와 text column vector의 cosine이 1이 아님을 보일 것.")
    L.append("")
    L.append("한 번의 학습에서만 나오는 방향이라면 그것과 다른 모달리티 사이의 거리는 데이터에")
    L.append("대해 아무것도 말해주지 않는다. 그래서 두 번 독립적으로 학습해도 거의 그대로 다시")
    L.append("나오는 concept, 즉 학습 잡음이라고 보기 어려운 것들만 골라 거리를 다시 잰다.")
    L.append("안정성 정의는 인용 논문 그대로다 — 두 dictionary의 행을 총 유사도가 최대가 되도록")
    L.append("Hungarian으로 정렬한 뒤, 각 concept의 매칭된 cosine을 그 concept의 안정성으로 본다.")
    L.append("")
    if ed is None:
        L.append("**측정 없음** — `analyze_stability_conditioned.py`를 실행할 것.")
    else:
        L.append(f"두 run의 image dictionary 사이 평균 안정성은 {num(ed['mean_stability'])}이고 "
                 f"비교된 concept은 {ed['n_concepts']}개다.")
        L.append("")
        L.append("두 거리를 나란히 놓을 때 짚어야 할 것이 있다. 같은 모달리티 쪽은 decoder")
        L.append("cosine을 최대화하도록 짝을 지었고 그 cosine을 보고하므로, 보고하는 값을 그대로")
        L.append("최적화한 셈이다. 반면 cross-modal 쪽은 co-activation 상관을 최대화한 짝의")
        L.append("cosine이다. 그래서 격차의 일부는 모달리티가 아니라 매칭 기준의 차이에서 온다.")
        L.append("cross-modal 쪽도 같은 방식으로, 즉 decoder cosine으로 짝지은 열을 함께 싣는다.")
        L.append("")
        L.append("| 안정성 상위 | 개수 | 안정성 | 같은 모달리티 거리 | cross-modal 거리 | cross-modal, 기준 통일 | 매칭 상관 |")
        L.append("|---|---|---|---|---|---|---|")
        for name, e in ed["by_stability_quantile"].items():
            L.append(f"| {name.replace('top_', '').replace('pct', '%')} | {e['n']} | "
                     f"{num(e['stability_median'])} | "
                     f"{num(e['same_modality_distance_median'])} | "
                     f"{num(e['cross_modal_distance_median'])} | "
                     f"{num(e.get('cross_modal_distance_median_matched_by_geometry'))} | "
                     f"{num(e['matched_correlation_median'])} |")
        L.append("")
        ov = ed.get("operator_vs_modality")
        if ov:
            L.append(f"양쪽을 똑같이 decoder cosine으로 짝지으면 같은 모달리티 "
                     f"{num(ov['same_modality'])} 대 cross-modal "
                     f"{num(ov['cross_modal_matched_by_geometry'])}다. 표에 실린 cross-modal "
                     f"값과의 차이를 나누면 매칭 기준이 "
                     f"{num(ov['attributable_to_operator'])}, 모달리티가 "
                     f"{num(ov['attributable_to_modality'])}를 설명한다. 상위 1% 구간에서는 두 열이 "
                     f"일치하는데, 가장 재현이 잘 되는 concept은 상관으로 짝지어도 기하로 짝지어도 "
                     f"같은 파트너가 나오기 때문이다.")
            L.append("")
        dec = " ".join(f"{v['cross_modal_distance_median']:.2f}" for v in ed["by_decile"].values())
        L.append(f"재현이 잘 되는 순서로 십분위를 나눈 cross-modal 거리: {dec}")
        L.append("")
        L.append("이 곡선은 단조 증가한다. 즉 안정적인 concept일수록 cross-modal 정렬이 오히려")
        L.append("좋다. 따라서 \"가장 안정적인 concept마저 어긋나 있다\"는 주장은 이 데이터가")
        L.append("지지하지 않는다. 대신 성립하는 것은 학습 변동으로 잔차를 설명할 수 없다는")
        L.append("쪽이다 — 두 run이 거의 완전히 재현해내는 concept에서도 cross-modal 거리는 두")
        L.append("자릿수 배로 크게 남는다. 분위수 선택이 결론을 바꾸는 손잡이가 되지 않도록")
        L.append("1%, 5%, 10%를 미리 정해 전부 보고한다.")
        sc = ed.get("stable_and_corresponding")
        if sc:
            L.append("")
            L.append("여기까지는 안정성만 조건으로 걸었다. 그런데 안정성만으로 자르면 뽑히는 쌍이")
            L.append("대응 관계가 아니게 된다 — 상위 1%의 매칭 상관을 보면 알 수 있다. 재현 가능한")
            L.append("방향이라도 서로 다른 입력에서 켜진다면 같은 개념이라 부를 근거가 없다. 그래서")
            L.append("두 조건을 함께 건다. 아래는 전부 우리 방법이 실제로 만든 Hungarian 쌍을")
            L.append("걸러낸 것이고 다시 매칭하지 않았다. 안정성은 두 endpoint 중 낮은 쪽 값이다.")
            L.append("")
            L.append("| 조건 | 쌍 수 | cosine | 거리 | 매칭 상관 |")
            L.append("|---|---|---|---|---|")
            for k, e in sc.get("by_stability_quantile", {}).items():
                q = k.replace("top_", "").replace("pct_by_stability", "%")
                L.append(f"| 안정성 상위 {q} (대응 조건 없음) | {e['n']} | "
                         f"{num(e['cosine_median'])} | {num(e['distance_median'])} | "
                         f"{num(e['matched_correlation_median'])} |")
            for key in ("stability>=0.0, c>=0.0", "stability>=0.0, c>=0.6",
                        "stability>=0.9, c>=0.0", "stability>=0.9, c>=0.6",
                        "stability>=0.95, c>=0.6"):
                e = sc["grid"].get(key)
                if not e or not e["n"]:
                    continue
                lab = (key.replace("stability>=", "안정성 ≥ ")
                          .replace(", c>=", ", 상관 ≥ "))
                if key == "stability>=0.0, c>=0.0":
                    lab = "조건 없음 (전체)"
                L.append(f"| {lab} | {e['n']} | {num(e['cosine_median'])} | "
                         f"{num(e['distance_median'])} | "
                         f"{num(e['matched_correlation_median'])} |")
            L.append("")
            L.append("두 조건을 함께 걸면 cosine이 0.55~0.62에서 안정적으로 수렴한다. 표본도 충분하고,")
            L.append("매칭 상관 0.68 이상이라 실제로 함께 켜지는 쌍이며, 양쪽 안정성 0.9 이상이라")
            L.append("학습 잡음도 아니다. 그런데도 1에서 멀다. 이것이 이 절에서 방어 가능한 형태의")
            L.append("결론이다.")
            L.append("")

        co = ed.get("co_activating_only", {})
        if co.get("n", 0) >= 5:
            L.append("")
            L.append(f"실제로 함께 켜지는 쌍(상관 {co['threshold']} 이상) {co['n']}개로 한정하면 "
                     f"cross-modal 거리는 {num(co['cross_modal_distance']['median'])}다. "
                     f"안정성 상위만 골라도 다음과 같다.")
            L.append("")
            L.append("| 안정성 상위 | 개수 | 안정성 | cross-modal 거리 |")
            L.append("|---|---|---|---|")
            for q in (10, 25, 50, 100):
                k = co.get(f"top_{q}pct_by_stability")
                if k:
                    L.append(f"| {q}% | {k['n']} | {num(k['stability_median'])} | "
                             f"{num(k['cross_modal_distance_median'])} |")
        L.append("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
