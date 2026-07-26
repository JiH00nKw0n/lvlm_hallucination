# Rebuttal measurements — cc3m_k32, runs r1r2

All numbers come from SAEs trained for this response; no checkpoint from the
submitted work is reused. Runs differ only by seed, and the seed controls the
initialization as well as the data order.

The correlation matrices come from the same function that produced the
submitted paper's figure. Re-running it through the new code path on the
original checkpoint reproduced the stored matrix to a maximum absolute
difference of 3e-4, which is the rounding from storing it in half precision.

## Is the reported gap larger than ordinary training variability?

*Reviewer PBPC Q1; AC's first concern.*

Cosine distance between matched feature directions, restricted to pairs whose
co-activation correlation clears the threshold. Each latent contributes one
value (the median over its own qualifying pairs) so that frequently firing
latents cannot dominate.

| comparison | alive | pairs at c>=0.6 | median distance | 95% CI |
|---|---|---|---|---|
| image SAE, two runs | 2584/2612 | 2154 | 0.111 | [0.103, 0.123] |
| text SAE, two runs, same caption | 2404/2433 | 2232 | 0.104 | [0.096, 0.115] |
| text SAE, two runs, different captions | -- | -- | -- | -- |
| image vs text (the paper's measurement) | 2584/2404 | 189 | 0.517 | [0.485, 0.551] |
| random directions | -- | -- | 1.00 | -- |

Taken latent by latent over the 158 image latents that clear the
threshold in both comparisons, crossing modality costs 0.360 more distance than a second training run does (95% CI [0.313, 0.386], Wilcoxon p=1.3e-25).

## Do matched latents stand for the same concept?

*Reviewer PBPC Q2; AC on match quality.*

Across 65 COCO object categories with enough annotated
examples, the image latent and the text latent chosen independently from
disjoint halves of the photographs land on permutation-matched coordinates:

| | hit@1 | hit@5 | hit@10 | median rank | MRR |
|---|---|---|---|---|---|
| learned permutation | 44.6% | 69.2% | 70.8% | 2 of 3209 | 0.5533 |
| random permutation | 0.0% | -- | -- | -- | -- |
| category labels shuffled | 0.0% | -- | -- | -- | -- |
| chance | 0.0% | -- | -- | -- | -- |
| image side vs its other half (ceiling) | 96.9% | -- | -- | -- | -- |

p against the random permutation: 0.0000. 49 distinct image latents were chosen across 65 categories, so the result is not a few busy latents winning everything.

## How strong and how unambiguous are the matches?

*Reviewer PBPC Q3.*

Over 3209 matched pairs, the co-activation correlation has
median 0.147, quartiles [0.048, 0.322], and
5th-95th percentile [0.006, 0.598].

| share of matches below | c<0.05 | c<0.1 | c<0.2 | c<0.3 | c<0.4 | c<0.6 |
|---|---|---|---|---|---|---|
| | 26.2% | 40.1% | 58.8% | 72.3% | 82.6% | 95.1% |

The runner-up is within 10% of the assigned partner for 28.3% of matches. 28.2% of pairs are each other's first choice.

Destroying the image-caption pairing and recomputing puts the noise floor at correlation 0.0749 (99th percentile of matches found in noise); 33.8% of real matches fall below it.

Weak matches are not merely noted, they are identifiable and worth removing.
Keeping only matches above a correlation cutoff and rerunning cross-modal
retrieval on the held-out split:

| kept | coordinates | I->T R@1 | I->T R@5 | T->I R@1 | partners shuffled, I->T R@1 |
|---|---|---|---|---|---|
| c>=0.0 | 3163 | 18.12 | 37.16 | 11.14 | 0.00 |
| c>=0.1 | 1923 | 16.06 | 35.32 | 12.18 | 0.04 |
| c>=0.2 | 1323 | 15.66 | 33.18 | 12.42 | 0.02 |
| c>=0.3 | 888 | 12.94 | 27.86 | 10.54 | 0.02 |
| c>=0.4 | 558 | 9.44 | 21.38 | 7.35 | 0.04 |
| c>=0.6 | 157 | 2.66 | 6.76 | 1.80 | 0.00 |

The shuffle keeps the same coordinates and only breaks which text latent each
is paired with, so its collapse to near zero shows the retrieval is carried by
the correspondence rather than by how much those coordinates activate.

## Is the gap just feature splitting?

*AC's third concern.*

One-to-many groups — an image latent correlating above 0.4 with two or
more text latents — cover 6.1% of alive image
latents (220 groups). Share of the image direction's energy that the
text partners explain:

| subspace | median explained |
|---|---|
| all partners | 0.213 |
| strongest partner alone | 0.111 |
| strongest partner + random text atoms | 0.113 |
| random text atoms | 0.005 |
| random directions | 0.004 (analytic 0.005) |

The split partners add 0.027 beyond the
strongest one, against 0.002 for the control, so
the effect is real but small: 78.7% of the image
direction stays unexplained. 15.9% of groups exceed half explained.

## Could any matching, or one global transform, close the gap?

Ignoring the matching entirely and taking each image direction's closest text
direction anywhere in the dictionary gives median cosine 0.291 (distance 0.709). The same search over random directions gives 0.155, and the analytic value for a maximum over that many candidates is 0.178. No assignment procedure can do better than the oracle.

Fitting one transform on half the matched pairs and scoring it on the other half:

| | held-out cosine |
|---|---|
| identity | 0.232 |
| best rotation | 0.139 |
| best linear map | 0.143 |
| rotation fitted on shuffled pairs | 0.002 |

## Does the gap survive on concepts both runs reproduce?

*AC on dictionary non-identifiability.*

Mean stability between the two runs' image dictionaries: 0.613 over 3209 concepts.

| stability cut | n | stability | same-modality distance | cross-modal distance | matched correlation |
|---|---|---|---|---|---|
| top_1pct | 32 | 0.996 | 0.004 | 0.122 | 0.077 |
| top_5pct | 160 | 0.991 | 0.009 | 0.356 | 0.099 |
| top_10pct | 321 | 0.985 | 0.015 | 0.413 | 0.190 |
| top_25pct | 802 | 0.967 | 0.033 | 0.575 | 0.263 |
| top_50pct | 1604 | 0.921 | 0.079 | 0.682 | 0.231 |
| top_100pct | 3209 | 0.735 | 0.265 | 0.848 | 0.147 |

Decile curve, most reproducible first: 0.41 0.61 0.68 0.76 0.80 0.84 0.86 0.93 0.95 0.98

Restricted to the 157 pairs that genuinely co-activate (correlation >= 0.6), the cross-modal distance is 0.517.

