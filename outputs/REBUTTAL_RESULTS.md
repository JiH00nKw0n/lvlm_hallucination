# Rebuttal measurements — coco_k8, runs r1r2

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
| image SAE, two runs | 849/850 | 513 | 0.082 | [0.076, 0.091] |
| text SAE, two runs, same caption | 436/425 | 370 | 0.070 | [0.065, 0.073] |
| text SAE, two runs, different captions | 436/425 | 103 | 0.067 | [0.061, 0.072] |
| image vs text (the paper's measurement) | 849/436 | 73 | 0.495 | [0.451, 0.539] |
| random directions | -- | -- | 1.00 | -- |

Taken latent by latent over the 68 image latents that clear the
threshold in both comparisons, crossing modality costs 0.403 more distance than a second training run does (95% CI [0.380, 0.427], Wilcoxon p=7.6e-13).

Repeating the whole comparison for each pairing of the three runs:

| runs | img x img | txt x txt | txt, different captions | img x txt | paired difference |
|---|---|---|---|---|---|
| r1r2 | 0.082 | 0.070 | 0.067 | 0.495 | 0.403 |
| r1r3 | 0.059 | 0.058 | 0.055 | 0.495 | 0.432 |
| r2r3 | 0.081 | 0.013 | 0.008 | 0.519 | 0.463 |

## Do matched latents stand for the same concept?

*Reviewer PBPC Q2; AC on match quality.*

Across 65 COCO object categories with enough annotated
examples, the image latent and the text latent chosen independently from
disjoint halves of the photographs land on permutation-matched coordinates:

| | hit@1 | hit@5 | hit@10 | median rank | MRR |
|---|---|---|---|---|---|
| learned permutation | 69.2% | 90.8% | 95.4% | 1 of 499 | 0.7715 |
| random permutation | 0.2% | -- | -- | -- | -- |
| category labels shuffled | 1.5% | -- | -- | -- | -- |
| chance | 0.2% | -- | -- | -- | -- |
| image side vs its other half (ceiling) | 95.4% | -- | -- | -- | -- |

p against the random permutation: 0.0000. 54 distinct image latents were chosen across 65 categories, so the result is not a few busy latents winning everything.

## How strong and how unambiguous are the matches?

*Reviewer PBPC Q3.*

Over 499 matched pairs, the co-activation correlation has
median 0.242, quartiles [0.097, 0.432], and
5th-95th percentile [0.019, 0.767].

| share of matches below | c<0.05 | c<0.1 | c<0.2 | c<0.3 | c<0.4 | c<0.6 |
|---|---|---|---|---|---|---|
| | 14.4% | 25.5% | 44.1% | 60.7% | 73.3% | 87.0% |

The runner-up is within 10% of the assigned partner for 18.0% of matches. 42.5% of pairs are each other's first choice.

Destroying the image-caption pairing and recomputing puts the noise floor at correlation 0.0295 (99th percentile of matches found in noise); 8.8% of real matches fall below it.

Weak matches are not merely noted, they are identifiable and worth removing.
Keeping only matches above a correlation cutoff and rerunning cross-modal
retrieval on the held-out split:

| kept | coordinates | I->T R@1 | I->T R@5 | T->I R@1 | partners shuffled, I->T R@1 |
|---|---|---|---|---|---|
| c>=0.0 | 499 | 3.48 | 9.82 | 1.18 | 0.02 |
| c>=0.1 | 372 | 10.22 | 25.04 | 8.38 | 0.02 |
| c>=0.2 | 279 | 9.64 | 23.16 | 7.55 | 0.00 |
| c>=0.3 | 196 | 7.26 | 17.18 | 6.15 | 0.00 |
| c>=0.4 | 133 | 5.56 | 12.36 | 4.59 | 0.00 |
| c>=0.6 | 65 | 2.48 | 5.54 | 1.82 | 0.00 |

The shuffle keeps the same coordinates and only breaks which text latent each
is paired with, so its collapse to near zero shows the retrieval is carried by
the correspondence rather than by how much those coordinates activate.

## Is the gap just feature splitting?

*AC's third concern.*

One-to-many groups — an image latent correlating above 0.4 with two or
more text latents — cover 3.9% of alive image
latents (36 groups). Share of the image direction's energy that the
text partners explain:

| subspace | median explained |
|---|---|
| all partners | 0.231 |
| strongest partner alone | 0.192 |
| strongest partner + random text atoms | 0.195 |
| random text atoms | 0.006 |
| random directions | 0.004 (analytic 0.005) |

The split partners add 0.025 beyond the
strongest one, against 0.003 for the control, so
the effect is real but small: 76.9% of the image
direction stays unexplained. 0.0% of groups exceed half explained.

## Could any matching, or one global transform, close the gap?

Ignoring the matching entirely and taking each image direction's closest text
direction anywhere in the dictionary gives median cosine 0.281 (distance 0.719). The same search over random directions gives 0.132, and the analytic value for a maximum over that many candidates is 0.156. No assignment procedure can do better than the oracle.

Fitting one transform on half the matched pairs and scoring it on the other half:

| | held-out cosine |
|---|---|
| identity | 0.306 |
| best rotation | 0.176 |
| best linear map | 0.142 |
| rotation fitted on shuffled pairs | 0.010 |

## Does the gap survive on concepts both runs reproduce?

*AC on dictionary non-identifiability.*

Mean stability between the two runs' image dictionaries: 0.718 over 499 concepts.

| stability cut | n | stability | same-modality distance | cross-modal distance | matched correlation |
|---|---|---|---|---|---|
| top_1pct | 5 | 0.992 | 0.008 | 0.363 | 0.603 |
| top_5pct | 25 | 0.982 | 0.018 | 0.446 | 0.524 |
| top_10pct | 50 | 0.975 | 0.025 | 0.480 | 0.486 |
| top_25pct | 125 | 0.965 | 0.035 | 0.511 | 0.439 |
| top_50pct | 250 | 0.941 | 0.059 | 0.567 | 0.381 |
| top_100pct | 499 | 0.862 | 0.138 | 0.697 | 0.242 |

Decile curve, most reproducible first: 0.47 0.54 0.60 0.64 0.61 0.79 0.84 0.81 0.82 0.81

Restricted to the 65 pairs that genuinely co-activate (correlation >= 0.6), the cross-modal distance is 0.487.

