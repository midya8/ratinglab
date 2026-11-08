# Joint conditional batch responses

Version 0.3.0 adds finite target edits and joint row deletion within one frozen ridge block. This is an educational implementation of established ridge and Woodbury algebra, not a new estimator or a full-ALS retraining shortcut.

## Three distinct states

A saved user block can cease to solve its own normal equations after the last item pass. Re-solve the original block first. Every reported batch change starts from that conditional baseline, not silently from the saved point. Keep the training mean, ridge penalty and all opposite-side biases/factors fixed. Both the block bias and factors carry the same positive ridge penalty.

The source wrapper accepts original training row IDs within exactly one selected user/item block. Validation/test rows, foreign-block rows, duplicates, Boolean/fractional IDs and invalid rating edits are refused. Edits may be continuous hypothetical targets but must keep each rating in [0,2]. Source data, split, saved fits and model selection are not changed.

## Derivation

Let A contain the full block design, t its fixed-offset targets, H = AᵀA + λI and θ = H⁻¹Aᵀt. Let B contain the selected original rows, with distinct original indices. Compute Z by solving HZ = Bᵀ, K = BZ, S = I − K and e = tₛ − Bθ.

- Target edits δ: θ_edit = θ + Zδ. The design and H are unchanged. Separate-original-baseline edits add exactly in mathematical arithmetic.
- Joint deletion: θ_delete = θ − ZS⁻¹e. Solve S c = −e, then θ_delete = θ + Zc.
- Separate deletion contrast: θ_separate = θ − Z diag(S)⁻¹e. Each deletion starts at the same original baseline. This is not a sequential deletion algorithm. Joint-minus-separate is generally nonzero.

For a direct proof, the deleted normal system is H₋ = H − BᵀB and rhs₋ = Aᵀt − Bᵀtₛ. Since H₋Z = Bᵀ(I − K) = BᵀS, substitution gives H₋(θ − ZS⁻¹e) = rhs₋. The positive ridge makes H₋ positive definite, so this is its unique mathematical solution.

### Positive Schur system and bound

The full ridge hat matrix A H⁻¹ Aᵀ has largest eigenvalue ‖A‖₂²/(λ+‖A‖₂²). K is its principal submatrix, so λ_min(S) ≥ λ/(λ+‖A‖₂²) ≥ λ/(λ+‖A‖F²) > 0. This also covers deleting all rows: the remaining objective is λ‖θ‖² and its mathematical minimizer is zero. The floating-point answer is not forced to exact zero.

Empty selection returns the baseline. A single selected row recovers the rank-one formula. Row order changes coordinates of the small system, not the mathematical updated point. Zero design columns and rank-deficient observation designs are allowed when the penalized numerical systems remain usable.

## Numerical contract

This is binary64 numerical algebra, not rational, interval or exact arithmetic. Python uses NumPy solves and a symmetric Schur eigensystem check. The independent browser implementation uses Cholesky solves. Both refuse deletion when numerical positivity is lost, the Schur infinity-norm condition exceeds 10¹², or the scaled direct normal residual exceeds 10⁻¹⁰. No jitter or silent repair is applied. Backend refusal boundaries need not agree on ill-conditioned inputs.

The update never explicitly forms H⁻¹ or S⁻¹. The separate condition diagnostic evaluates the inverse action on identity columns (NumPy's infinity-norm condition routine on Python; independent triangular solves in JavaScript). Target edits do not need a Schur solve or positive computed Schur eigenvalues. Native export still reports its diagnostics; a numerically unusable deletion may be refused even when an edit is allowed.

Residual checks rebuild the changed normal equations from remaining design rows, not by subtracting a nearly cancelling Gram matrix. The scaling is

`1 + max(abs(rhs_changed)) + d * max(abs(H_changed)) * max(abs(point))`.

A small residual does not prove small forward error, a good full retrain approximation or predictive value. The lower eigenvalue bound is a mathematical theorem; its floating display is not a certified lower interval endpoint. The native and browser normal solves may accumulate roundoff differently.

## Independent checks

- 320 tiny rational cases: ten half-integer 4×3 designs, all 16 subsets, edits and deletions. An independent Fraction elimination solver builds direct retained-row normal systems. These are not just comparisons between two copies of Woodbury.
- 128 selected-factor/bias source-bound cases compare to direct augmented least squares, with √λI rows and zero targets.
- 320 complete deterministic source-bound responses plus four controls compare native NumPy and independent browser Cholesky: both selected fits, user/item blocks, empty/single/two/three/full selections, edits and deletions. This finite collection is not every possible batch.
- A three-row repeated-design control has joint-minus-separate query change −0.675, exposing deletion nonadditivity.
- Empty/full/single deletion, row permutation, target-edit additivity, lost-positivity refusal, saved-data/model nonmutation, held-out-label poisoning, strict duplicate-key JSON and CLI tests.
- Actual Chrome checks all 320 native responses, 72 UI combinations, keyboard selection, invalid edit refusal, clipboard denial/success, actual JSON Blob download and 320/390/1280 layouts.

The original single-row 2,784 deletions, 576 references, fresh fitting, gradient, split and original browser tests stay in the suite.

## Reproduce offline

```sh
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m ratinglab batch --input examples/batch.json
python -m ratinglab batch-export > /tmp/batch-reference.json
python -c "import json; assert json.load(open('/tmp/batch-reference.json')) == json.load(open('ratinglab/artifacts/batch-reference.json'))"
```

Strict input keys are `design`, `target`, `regularization`, `rows`, `query`, plus optional `deltas` and `delete`. `rows` indexes the supplied original design, not the source dataset. At most 33 columns and 64 selected rows are accepted. Use `ratinglab.batch.block_response` for dataset-source IDs. Duplicate keys, nonfinite JSON, extra fields, Booleans in numeric arrays and simultaneous edit/deletion are refused.

```python
from ratinglab.batch import response
w = response([[1,1],[1,1],[1,0]], [1,2,0], 1., [0,1], [1,1], delete=True)
assert abs(w['interaction'] + .675) < 1e-12
```

Regenerate the compact package artifact with `python -c "from ratinglab.batch_export import write; write('ratinglab/artifacts/batch-reference.json')"`. The full witness records the original numerical problem, source/block IDs, selected ratings, baseline, matrices, coefficients, changed point, separate contrast and numerical residuals. It is explanatory evidence, not an authentication mechanism or a security boundary.

## Limits

This is not full-model leave-many-out cross-validation: the opposite factors and original training mean were learned using the deleted observations. It is not data unlearning, causal influence, uncertainty, attribution to people, recommendation utility or preference estimation. There is no speed benchmark or claimed acceleration. These small blocks may be cheaper to solve directly than by a Schur system. No new performance result or test-based model selection is introduced. Existing data licensing, attribution and the slightly stronger bias-only test control remain visible.

## Sources

- Nick Higham, 2020-09-29, [Sherman-Morrison-Woodbury formula](https://nhigham.com/2020/09/29/what-is-the-sherman-morrison-woodbury-formula/): low-rank systems, nonsingularity and factorization rather than explicit inversion.
- Thomas Lienart, [CV Ridge, part II](https://tlienart.github.io/posts/2019/12/09-ridge-cv-2/): deleting multiple rows and recursive update tradeoffs. Its displayed XᵀDₛy line has an apparent notation slip; the direct removed-row normal equations are used and independently tested here.
- Liland, Skogholt and Indahl, [A new formula for faster computation of k-fold cross-validation and good regularisation parameter values in Ridge Regression](https://ar5iv.labs.arxiv.org/html/2211.15128): ridge as augmented least squares and segmented residual systems. Its centered/re-estimated-intercept treatment is different from our frozen mean plus penalized block bias. Its cross-validation claims do not transfer to this conditional ALS calculation.
- John Fox and Georges Monette, [cv technical notes](https://gmonette.github.io/cv/articles/cv-notes.html): Woodbury and direct CV comparisons.
- [MATH3714 influence notes](https://seehuhn.github.io/MATH3714/S08-influence.html): single-observation influence background.
- [Cross Validated multi-observation removal discussion](https://stats.stackexchange.com/questions/449492/least-squares-removing-first-k-observations-woodbury-formula): community derivation, lower authority than the primary paper and numerical linear algebra reference.
