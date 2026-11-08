# Conditional finite response, not full retraining

Version 0.2 adds a separate [live inspector](https://midya8.github.io/ratinglab/influence.html). It changes no original fit, split, metric, source rating, original score-inference code or original figure.

## The claim is deliberately conditional

Fix the training mean μ, penalty λ > 0 and every opposite-side bias and factor row at one saved candidate. For one user (or item), write its joint bias/factor vector as θ, observed design rows as aᵀ = [1, qᵀ], and centered targets as t = rating − μ − opposite bias. Its problem is

    f(θ) = ‖Aθ − t‖² + λ‖θ‖²
    H = AᵀA + λI, b = Aᵀt, θ* = H⁻¹b.

H is positive definite because vᵀHv = ‖Av‖² + λ‖v‖² > 0 for nonzero v. This conditional minimizer is unique. No inverse matrix is formed in the implementation: solve Hθ*=b and Hz=a.

The final saved item block is the last solved block. A saved user block may no longer solve its own equations after that item pass. Therefore three states are distinct in the inspector:

1. Original saved point and its score.
2. Fresh conditional baseline θ* with opposite parameters fixed.
3. Finite edited/deleted conditional point, measured against state 2.

The saved-to-baseline coordinate movement is exposed, not folded into the observation's response.

## Finite target edit

For a selected existing training row aᵀ and target tᵢ, set z=H⁻¹a and h=aᵀz. Changing only this rating by a finite δ leaves H fixed and changes b to b+aδ. Therefore

    θ_edit = θ* + δz.

For query design qᵀ, the raw-score derivative with respect to that target is qᵀz, and the finite raw-score change is δqᵀz. This is not merely a first-order approximation within this fixed quadratic problem. It says nothing about the response when other blocks are optimized. The browser restricts the edited rating to [0,2]; a continuous value is hypothetical. No source or model is mutated.

## Delete only this row from this block

Deleting aᵀ,tᵢ gives H₋=H−aaᵀ and b₋=b−atᵢ. H₋ is still positive definite because the same positive penalty remains. Sherman-Morrison gives

    θ_delete = θ* − z(tᵢ − aᵀθ*)/(1−h).

The residual at the deleted row after this conditional re-solve is

    tᵢ − aᵀθ_delete = (tᵢ − aᵀθ*)/(1−h).

This last expression is an algebraic deleted-row residual, not a cross-validation performance estimate. Original frozen opposite factors and μ retain information from the row. The procedure does not erase that information, retrain ALS, reselect a hyperparameter, unlearn data, or provide a causal intervention.

## Why the denominator is positive

Let s=aᵀH₋⁻¹a ≥ 0. Applying the rank-one update to H=H₋+aaᵀ yields h=s/(1+s), hence 0≤h<1 and 1−h=1/(1+s). Since H₋≽λI, s≤‖a‖²/λ, so

    1−h ≥ λ/(λ+‖a‖²) > 0.

This covers rank-deficient designs and removal of a block's only row. The latter has θ_delete=0, though its query score still includes the frozen mean and opposite bias. Implementations use binary64 arithmetic, check positive numerical denominators and expose normal residuals. No interval enclosure, exact rational certificate or arbitrary-conditioning guarantee is claimed for real-model readouts. The floor is a theorem of the represented conditional problem, not a probabilistic confidence bound.

## Independent checks

- 60 new small integer/rational designs compare both finite edits and deletions to a separately implemented Fraction Gaussian-elimination oracle, not to the rank-one formula. The existing 60 original ridge references remain.
- All 696 training rows, both block sides and both independently selected factor/bias models produce 2,784 deletion responses. Every one is compared to a newly assembled deleted design and fresh direct ridge solve. Empty-after-deletion blocks are included.
- A central finite-difference query derivative and the deleted-row residual identity are checked separately.
- A synthetic full-ALS refit changes the training mean and opposite factors, documenting that this procedure is not that refit.
- Scalar JavaScript implements its own pivoted elimination and normal-matrix assembly. It checks 576 native vector/leverage/query references and all 2,784 deletion normal residuals and denominator floors. It never imports NumPy or a production Python solve.
- Real Chrome exercises all 576 reference states in the actual form, all eleven models on both sides, physical keyboard edits, invalid-input clearing and recovery into deletion, clipboard failure/success, an actual JSON Blob download and 320/390/1280-pixel layouts.
- Thirty Python and nine Node/real-Chrome tests in total. The installed wheel imports eight modules, runs the original six commands plus two conditional-response module commands, reproduces eleven fresh fits and both full audits, and reproduces all 576 conditional references outside checkout.

## Reproduce

    python -m unittest discover -s tests -v
    node --test tests/*.cjs
    python -m ratinglab.influence demo
    python -m ratinglab.influence references --output /tmp/influence-reference.json

The latter writes explicitly requested reproduction output; the demo is read-only. `block_response` requires a training source-row ID belonging to its chosen block, known integer entity indices, finite deltas and exclusive edit/delete operations. Module commands do not accept arbitrary remote content or invoke an external API.

## Sources and attribution

These are established linear-algebra identities and ridge methods, not a novel influence estimator.

- Petersen and Pedersen, *The Matrix Cookbook*, version November 15, 2012, §3.2.4 Sherman-Morrison: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf . The rank-one inverse identity is specialized above; the positive-penalty deletion denominator argument is given explicitly here.
- Independent derivation of deleted-row residual and coefficient response, OLS setting: https://stats.stackexchange.com/questions/164223/proof-of-loocv-formula . We do not copy the OLS nonsingularity premise: positive ridge regularization keeps deleted systems nonsingular, including a one-row block.
- Will Gao, *Ridge Regression and the Influence of a Single Point*, ridge positive-definiteness and Sherman-Morrison background: https://willhgao.com/notes/linear-regression/ . This secondary note identifies its adaptation from Alexis Drouot's Winter 2025 course and warns that errors may remain. It is background, not the sole support for the derivation or tests.
- Original matrix-factorization background and dataset/selection-bias sources remain in [METHOD.md](METHOD.md). Data license and original byte hash are unchanged.

Conditional leverage is not semantic importance. Derivatives are not causal preferences. A finite conditional response is not a bound on full-model retraining or on recommendation utility. No new test-set claim is introduced by this extension.
