# RatingLab method, source and limits

## Data and permissions

Original source: Ponce Medellín, R. & González Serna, J. (2011). *Restaurant & Consumer Data*. UCI Machine Learning Repository. DOI 10.24432/C5DP41.

- Dataset page: https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data
- Archive linked by that page: https://archive.ics.uci.edu/static/public/232/restaurant+consumer+data.zip
- Original page footer explicitly gives Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/legalcode . Checked against the original HTML on November 8, 2026. Sharing/adaptation allowed with attribution. No endorsement is implied.
- Only `rating_final.csv` is extracted into this repository. SHA256 `fc994ea4e872365fc63e7f0bc970b36f86edeb4e80218d6a674a3d0e51181901`, 22,108 bytes. Original CRLF bytes retained. It has 1,161 distinct user/item pairs and ratings 0,1,2; no duplicate pair is silently averaged.
- There are 138 users and 130 rated items. The original user and item identifiers are sorted and mapped to stable zero-based indices. IDs are lookups, never numeric predictors. No attempt is made to identify a person. The full source identifier universe defines array dimensions, not a learned statistic.
- The source CSV also contains food and service ratings, but the parser ignores them. No other archive files are bundled: no consumer profiles, demographics, religion, location, names or restaurant metadata. Only overall ratings are modeled and exported.

MovieLens100K was considered and rejected because its original README prohibits redistribution without separate permission. No MovieLens data is bundled or used: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt . Jester was not chosen because its original research-use terms did not give equally clear general redistribution scope.

## Split and evaluation

Shuffle original row indices with NumPy `default_rng(20261108)`. First floor(3N/5)=696 are training, next floor(4N/5)-696=232 validation, final 233 test. Sort indices within each partition, preserve original row identity and check exact disjoint coverage. No ratings or timestamps guide the split. There is no timestamp in the used source table, so no chronological or future-prediction claim.

Compute the global mean using training overall ratings only. Train ranks 2/4 × regularization .3/3 × seeds 0/1, fixed 80 complete sweeps. Pick minimum validation raw RMSE, then lexicographic candidate ID for an exact tie. Separately fit rank-zero bias controls with regularization .3/3/30 and the same validation rule. Both choices are fixed before test metrics are computed. No validation refit or post-test model choice.

Use all 233 test observations, not only a convenient warm subset. Count cold entities explicitly: this split happens to have no cold validation/test users or items. Cold fallback is tested on a separate synthetic problem, not advertised as empirical new-user performance. Empty training rows have exactly zero optimal bias and factors under positive ridge; prediction then falls back to the mean and any known opposite bias.

Raw RMSE and MAE are evaluated on observed pairs. Clipped-to-[0,2] RMSE and out-of-range counts are separate diagnostics. Selection uses raw RMSE only. The training-mean control is not a test mean. No top-N ranking, negative sampling, missing-rating truth, ranking quality, propensity adjustment or unbiased recommendation utility is claimed.

## Training objective

For observed TRAINING pairs Ω:

prediction(u,i) = μ + b_u + c_i + p_u dot q_i

L = SUM_(u,i in Ω) (prediction(u,i)-r_ui)^2 + λ(SUM_u b_u² + SUM_i c_i² + ||P||_F² + ||Q||_F²).

μ is fixed at the training mean. Every bias and factor is penalized once in this global objective, not once for every observed edge. No normalization by the number of ratings and no degree-weighted regularization. λ therefore belongs to this explicit convention, not every implementation of ALS. Missing pairs contribute no error term; observed zero ratings do.

Initialize factors with `default_rng(seed).normal(0,.05)`, biases zero. Empty entities are set to zero even at budget zero. Update user rows in increasing index, then item rows in increasing index. For one user, construct A with a column of ones and observed opposite item factors; t = observed r - μ - opposite item bias. Joint bias/factor optimum θ solves:

Hθ = rhs, H=AᵀA+λI, rhs=Aᵀt.

The item update is symmetric. λ>0 makes every block system positive definite, including a block with no observations. NumPy solves the displayed system; there is no black-box matrix-factorization model. Each block's local penalized loss drop equals its change to the global objective because other parameter penalties are fixed. Every update checks decrease within 1e-9(1+old local loss); each half-pass checks the sum of drops against the full objective decrease within 1e-8(1+old global objective). The maximum normal residual over all blocks is retained after each half-pass.

Full analytic gradients include both biases and both factor matrices. A half-pass record contains the full objective, SSE, penalty, full-gradient maximum, block normal residual and total block decrease. Saved histories are numerical summaries, not a complete bitwise block replay or an exact global certificate. A user block need not remain optimal after the subsequent item update; the final item blocks were solved last.

Stop only after a full sweep if max absolute full gradient <=1e-7. Otherwise report `budget_exhausted` after 80 sweeps. All eight factor fits exhaust this criterion. The convex bias-only case is separate; its numerical gradient tolerance does not establish nonconvex global optimality elsewhere. No timing or scalability benchmark is claimed.

## Independent checks and informative controls

- A separate Fraction Gaussian-elimination oracle solves 60 tiny integer-design, dyadic-target, dyadic-positive-ridge problems, without importing NumPy or ALS. All block coordinates agree within 1e-11. These exact references test the ridge solve, not a global optimum of real factorization.
- Central differences check every bias and factor gradient coordinate at a nontrivial fitted state. Scalar JavaScript independently computes all 11×1,161=12,771 saved observed-pair scores and normal systems. A separate Python scalar export is checked against native vectorized predictions within 1e-12.
- Eleven complete fresh fits preserve configuration choices, status, test RMSE within six decimal places and final objective within five decimal places. Floating reductions are not promised bitwise portable.
- A fresh same-runtime fit with all TEST target labels replaced by 99 must have exactly identical models, complete histories and validation metrics to the unpoisoned fresh run. Test metrics change, never fit parameters or selection.
- A supplied 2×2 interaction matrix with mean1, zero biases, rank1, λ=.1 and all-zero factors has exact zero gradient and objective4. User/item rows [+1,-1] give objective.4. Stationarity alone is not global optimality.
- A shared orthogonal rotation P→PR and Q→QR preserves every dot product and Frobenius penalty, while individual coordinate products change. They cannot be identified semantic/causal explanations.
- Cold-start zero-row behavior, zero budget, duplicate edges, invalid indices/configuration, strict JSON, tampered metric and source/split audits are checked.
- Actual Chrome checks every saved scalar inference, all eleven fits and 33 UI pair states, physical keyboard rotation, normal systems, invalid-input recovery, blocked/successful clipboard, an actual JSON Blob download and 320/390/1280 layouts, including reaching the final table column.

Synthetic controls are descriptive algebra examples, not causal ablations. The historical fit is not a claim to recover the original research prototype or reproduce a published score. Source authors' original top-N task is not our evaluated task.

## Sources and provenance

- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer. Primary reference for observed-only factor models, user/item biases, regularization and alternating quadratic blocks. Read equations 1-5 and the ALS section. Our stated once-per-parameter regularizer is the implementation convention; not all equation conventions are identical. https://chrisvolinsky.com/files/publications/ieeecomputer2009.pdf
- Dive into Deep Learning, section21.3, *Matrix Factorization*. Independent book cross-check of biased factor models, observed training pairs and RMSE. No framework implementation copied. https://www.d2l.ai/chapter_recommender-systems/mf.html
- HPI collaborative-filtering lecture, *Matrix Factorization Techniques for Recommender Systems*. Secondary context for bias terms and fit objectives. https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/lehre/SS2011/Collaborative_Filtering/pres1-matrixfactorization.pdf
- Marlin & Zemel, *Recommender Systems: Missing Data and Statistical Model Estimation*. Primary discussion of missing-not-at-random data and the failure of observed-only tests to establish missing-pair truth. https://www.cs.toronto.edu/~zemel/documents/2011-IJCAI-MZRS.pdf
- Steck (2013), *Evaluation of Recommendations: Rating-Prediction and Ranking*, RecSys. Evaluation distribution matters; RMSE and top-N tasks are not interchangeable. https://doi.org/10.1145/2507157.2507160
- Original UCI page, archive, source README and license as above. Original source credits Rafael Ponce Medellín and Juan Gabriel González Serna; public site citation shortens the names. Attribution does not imply endorsement.

MIT original code and synthetic controls. CC BY4.0 original UCI ratings. No paid API, private data, user profiling, causal preference model, calibrated certainty, missing-pair guarantee or restaurant advice.
