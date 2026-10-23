# CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CausalGBM** is a two-stage framework for fair tabular classification. Stage 1 learns a directed acyclic graph (DAG) over features, the protected attribute and the outcome with DAGMA, and scores each feature by predictive utility minus proxy strength. Stage 2 trains gradient boosting on the features that pass the score threshold. The result is a ranked list of *candidate proxy features* for audit, plus a classifier trained without them.

> 📄 **Paper:** Divyansh Bhatia and N. Mehala. *CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection.* Proceedings of the 18th Asian Conference on Machine Learning (ACML 2026), PMLR 353, 2026.


## Key results (camera-ready, Table 2)

Mean over 10 seeds, 70/30 stratified splits, default configuration (α = 0.5, τ = 0.2, m = max(3, ⌊0.2d⌋)). EOD = Equalized Odds Difference (lower is fairer).

| Dataset | XGBoost EOD | CausalGBM EOD | Reduction | CausalGBM AUC | Significant (p < 0.05) |
|---|---|---|---|---|---|
| ACS Income | 0.071 | **0.008** | 85%† | 0.859 | ✓ |
| Adult | 0.067 | **0.030** | 55% | 0.868 | ✓ |
| Synth. Loan | 0.303 | **0.049** | 84% | 0.646 | ✓ |
| Synth. Hiring | 0.244 | **0.049** | 80% | 0.673 | ✓ |
| COMPAS | 0.189 | **0.162** | 14% | 0.679 | ✓ (not after Nadeau–Bengio correction) |
| Taiwan Credit | 0.030 | **0.019** | 37% | 0.742 | – |
| Student Perf. | 0.161 | **0.103** | 36% | 0.705 | – |
| Bank | 0.096 | 0.085 | 11% | 0.940 | – |
| Online Shoppers | 0.023 | 0.026 | −13% | 0.881 | – |
| German | 0.097 | 0.097 | 0% (rollback) | 0.782 | – |

† Relative to XGBoost trained without RAC1P, a near-copy of the protected attribute (EOD = 0.052). Relative to the full-feature XGBoost row the reduction is 89%.

Across 10 datasets and 17 baselines (including FairGBM and M²FGB), CausalGBM has the lowest EOD on 7 of 10 at default configurations, at a relative AUC cost of 3–14%. At matched AUC, in-processing methods remain competitive on several datasets: the advantage is regime-dependent (proxy-driven unfairness favours CausalGBM; label bias, e.g. COMPAS, favours in-processing). Runtime is 1.9 s (Adult) and 5.6 s (ACS Income), 33–56× faster than transformer baselines.

## How it works

```
Stage 1  Z = [X, A, Y]  ──DAGMA──▶  weighted DAG  W
         proxy strength   W'_{A→Xj} = max(|W_{A→Xj}|, |corr(Xj, A)|)
         score            c_j = W_{Xj→Y} − α · W'_{A→Xj}
         keep             K = { j : c_j ≥ τ }, at least m = max(3, ⌊0.2d⌋) features
Stage 2  train XGBoost on X[:, K]
         if cross-validated AUC < 0.60 → roll back to all features
```

- **DAG-based detection** ranks features by joint conditional predictive utility rather than marginal correlation, which is where most of the gain comes from (ablation: DAGMA beats independent OLS/Lasso/Ridge on 6/8 datasets).
- **Max-aggregation** gives a conservative association floor when the learned edge weight is unreliable (e.g. categorical features in a linear SEM). It cannot suppress a spurious correlation; whether such a feature is retained depends on its predictive-utility term and the minimum-feature rule.
- **Minimum-feature floor and AUC rollback** prevent model collapse under aggressive removal.

The method detects only *direct* proxies (A → X → Y). Indirect proxies on longer paths are a documented limitation (paper §4.10); the bottleneck is structure learning, not the scoring rule.

## Quick start

```bash
git clone https://github.com/dnbresearch/CausalGBM.git
cd CausalGBM
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```python
from causalgbm import CausalGBM

model = CausalGBM(
    alpha=0.5,          # proxy penalty weight
    tau=0.2,            # score threshold
    min_features=None,  # defaults to max(3, floor(0.2 * d))
)
model.fit(X_train, A_train, y_train)      # A = protected attribute (binary)
y_pred = model.predict(X_test)

print("Selected:", model.get_selected_features())
print("Candidate proxies removed:", model.get_removed_features())
```

The removed list is an **audit aid, not a certificate**: on real data these are candidates for human review, and removing them does not guarantee fairness when unfairness comes from the labels or from indirect paths the DAG does not recover.

## Repository layout

```
CausalGBM/
├── models.py                        # CausalGBM implementation
├── data_utils.py                    # Data loading and preprocessing
├── training.py                      # Training and evaluation utilities
├── causalgbm_experiments_v2.py      # Main experiment runner
├── run_experiment.py                # Single experiment runner
├── complete_benchmark.py            # Full benchmark suite
├── analysis.py                      # Results analysis and visualization
├── dag_visualization.py             # DAG visualization tools
├── min_features_ablation.py         # Min-feature floor ablation
├── evaluate_indirect_proxy.py       # Indirect-proxy limitation analysis (paper §4.10)
├── demo.py                          # Quick demonstration
├── synthetic_loan_data.csv          # Synthetic loan dataset (direct proxies)
├── synthetic_hiring_data.csv        # Synthetic hiring dataset (direct proxies)
├── synthetic_indirect_proxy_loan.csv# Synthetic loan dataset with indirect proxies
├── requirements.txt
└── README.md
```

## Datasets

**Real-world (Table 1 in the paper).** ACS Income is from Folktables (Ding et al., 2021); Adult, Bank Marketing, Taiwan Credit, Online Shoppers, German Credit and Student Performance are from the UCI repository; COMPAS is the ProPublica release. Multi-category protected attributes are binarised (COMPAS: African-American vs. other; ACS Income: White vs. non-White from RAC1P, which stays in X as a feature). The *Weekend* attribute on Online Shoppers is not a legally protected attribute; it is included as a sensitivity test.

| Dataset | n | d | Protected | Baseline EOD |
|---|---|---|---|---|
| ACS Income | 195,665 | 10 | Race | 0.071 |
| Adult | 48,842 | 11 | Sex | 0.067 |
| Bank | 41,188 | 20 | Age | 0.096 |
| Taiwan Credit | 30,000 | 22 | Sex | 0.030 |
| Online Shoppers | 12,330 | 17 | Weekend | 0.023 |
| COMPAS | 5,278 | 6 | Race | 0.189 |
| German | 1,000 | 20 | Sex | 0.097 |
| Student Perf. | 649 | 29 | Sex | 0.161 |

**Synthetic (included).** Both have known ground-truth structure with three feature types: legitimate causes of Y, proxies mediating A → Y, and spurious correlates linked to A only through an unobserved confounder C (no path to Y).

| File | n | d | Purpose |
|---|---|---|---|
| `synthetic_loan_data.csv` | 10,000 | 8 | Validation (paper §4.7) |
| `synthetic_hiring_data.csv` | 10,000 | 10 | Validation (paper §4.7, Figure 3) |
| `synthetic_indirect_proxy_loan.csv` | 10,000 | — | Indirect-proxy limitation (paper §4.10) |

On the two validation sets CausalGBM removes every ground-truth proxy and retains every spurious correlate across all 10 seeds, whereas correlation-based selection (|ρ| > 0.3) removes 2/2 spurious correlates on Synth. Loan, sacrificing 8% AUC for no fairness benefit.

The indirect-proxy set has 2 direct proxies, 2 indirect proxies (both with |ρ| < 0.1 to A) and 3 legitimate features. CausalGBM reaches EOD 0.079 vs. 0.092 for an oracle that removes exactly the four proxies, because it additionally drops a low-value spurious correlate; influence-propagation variants (total effects, k-hop, max-propagation) do not help, because DAGMA's linear SEM assigns zero weight to feature-to-feature edges.

## Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `alpha` | 0.5 | Proxy penalty. Selections are identical for α ∈ [0.25, 3.0] at the default τ; EOD varies < 0.025 for α ∈ [0.25, 1.0] across 5 datasets. In practice α acts as a hard threshold; use τ or `min_features` for fine control. |
| `tau` | 0.2 | Score threshold. |
| `min_features` | max(3, ⌊0.2d⌋) | Floor on retained features. ⌊0.2d⌋ and ⌊d/3⌋ coincide (m = 3) on ACS, Adult, COMPAS and both synthetics; on larger-d datasets the floor trades AUC against EOD and should be validated per dataset. |
| `dag_iterations` | 500 | DAGMA iterations. |
| rollback threshold | 0.60 | Cross-validated AUC below which all features are restored. |

## When to use it

Use CausalGBM when baseline EOD exceeds ~0.03 and unfairness is at least partly proxy-driven. In the paper's benchmarks, three signals available before deployment each predicted a failure case:

- baseline EOD below 0.03 (Online Shoppers: removal cannot help),
- n below 5,000 (German, Student Performance: DAG unreliable, AUC degrades),
- cross-validated AUC below 0.60 after selection (German: rollback).

When unfairness stems from label bias (e.g. COMPAS), in-processing methods such as FairGBM achieve better fairness at matched AUC. Monitor worst-group accuracy alongside EOD: proxy removal can reduce predictive power for the worst-off group (ACS WGA 0.822 → 0.777).

**Known limitations:** only direct A → X proxies; linear SEM may miss nonlinear proxies; DAGs identified only up to Markov equivalence; O(d³) DAGMA cost is prohibitive for d > 100; single binary protected attribute.

## Reproducing the paper

All paper results use **10 seeds** unless stated otherwise (the structure-learning comparison in §4.5 uses 5). Every feature-selection step — standardisation, DAG learning, scoring, thresholding, the minimum-feature rule and the rollback check — uses training data only; the test split is used solely for reporting.

<!-- AUTHORS: confirm each script → table mapping. Entries marked (?) are
     inferred from file names and must be checked against the tagged commit. -->

| Paper artefact | Script | Notes |
|---|---|---|
| Table 2 (CausalGBM vs XGBoost) | `causalgbm_experiments_v2.py --main_comparison` | |
| Table 3 (vs FairGBM, M²FGB) | `causalgbm_experiments_v2.py --main_comparison` (?) | baselines at published defaults |
| Table 4 (matched-AUC, SeqSel) | `acml_matched_auc.py` (?) | interpolated on sampled frontiers |
| Table 5 (DAGMA vs regression) | `acml_regression_ablation.py` (?) | rank-normalised scoring |
| Acyclicity ablation (§4.4) | `acml_acyclicity_ablation.py` (?) | |
| Aggregation ablation (§4.4) | `causalgbm_experiments_v2.py --ablation` | |
| Structure-learning comparison (§4.5) | (?) | 6 algorithms, 5 seeds, 3 datasets |
| Hyperparameter validation (§4.6) | `causalgbm_experiments_v2.py --sensitivity`, `min_features_ablation.py` | |
| Synthetic validation (§4.7) | `causalgbm_experiments_v2.py --dag_recovery` | |
| Indirect-proxy analysis (§4.10) | `evaluate_indirect_proxy.py` | |
| Figure 2 (Pareto frontiers) | (?) | α, λ, γ sweeps |

### Command-line interface (`causalgbm_experiments_v2.py`)

| Argument | Description | Default |
|---|---|---|
| `--all` | Run all experiments | off |
| `--main_comparison` | Main method comparison | off |
| `--dag_recovery` | DAG recovery on synthetic data | off |
| `--sensitivity` | Hyperparameter sensitivity | off |
| `--ablation` | Aggregation ablation (DAG / correlation / max) | off |
| `--datasets` | Datasets to run | all |
| `--synthetic_loan_path`, `--synthetic_hiring_path` | Paths to synthetic CSVs | auto-detect |
| `--output_dir` | Results directory | `results/` |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--quick` | Reduced seed count for smoke tests | off |
| `--max_samples` | Cap on samples per dataset | 200000 |

```bash
# Smoke test (a few minutes)
python causalgbm_experiments_v2.py --main_comparison --quick \
    --datasets adult synthetic_loan --output_dir results/smoke

# Full main comparison (Table 2; several hours on CPU)
python causalgbm_experiments_v2.py --main_comparison --output_dir results/table2

# Synthetic validation and aggregation ablation
python causalgbm_experiments_v2.py --dag_recovery --ablation \
    --datasets synthetic_loan synthetic_hiring --output_dir results/synthetic

# Indirect-proxy analysis
python evaluate_indirect_proxy.py
```

Outputs are written to `--output_dir`: `main_comparison_results.csv`, `dag_recovery_results.csv`, `feature_selection_results.csv`, `ablation_results.csv`, `sensitivity_analysis_results.csv` and a human-readable `experiment_summary.txt`.

### Reading the feature analysis

```
Feature          |ρ(X,A)|   W_A→X    W_X→Y    score    selected
relationship      0.65     0.82     0.31    -0.10       ✗
marital-status    0.45     0.56     0.28     0.00       ✗
education         0.12     0.08     0.45    +0.41       ✓
occupation        0.18     0.15     0.38    +0.30       ✓
hours-per-week    0.08     0.05     0.22    +0.19       ✓
age               0.06     0.04     0.35    +0.33       ✓
```

A feature is kept when its predictive weight outweighs α times its proxy strength. A high |ρ| with a positive score means the feature is predictive despite its association with A — on ACS Income, SCHL (education) has W' = 0.072 but W_X→Y = 0.889 and is correctly retained.

## Citation

```bibtex
@inproceedings{bhatia2026causalgbm,
  title     = {{CausalGBM}: Achieving Fairness in Tabular Classification via Causal Feature Selection},
  author    = {Bhatia, Divyansh and Mehala, N.},
  booktitle = {Proceedings of the 18th Asian Conference on Machine Learning (ACML 2026)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {353},
  publisher = {PMLR},
  year      = {2026}
}
```

## References

- Bello, Aragam, Ravikumar (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. NeurIPS.
- Chen, Guestrin (2016). XGBoost: A scalable tree boosting system. KDD.
- Hardt, Price, Srebro (2016). Equality of opportunity in supervised learning. NeurIPS.
- Cruz, Belém, Bravo, Saleiro, Bizarro (2023). FairGBM: Gradient boosting with fairness constraints. ICLR.
- Pereira, Valdrighi, Raimundo (2025). M²FGB: A min-max gradient boosting framework for subgroup fairness. FAccT.
- Galhotra, Shanmugam, Sattigeri, Varshney (2022). Causal feature selection for algorithmic fairness. SIGMOD.

## Contributing

Directions we would welcome: nonlinear DAG learning (NOTEARS-MLP / DAGMA-MLP) for indirect proxies, multiple and intersectional protected attributes, and other tree learners (LightGBM, CatBoost — Stage 2 is classifier-agnostic).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Open a GitHub issue, or email dnbhatia@ebay.com / mehala.n@vidyashilp.edu.in.
