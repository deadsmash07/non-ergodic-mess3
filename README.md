# Residual stream geometry of transformers trained on non-ergodic HMM mixtures

**[Full writeup (PDF)](writeup.pdf)** | **[Executive summary (PDF)](executive_summary.pdf)**

## Introduction

We trained small transformers on non-ergodic mixtures of Mess3 hidden Markov models (HMMs) and analyzed the geometry of the residual stream. Each training sequence was generated entirely by one of K ergodic components with different transition parameters; the model received no signal indicating which component produced a given sequence. This setup requires the model to simultaneously infer the generating component and track the within-component belief state to predict future tokens. We swept K in {2, 4, 8} and derived pre-registered geometric predictions based on the factored world hypothesis (Shai et al., 2026), while noting that our latent factors (component identity and within-component belief) differ structurally from the parallel-process factors treated by Theorem 2.3 of that work: component identity is a sequence-level latent variable that is fixed for the entire sequence, not a token-level factor that updates at each position.

## Methods

We used GPT-2-style transformers (2 layers, d_model=64, d_MLP=256, 4 heads) implemented in TransformerLens, trained with Adam (lr 1e-3, cosine decay, batch size 256, 10,000 steps, 3 seeds per K). The Mess3 process is a 3-state HMM with deterministic emission parameterized by transition probability p in (0,1); components used p in {0.1, 0.9} for K=2, p in {0.1, 0.3, 0.7, 0.9} for K=4, and p in {0.1, ..., 0.9}\{0.5} for K=8. We analyzed final-layer activations on 10,000 held-out test sequences via: Ridge regression for belief state recovery, logistic regression probes for component identification, principal component analysis (PCA) for dimensionality measurement, and subspace overlap metrics for orthogonality assessment. For the capacity ablation, we trained additional models at d_model in {32, 128, 256}.

## Findings

**Belief states are linearly encoded in the residual stream.** Ridge regression from final-layer activations to ground-truth predictive belief vectors yields R²=0.995 and 0.996 for the two components at K=2, and R² between 0.995 and 0.997 for the first four components at K=8. PCA projections of within-component activations reproduce the triangular 2-simplex structure of the Mess3 belief geometry. Lower R² values at K=4 for p=0.7 (R²=0.921) and p=0.9 (R²=0.928) reflect the compact belief simplex when the transition matrix approaches uniformity, reducing the signal available for linear recovery. These results extend the single-process finding of Shai et al. (2024) to a multi-component non-ergodic setting.

**The model performs implicit Bayesian inference for component identification.** A logistic regression probe trained on residual stream activations at each context position t in {1, ..., 16} matches the Bayesian-optimal posterior P(C=k | x_{1:t}) within approximately 2% at every position, for all values of K. For K=2, accuracy reaches 1.0 by position 8. For K=8, accuracy plateaus at 0.47 at position 16, matching the Bayesian bound, which is limited by the statistical similarity of adjacent components (e.g., p=0.3 vs. p=0.4) given only 16 tokens of evidence. This is consistent with the framework of Xie et al. (2022), whose Theorem 1 applies to exactly this mixture-of-HMMs structure.

**The representation transitions from factored to entangled as K increases.** We measured subspace overlap between the component-identity subspace (identified via between-component PCA) and the within-component belief subspace (identified via pooled within-component PCA), using late-position activations (positions 8-16). At K=2, the overlap is 0.011 with a minimum principal angle of 83.9 degrees, indicating nearly orthogonal subspaces. At K=4, the overlap increases to 0.065 (angle 69.7 degrees). At K=8, the overlap reaches 0.934 (angle 8.2 degrees): the two subspaces are nearly identical.

This transition is corroborated by within-component dimensionality. For K=2, PCA within each component yields exactly 2 effective dimensions (at 95% cumulative explained variance), matching the 2-simplex prediction for the Mess3 belief geometry. For K=4 and K=8, within-component dimensionality ranges from 3 to 7, indicating that component-identity information is mixed into the within-component representation rather than confined to a separate subspace.

The total effective dimensionality is approximately 5-6 at fixed context positions, regardless of K. This rules out the joint (unfactored) prediction (5, 11, 23 for K=2, 4, 8) but does not match the factored prediction (K+1 = 3, 5, 9). The flat dimensionality is confirmed by threshold-free measures: participation ratio is 3.3, 4.6, 4.7 and stable rank is 2.3, 2.8, 3.1 for K=2, 4, 8 respectively.

**The flat dimensionality is capacity-independent.** We trained K=4 and K=8 at d_model in {32, 64, 128, 256} (26K to 1.6M parameters). Effective dimensionality remains at 6-7 across this 60x range of model capacity. All models achieve identical final loss (0.890 for K=4, 0.944 for K=8), indicating that the task does not benefit from additional capacity and that the flat dimensionality is a property of the learned representation.

**Training dynamics show rapid compression.** All configurations show effective dimensionality dropping from approximately 21 at initialization to 6-9 within the first 2,000 of 10,000 gradient steps, followed by a stable plateau. This compression phase coincides with rapid loss decrease.

## Prediction scorecard

| Prediction | Result | Evidence |
|-----------|--------|----------|
| P1: d_eff ~ K+1 | Not confirmed | Flat at 5-6 across K |
| P2: Orthogonal subspaces | K=2: yes; K=8: no | Overlap: 0.01 vs. 0.93 |
| P3: Sigmoidal component ID | Confirmed | Probe matches Bayesian optimal |
| P4: Belief recovery R² > 0.9 | Confirmed | R² > 0.99 for K=2, 8 |

## Conclusions

Our non-ergodic Mess3 mixture produces a setting where two latent factors (component identity and within-component belief) operate on different timescales and have an asymmetric dependence. When the components are well-separated (K=2), the model resolves component identity after a few tokens and organizes the two factors into orthogonal subspaces with the predicted within-component dimensionality. When many similar components are present (K=8) and component identity remains uncertain throughout the sequence, the model uses a shared, entangled low-dimensional representation. This entangled structure uses approximately 6 effective dimensions, independent of both K and model capacity.

The structural difference between our sequence-level latent variable and the token-level parallel factors in Shai et al. (2026) means that the conditions of Theorem 2.3 do not apply directly to our setting. Our results are consistent with this: the factored representation emerges where component identification is easy and the two factors can be treated independently, and breaks down where it is hard.

## Running the experiments

Install dependencies:
```bash
pip install -r requirements.txt
```

For Google Colab:
```bash
pip install -U transformer_lens einops transformers==4.37.2
```

**Main experiment:** Open `non_ergodic_mess3_experiment.ipynb` in Google Colab and run all cells. Trains 9 models (K=2,4,8 x 3 seeds) and runs all analyses including validation V1-V5.

**Ablation experiment:** Open `ablation_experiments.ipynb` for the d_model capacity sweep (32, 64, 128, 256).

## References

1. Shai, A. et al. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)
2. Shai, A. et al. (2026). Transformers Learn Factored Representations. [arXiv:2602.02385](https://arxiv.org/abs/2602.02385)
3. Xie, S. M. et al. (2022). An Explanation of In-Context Learning as Implicit Bayesian Inference. *ICLR 2022*. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080)
4. Piotrowski, T. & Riechers, P. (2025). Constrained Belief Updates Explain Geometric Structures in Transformer Representations. [arXiv:2502.01954](https://arxiv.org/abs/2502.01954)
