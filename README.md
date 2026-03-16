# Factored Representations in Transformers Trained on Non-Ergodic HMM Mixtures

This project investigates whether small transformers trained on non-ergodic mixtures of Mess3 Hidden Markov Models learn **factored representations** — organizing latent factors (component identity and within-component belief state) into orthogonal subspaces with linear dimensionality scaling.

**[Full writeup (PDF)](writeup.pdf)** | **[Executive summary (PDF)](executive_summary.pdf)**

## Background

Natural language is non-ergodic: different documents follow different latent dynamics (topics, styles, domains). A language model must identify the latent "component" from context, then predict according to that component's dynamics. This experiment provides a controlled synthetic setting where we can derive the optimal geometry and test whether transformers find it.

The **Mess3 process** is a 3-state HMM with deterministic emission, parameterized by transition probability `p`:

```
T(p) = [[1-p,  p/2,  p/2],
         [p/2,  1-p,  p/2],
         [p/2,  p/2,  1-p]]
```

Our **non-ergodic dataset** is a mixture of `K` Mess3 processes with different `p` values. Each training sequence comes entirely from one component, but the model receives no explicit signal about which one.

## Repository Structure

```
├── non_ergodic_mess3_experiment.ipynb   # Main experiment notebook (Colab)
├── ablation_experiments.ipynb           # Capacity sweep ablation (Colab)
├── results_v2/                          # All generated figure PDFs
├── writeup.pdf                          # Full paper
├── executive_summary.pdf                # Executive summary
└── requirements.txt
```

## Running Experiments

### Main Experiment (Colab)

Open `non_ergodic_mess3_experiment.ipynb` in Google Colab and run all cells. This trains 9 models (K=2,4,8 x 3 seeds), runs baseline analyses, and validation experiments V1-V5.

### Ablation Experiments (Colab)

Open `ablation_experiments.ipynb` for the `d_model` capacity sweep (32, 64, 128, 256) testing whether dimensionality is capacity-bounded.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Type | GPT-2 decoder-only (TransformerLens) |
| Layers | 2 |
| d_model | 64 |
| d_mlp | 256 |
| Attention heads | 4 (d_head=16) |
| Context length | 16 |
| Vocab size | 4 (BOS + 3 tokens) |

## Results

- Belief states are linearly recoverable from residual stream activations (R² > 0.99 for K=2, K=8)
- Linear probe for component ID tracks the Bayesian-optimal classifier at all context positions
- Effective dimensionality is approximately flat at ~6 regardless of K, ruling out the joint representation (3K-1) but not matching the factored prediction (K+1)
- Factored structure (orthogonal subspaces) holds for K=2 (overlap 0.01) but breaks down for K=8 (overlap 0.93)
- Flat dimensionality persists across model capacities (d_model 32 to 256)
- Compression phase transition occurs in the first ~2000 training steps

See the [full writeup](writeup.pdf) for detailed analysis and discussion.

## References

1. Shai, A. et al. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)

2. Shai, A. et al. (2026). Transformers Learn Factored Representations. [arXiv:2602.02385](https://arxiv.org/abs/2602.02385)

3. Xie, S. M. et al. (2022). An Explanation of In-Context Learning as Implicit Bayesian Inference. *ICLR 2022*. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080)

4. Piotrowski, T. & Riechers, P. (2025). Constrained Belief Updates Explain Geometric Structures in Transformer Representations. [arXiv:2502.01954](https://arxiv.org/abs/2502.01954)
