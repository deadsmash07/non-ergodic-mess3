# Factored Representations in Transformers Trained on Non-Ergodic HMM Mixtures

This project investigates whether small transformers trained on non-ergodic mixtures of Mess3 Hidden Markov Models learn **factored representations** — organizing latent factors (component identity and within-component belief state) into orthogonal subspaces with linear dimensionality scaling.

## Background

Natural language is non-ergodic: different documents follow different latent dynamics (topics, styles, domains). A language model must identify the latent "component" from context, then predict according to that component's dynamics. This experiment provides a controlled synthetic setting where we can derive the optimal geometry and test whether transformers find it.

The **Mess3 process** is a 3-state HMM with deterministic emission, parameterized by transition probability `p`:

```
T(p) = [[1-p,  p/2,  p/2],
         [p/2,  1-p,  p/2],
         [p/2,  p/2,  1-p]]
```

Our **non-ergodic dataset** is a mixture of `K` Mess3 processes with different `p` values. Each training sequence comes entirely from one component, but the model receives no explicit signal about which one.

## Key Predictions

| Prediction | Description |
|-----------|-------------|
| **P1** | Effective dimensionality = K+1 (K-1 for component identity + 2 for belief simplex) |
| **P2** | Component-identity and belief-state subspaces are approximately orthogonal |
| **P3** | Component ID accuracy increases sigmoidally with context position |
| **P4** | Layer-wise emergence: embedding → component ID → belief tracking |
| **P5** | Geometry is factored (orthogonal subspaces), not joint (tensor product) |
| **P6** | Dimensionality scales linearly as K+1, not as 3K-1 |

## Repository Structure

```
├── src/
│   ├── data.py          # Mess3 process, non-ergodic dataset, belief computation
│   └── model.py         # TransformerLens model creation, training, activation extraction
├── tests/
│   ├── test_data.py     # 18 tests for data generation and belief states
│   └── test_model.py    # 6 tests for model creation, training, activations
├── configs.py           # Experiment configurations (K=2,4,8 sweeps)
├── docs/
│   ├── experiment_log.md                             # Running findings log
│   └── plans/
│       ├── 2026-03-06-non-ergodic-mess3-design.md    # Experiment design document
│       └── 2026-03-06-non-ergodic-mess3-implementation.md
├── non_ergodic_mess3_experiment.ipynb   # Main experiment notebook (Colab)
├── ablation_experiments.ipynb          # Ablation experiments (capacity, seq_len sweeps)
├── results/                            # Generated figure PDFs
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For Google Colab:
```bash
pip install -U transformer_lens einops transformers==4.37.2
```

## Running Experiments

### Tests
```bash
python -m pytest tests/ -v
```

### Main Experiment (Colab)

Open `non_ergodic_mess3_experiment.ipynb` in Google Colab and run all cells. This trains 9 models (K=2,4,8 x 3 seeds) and generates analysis figures.

### Ablation Experiments (Colab)

Open `ablation_experiments.ipynb` for:
- **A1**: `d_model` capacity sweep (32, 64, 128, 256) to test if dimensionality is capacity-bounded
- **A2**: Sequence length sweep (16, 32, 64) to test if longer context improves component identification

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

## Preliminary Results

See [`docs/experiment_log.md`](docs/experiment_log.md) for detailed findings.

**Key results from Experiment 1 (K=2,4,8 baseline):**

- Belief states are **linearly recoverable** from residual stream activations (R² > 0.99 for K=2, K=8)
- Linear probe for component ID **tracks the Bayesian optimal** classifier at all context positions
- Effective dimensionality is **approximately flat at ~6-7** regardless of K, strongly rejecting the joint (unfactored) hypothesis (3K-1) but not precisely matching the factored prediction (K+1)
- A **compression phase transition** occurs in the first ~2000 training steps, where dimensionality drops from ~20 to its final value

## References

1. Shai, A., Akyurek, E., & Tegmark, M. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)

2. Shai, A., Beren, S., & Tegmark, M. (2026). Transformers Learn Factored Representations. [arXiv:2602.02385](https://arxiv.org/abs/2602.02385)

3. Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. (2022). An Explanation of In-Context Learning as Implicit Bayesian Inference. *ICLR 2022*. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080)

4. Piotrowski, T. & Riechers, P. (2025). Constrained Belief Updates Explain Geometric Structures in Transformer Representations. *ICML 2025*. [arXiv:2502.01954](https://arxiv.org/abs/2502.01954)

5. Boyd, A., et al. (2025). From Monoliths to Modules: Decomposing Transducers for Efficient World Modelling. [arXiv:2512.02193](https://arxiv.org/abs/2512.02193)

6. Debowski, L. (2018). Is Natural Language a Perigraphic Process? *Entropy*, 20(2):85. [arXiv:1706.04432](https://arxiv.org/abs/1706.04432)

7. Shalizi, C. R. & Crutchfield, J. P. (2001). Computational Mechanics: Pattern and Prediction, Structure and Simplicity. [arXiv:cond-mat/9907176](https://arxiv.org/abs/cond-mat/9907176)
