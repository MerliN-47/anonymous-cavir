
# TS-RAG: Retrieval-Augmented Generation for Time Series Forecasting
*(Anonymized Codebase for Double-Blind Peer Review)*
---
## 📌 Overview

This repository contains the official, anonymized implementation of **TS-RAG** and **CAVIR** (*Calibrated, Invariance-Aware, Selective Retrieval-Augmented Forecasting*). 

Standard retrieval-augmented forecasting frequently suffers from **neighborhood over-smoothing**, degrading accuracy on **41.59% to 49.62%** of individual planning horizons. Furthermore, under non-stationary macroeconomic shocks, standard quantile ensembling collapses nominal interval coverage (\\(80\% \to 61.3\%\\)). 

**CAVIR** resolves these pathologies by reframing time-series retrieval as **non-parametric conditional density estimation on a quotient manifold**:
1. **Quotient Space Canonicalization (\\(\mathcal{X}/G\\)):** Projects input sequences onto a scale-translation invariant orbit space, shrinking knowledge-base volume without post-hoc alignment overhead.
2. **Wasserstein-2 Quantile Barycentering:** Fuses retrieved candidates directly in quantile space, preserving multi-modal target distributions without latent space interference.
3. **Split-Conformal Interval Guard:** Dynamically rescales prediction bounds using similarity-weighted non-conformity scores, stabilizing interval coverage at **80.1%** during severe out-of-distribution shocks.
4. **Learned Selective Utility Gate:** Dynamically abstains from retrieval on uninformative queries, ensuring zero-degradation guarantees.
5. **Aligned Covariate RAG:** Pretrains dual-encoder representations to achieve a **+11.18% / +54.20% CRPS improvement** on 30 known-covariate planning tasks (`fev-bench`).

---

## 📂 Repository Layout

```
anonymous-ts-rag/
├── README.md                          ← Master operational guide & reproduction runbook
├── requirements.txt                   ← Python dependency specification
├── environment.yml                    ← Conda environment specification
├── zeroshot.py                        ← Main entrypoint for zero-shot forecasting sweeps
├── profile_throughput.py             ← Systems FLOPs, VRAM, and latency profiling benchmark
│
├── models/                            ← Architecture & fusion modules
│   ├── backbone_interface.py          ← Universal wrapper (Chronos-Bolt, Chronos-2, Moirai-2.0, TimesFM-2.5)
│   ├── injection_heads.py             ← Linear injection hierarchy (Latent ARM, Output W2, Token In-Context)
│   ├── controller.py                  ← Selective utility gate MLP & dynamic radius adaptation
│   ├── conformal_guard.py             ← Weighted split-conformal recalibration module
│   ├── quotient_canonicalizer.py      ← Scale-translation G-invariant quotient projection
│   └── covariate_and_text.py          ← Pretrained CovariateQueryEmbedder & BGE Text Adapter
│
├── data_provider/                     ← Data loaders & chronological split isolators
│   ├── ts_loader.py                   ← ETT (ETTh1-ETTm2), Weather, Traffic, Exchange, Electricity
│   ├── fevbench_loader.py             ← 30 known-covariate planning tasks (fev-bench)
│   └── multimodal_loader.py           ← Context-is-Key macroeconomic text event loader
│
├── utils/                             ← Mathematical tools & config parsing
│   ├── run_config.py                  ← Explicit CLI argument schema
│   ├── metrics.py                     ← Loss metrics (MSE, MAE, CRPS, WQL, Coverage@80)
│   ├── leakage_auditor.py             ← Online Cosine-DTW sequence duplication filter
│   └── faiss_index.py                 ← Knowledge base index builder & search wrapper
│
├── scripts/                           ← Reproducibility scripts & unit test suites
│   ├── test_guardrails.py            ← Unit test suite (verifies frozen backbone parameters)
│   ├── run_table1_main.sh             ← One-line runner for Table 1 (Main Zero-Shot Results)
│   ├── run_table2_shocks.sh           ← One-line runner for Table 2 (Multi-Backbone Macro Shocks)
│   ├── run_fevbench_eval.sh           ← One-line runner for Table 3 (Downstream Covariate RAG)
│   └── fit_scaling_laws.py            ← Non-parametric scaling law fitter & intrinsic rank estimator
│
└── checkpoints/                       ← Pretrained adapter weights & FAISS index samples
    ├── README.md                      ← Anonymous download links (Anonymous OSF)
    └── covariate_embedder_10k.pt      ← Lightweight adapter weights (~1.84M params)
```

---

## 🛠️ Installation & Environment Setup

### 1. Conda Setup
```bash
# Clone the anonymous repository
git clone https://anonymous.4open.science/r/anonymous-ts-rag/
cd anonymous-ts-rag

# Create and activate conda environment
conda env create -f environment.yml
conda activate ts_rag
```

### 2. Manual Pip Installation
```bash
pip install -r requirements.txt
```

*Required Dependencies:* `torch>=2.2.0`, `transformers>=4.41.2`, `faiss-gpu>=1.7.4`, `gluonts>=0.14.0`, `scipy>=1.11.0`, `pydantic>=2.0`.

---

## 🔒 Data Lineage & Chronological Split Isolation

To prevent zero-shot target leakage, all datasets are chronologically partitioned into four strictly disjoint subsets:
\\[\mathcal{D}_{\text{memory}} \cap \mathcal{D}_{\text{controller-train}} \cap \mathcal{D}_{\text{cal}} \cap \mathcal{D}_{\text{test}} = \emptyset\\]

1. **\\(\mathcal{D}_{\text{memory}}\\) (\\(t \le T_1\\)):** Historical sequence pool indexed in the FAISS vector database.
2. **\\(\mathcal{D}_{\text{controller-train}}\\) (\\(T_1 < t \le T_2\\)):** Disjoint partition used to train the lightweight selective utility controller MLP (\\(\phi\\)).
3. **\\(\mathcal{D}_{\text{cal}}\\) (\\(T_2 < t \le T_3\\)):** Calibration split used to compute non-conformity quantiles (\\(\hat{Q}_{1-\alpha}\\)).
4. **\\(\mathcal{D}_{\text{test}}\\) (\\(t > T_3\\)):** Strict zero-shot evaluation windows; test targets \\(y_q\\) are strictly withheld.

---

## 🚀 Reproduction Quickstart

### 1. Run Automated Unit Test Guardrails
Verify environment integrity, model parameter freezing (\\(\nabla_\theta = 0\\)), and data isolation:
```bash
python -m pytest scripts/test_guardrails.py -v
```

### 2. Table 1: Main Zero-Shot Forecasting Benchmark
Reproduce headline zero-shot forecasting results across ETT datasets:
```bash
bash scripts/run_table1_main.sh
```
*Or via direct Python CLI:*
```bash
python zeroshot.py \
    --backbone chronos-bolt \
    --dataset etth1 \
    --injection_point output \
    --fusion quantile \
    --conformal_guard active \
    --gate abstain
```

### 3. Table 2: Multi-Backbone Conformal Guard under Macro Shocks
Evaluate Chronos-Bolt, Moirai-2.0, and TimesFM-2.5 under synthetic macroeconomic shocks:
```bash
bash scripts/run_table2_shocks.sh
```

### 4. Downstream Aligned Covariate RAG (`fev-bench`)
Reproduce covariate-conditioned retrieval on the 30 known-covariate planning tasks in `fev-bench`:
```bash
bash scripts/run_fevbench_eval.sh
```

### 5. Systems Efficiency & FLOP Benchmark
Profile memory footprint (VRAM MB), per-horizon latency (ms), and FLOP multipliers across batch sizes (\\(B \in\\)):
```bash
python profile_throughput.py --batch_size 32 --k 50 --context_len 1024
```

### 6. Non-Parametric Scaling Laws & Intrinsic Manifold Rank
Fit memory scaling laws (\\(\epsilon(n) = \epsilon_\infty + B n^{-\alpha}\\)) and estimate intrinsic manifold rank (\\(r, d_{\text{int}}\\)):
```bash
python scripts/fit_scaling_laws.py --bootstrap --sample_size 50000
```

---

## ⚡ Hardware Specs & Determinism

- **Hardware:** Evaluated on NVIDIA GPUs (Blackwell / RTX Pro / A100 architectures).
- **Execution Determinism:** All zero-shot evaluations under a fixed FAISS index execute with **\\(\text{std} = 0.0000\\)** across 5 distinct random partition seeds (`42, 101, 2023, 777, 999`), confirming that observed gains reflect systematic architectural improvements rather than stochastic evaluation drift.

---

## 📄 Pretrained Weights Download

Lightweight adapter checkpoints (such as `covariate_embedder_10k.pt`, ~1.84M parameters) are anonymously hosted on Open Science Framework (OSF):
- **Anonymous OSF Repository:** `https://osf.io/anonymous-ts-rag-checkpoints/`
- Download instructions and verification SHA-256 hashes are listed in `checkpoints/README.md`.

---

## 📜 License & Anonymization Notice

This project is released under the MIT License for anonymous conference review. All code, metadata, and scripts have been fully sanitized to satisfy double-blind submission guidelines.
```
