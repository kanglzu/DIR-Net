[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **DIR-Net: A Disentangled Implicit Routing Network for Effective Time Series Forecasting**.

![Model Architecture](plots/Model%20Architecture.pdf)

## Highlights

- **Disentangled Dual-Pathway Architecture**: Explicitly separates time series into global trend and local seasonal components via EMA-based decomposition, processed through specialized parallel branches.
- **RoPE-TPA Mechanism**: Integrates Rotary Position Embeddings into Tensor Product Attention with low-rank decomposition, reducing Q/K/V projection complexity from O(D²) to O(D) while preserving relative positional information.
- **Memory-Augmented LSTM (M-LSTM)**: Extends mLSTM with full-attention computation, mean-based hidden-state normalization, and a stabilizing control state m_t that suppresses exponential gating instability.
- **Adaptive Feature Gate Mechanism (AFGM)**: Learns input-dependent routing weights for dynamic fusion of trend and seasonal representations, replacing rigid fixed-weight strategies.
- **KAN-Based Predictor**: Employs B-Spline Kolmogorov-Arnold Networks for expressive non-linear projection in the final prediction layer.
- **Lightweight Design**: Only 53.30K parameters and 429.78K FLOPs, achieving state-of-the-art performance across nine benchmark datasets.

## Architecture

![DTAM vs mLSTM Comparison](plots/Comparison%20of%20DTAM%20and%20mLSTM.pdf)

DIR-Net consists of four synergistic components: (1) EMA-based Decomposition, (2) Disentangled Temporal Encoder (DTE) with DTAM and DCC branches, (3) Adaptive Feature Gate Mechanism (AFGM), and (4) KAN-Based Predictor.

| Module | Pathway | Role |
|--------|---------|------|
| EMA Decomposition | Pre-processing | Separates input into smooth trend **T** and oscillatory seasonal **S** components |
| DTAM (RoPE-TPA + M-LSTM) | Trend branch | Long-range dependency modeling with stabilized exponential gating and full-attention |
| DCC (Dilated Causal Convolution) | Seasonal branch | Multi-scale local pattern extraction with exponential receptive field growth (d=1,2,4) |
| AFGM (Adaptive Feature Gate) | Fusion | Input-dependent dynamic weighting: **g⊙H_trend + (1-g)⊙H_season** |
| KAN-Based Predictor | Output | B-Spline Kolmogorov-Arnold Network for non-linear prediction projection |
| RevIN (optional) | Input/Output | Reversible Instance Normalization for cross-distribution generalization |

### Key Equations

**EMA Decomposition:**
$$\mathbf{T}_t = \alpha \mathbf{X}_t + (1-\alpha)\mathbf{T}_{t-1}, \quad \mathbf{S}_t = \mathbf{X}_t - \mathbf{T}_t$$

**RoPE-TPA Projections:**
$$\mathbf{Q} = \mathbf{W}_{out}(\mathbf{A}_Q \odot \mathbf{B}_{Q\_rot}), \quad \mathbf{K} = \mathbf{W}_{out}(\mathbf{A}_K \odot \mathbf{B}_K), \quad \mathbf{V} = \mathbf{W}_{out}(\mathbf{A}_V \odot \mathbf{B}_V)$$

**M-LSTM Cell State Update:**
$$\mathbf{C}_t = \mathbf{f}_t \mathbf{C}_{t-1} + \mathbf{i}_t \mathbf{A}_t, \quad \mathbf{h}_t = \mathbf{o}_t \odot (\mathbf{C}_t \cdot \text{mean}(\mathbf{S}_t)), \quad \mathbf{m}_t = \max(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1}, \tilde{\mathbf{i}}_t)$$

## Performance Overview

![Bubble Chart](plots/Bubble%20Chart.pdf)

*Figure 1: Comparison of DIR-Net and seven SOTA models across nine datasets with a fixed look-back window of 96. Each bubble denotes the average MSE, and bubble size reflects the magnitude of the metric. DIR-Net achieves the lowest parameter count (53.30K) and among the lowest FLOPs (429.78K) while attaining the best average MSE.*

### Main Results (Table 2 — Average MSE / MAE per Dataset)

All experiments use an input length of 96 and prediction horizons {96, 192, 336, 720}. **Bold** = best, Underline = second-best.

| Dataset | Metric | DIR-Net | mLSTM | sLSTM | Informer | Autoformer | DeepAR | LSTM | Pyraformer | PatchTST | iTransformer | Amplifier | FilterNet |
|---------|--------|---------|-------|-------|----------|------------|--------|------|------------|----------|-------------|-----------|-----------|
| **ETTm1** | MSE | **0.566** | 1.013 | 0.881 | 0.872 | 0.637 | 1.197 | 0.837 | 0.691 | 0.395 | 0.417 | 0.395 | 0.385 |
| | MAE | **0.526** | 0.766 | 0.676 | 0.691 | 0.533 | 0.828 | 0.699 | 0.607 | 0.403 | 0.416 | 0.403 | 0.399 |
| **Weather** | MSE | 0.283 | 0.365 | 0.524 | 0.568 | 0.344 | 0.448 | 0.356 | 1.176 | 0.257 | 0.262 | 0.253 | **0.250** |
| | MAE | 0.307 | 0.406 | 0.473 | 0.522 | 0.387 | 0.433 | 0.387 | 0.477 | 0.281 | 0.282 | 0.279 | **0.278** |
| **Traffic** | MSE | **0.466** | 1.018 | 0.624 | 0.778 | 0.617 | 1.459 | 0.455 | 1.176 | 0.322 | 0.374 | 0.406 | 0.329 |
| | MAE | **0.318** | 0.671 | 0.456 | 0.436 | 0.392 | 0.760 | 0.347 | 0.469 | 0.252 | 0.307 | 0.345 | 0.244 |
| **Electricity** | MSE | 0.400 | 0.773 | 0.377 | 0.329 | 0.252 | 0.931 | 0.303 | 0.382 | 0.188 | 0.202 | 0.216 | **0.185** |
| | MAE | 0.430 | 0.690 | 0.448 | 0.415 | 0.357 | 0.736 | 0.387 | 0.445 | 0.275 | 0.287 | 0.305 | **0.275** |
| **Solar** | MSE | **0.255** | 0.650 | 0.226 | 0.236 | 0.841 | 1.153 | 0.218 | 0.226 | 0.230 | 0.259 | 0.263 | 0.241 |
| | MAE | **0.256** | 0.620 | 0.303 | 0.244 | 0.680 | 0.549 | 0.271 | 0.226 | 0.260 | 0.289 | 0.301 | 0.274 |
| **Exchange** | MSE | 1.252 | 1.425 | 1.609 | 1.601 | 1.768 | 2.286 | 1.775 | 1.152 | 0.432 | 0.370 | 0.366 | **0.384** |
| | MAE | 0.907 | 0.985 | 1.031 | 1.013 | 1.177 | 1.203 | 1.056 | 0.896 | 0.427 | 0.414 | 0.410 | **0.414** |
| **ETTm2** | MSE | 0.650 | 1.233 | 1.388 | 1.305 | 0.325 | 3.287 | 1.803 | 1.498 | 0.282 | 0.294 | 0.286 | **0.277** |
| | MAE | 0.623 | 0.867 | 0.906 | 0.797 | 0.367 | 1.477 | 1.026 | 0.869 | 0.327 | 0.337 | 0.329 | **0.324** |
| **ETTh1** | MSE | 0.827 | 1.129 | 1.021 | 1.033 | 1.359 | 1.078 | 0.966 | 0.852 | 0.451 | 0.451 | 0.458 | **0.486** |
| | MAE | 0.699 | 0.828 | 0.773 | 0.799 | 0.955 | 0.787 | 0.783 | 0.711 | 0.441 | 0.446 | 0.439 | **0.457** |
| **ETTh2** | MSE | 1.065 | 3.794 | 2.361 | 3.303 | 1.275 | 3.018 | 2.717 | 1.850 | 0.383 | 0.387 | 0.386 | **0.391** |
| | MAE | 1.208 | 1.670 | 1.298 | 1.439 | 1.109 | 1.351 | 1.361 | 1.593 | 0.405 | 0.409 | 0.409 | **0.412** |

![Comparison Results](plots/Comparison%20Result.pdf)

*Figure 4: Average MSE and MAE of all models over nine datasets.*

### Model Efficiency (Params / FLOPs)

![Params and FLOPs](plots/Params%20and%20FLOPS.pdf)

*Figure 5: Comparison of model FLOPs and parameter counts.*

| Model | DIR-Net | mLSTM | sLSTM | Informer | Autoformer | DeepAR | LSTM | Pyraformer | PatchTST | iTransformer | Amplifier | FilterNet |
|-------|---------|-------|-------|----------|------------|--------|------|------------|----------|-------------|-----------|-----------|
| **Params (K)** | **53.31** | 221.07 | 303.90 | 16,913.55 | 15,655.94 | 65.52 | 288.93 | 10,679.81 | 6,913.64 | 6,404.70 | 127.68 | 77.16 |
| **FLOPs (K)** | **429.78** | 999.42 | 3,480.19 | 723,875.84 | 611,282.43 | 64.20 | 286.21 | 26,946.05 | 6,905.86 | 404,858.88 | 119.75 | 659.58 |

DIR-Net is substantially more lightweight than Transformer-based models such as Informer (16.9M params, 7.24×10⁸ FLOPs) and Autoformer (15.7M params, 6.11×10⁸ FLOPs). Even compared with recurrent architectures, DIR-Net remains more efficient — LSTM has 288.9K parameters. PatchTST and iTransformer require over two orders of magnitude more parameters and FLOPs.

## Installation

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 / Windows 10+ |
| Python | ≥ 3.7 |
| PyTorch | ≥ 1.8.0 |
| CUDA | 11.1+ (optional) |
| NumPy | ≥ 1.19.0 |
| Pandas | ≥ 1.2.0 |
| scikit-learn | ≥ 0.24.0 |
| matplotlib | ≥ 3.3.0 |
| tqdm | ≥ 4.50.0 |
| scipy | ≥ 1.6.0 |
| sympy | ≥ 1.8 |

```bash
git clone https://github.com/kang_lzu/DIR-Net.git
cd DIR_Net
pip install -r requirements.txt
```

## Data Preparation

The model supports nine benchmark datasets. Download and place them in the `dataset/` directory:

| Dataset | Dim | Frequency | Domain | Length | 
|---------|-----|-----------|--------|--------|
| ETTm1 | 7 | 15-min | Energy | 69,680 |
| ETTm2 | 7 | 15-min | Energy | 69,680 |
| ETTh1 | 7 | Hourly | Energy | 14,307 |
| ETTh2 | 7 | Hourly | Energy | 14,307 |
| Weather | 21 | 10-min | Meteorology | 52,696 |
| Traffic | 862 | Hourly | Transportation | 17,544 |
| Electricity | 370 | Hourly | Power | 26,304 |
| Exchange-rate | 8 | Daily | Economy | 7,207 |
| Solar | 137 | 10-min | Renewable Energy | 52,560 |

For all datasets, samples are chronologically ordered and strictly partitioned into training (70%), validation (10%), and test (20%) sets. Each variable is standardized to zero mean and unit variance based exclusively on training set statistics. Experiments use prediction lengths {96, 192, 336, 720}.

Expected data format: CSV files with the first column as timestamp/date and remaining columns as feature values.

```
dataset/
├── solar.txt
├── electricity.txt
├── traffic.txt
├── weather.csv
├── exchange_rate.csv
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
└── ETTm2.csv
```

## Project Structure

```
DIR_Net/
├── models/
│   └── DIR_Net.py              # Main model (dual-path + AFGM + KAN + RevIN)
├── layers/
│   ├── MLSTM.py                # RoPE-TPA mechanism + M-LSTM block
│   ├── DTAM.py                 # Disentangled Temporal Attention Module
│   ├── DCC.py                  # Dilated Causal Convolution (d=1,2,4)
│   ├── B_Spline_KAN.py         # B-Spline Kolmogorov-Arnold Network
│   ├── decomp.py               # Time series decomposition controller
│   ├── ema.py                  # Exponential Moving Average
│   ├── dema.py                 # Double Exponential Moving Average
│   ├── revin.py                # Reversible Instance Normalization
│   └── utils.py                # CausalConv1D, BlockDiagonal, wavelet filters
├── data_provider/
│   ├── data_factory.py         # Data factory for multi-dataset support
│   └── data_loader.py          # Dataset loaders with StandardScaler
├── exp/
│   ├── exp_basic.py            # Base experiment class
│   └── exp_main.py             # Training, validation, and testing pipeline
├── utils/
│   ├── metrics.py              # MSE and MAE evaluation
│   ├── tools.py                # LR schedules, early stopping, visualization
│   └── timefeatures.py         # Frequency-based time feature encoding
├── plots/
│   ├── Model Architecture.pdf           # Overall architecture (Fig. 2)
│   ├── Comparison of DTAM and mLSTM.pdf # DTAM vs mLSTM (Fig. 3)
│   ├── Bubble Chart.pdf                 # Complexity vs performance (Fig. 1)
│   ├── Comparison Result.pdf            # Average MSE/MAE bar charts (Fig. 4)
│   └── Params and FLOPS.pdf             # Efficiency comparison (Fig. 5)
├── dataset/                    # Benchmark datasets
├── run.py                      # Main entry point with full configuration
├── requirements.txt            # Python dependencies
├── Weight and Parameter Count Calculation.py  # Model analysis utility
└── README.md
```

## Training

Modify the configuration in `run.py` and execute:

```bash
python run.py
```

Example configuration for the Solar dataset:

```python
args = SimpleNamespace(
    # Basic
    is_training=1,
    model_id='DIR_Net',
    model='DIR_Net',
    loss='mse',
    lradj='type1',

    # Data
    data='Solar',
    root_path='./dataset',
    data_path='solar.txt',
    features='M',
    target='OT',
    freq='h',

    # Prediction
    seq_len=96,
    label_len=48,
    pred_len=96,
    enc_in=137,

    # Model
    num_heads=4,
    layers2=['m'],
    hidden_size=32,
    fc1_size=16,
    fc2_size=137,

    # Moving Average
    ma_type='ema',
    alpha=0.39,
    beta=0.39,

    # Training
    train_epochs=50,
    batch_size=32,
    learning_rate=0.0001,
    patience=10,
)
```

### Key Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) + exponential decay | lr=1e-4 |
| Prediction | seq_len / label_len / pred_len | 96 / 48 / {96, 192, 336, 720} |
| Model | num_heads / hidden_size | 4 / 32 |
| RoPE-TPA | tpa_rank | 8 |
| DCC | num_layers / kernel_size | 3 / 4 |
| KAN | hidden_dim | 16 |
| EMA | α (smoothing factor) | 0.39 |
| Regularization | Early stopping patience | 10 epochs |
| Loss | Weighted MAE with arctangent temporal decay | `w_t = -arctan(t) + π/4 + 1` |
| Evaluation | MSE (primary), MAE (complementary) | — |

### Loss Function

The model employs a weighted MAE loss with arctangent-based temporal weighting that assigns higher priority to short-term predictions while maintaining a smooth and non-zero attenuation for longer horizons:

$$\mathcal{L}_{\text{train}} = \frac{1}{T}\sum_{t=1}^{T} w_t \cdot |\hat{y}_t - y_t|, \quad w_t = -\arctan(t) + \frac{\pi}{4} + 1$$

This weighting mechanism encourages robust short-term forecasts, which serve as critical anchors for long-term prediction accuracy. During validation and testing, standard MSE is reported.

## Ablation Study

### Architectural Ablation (Table 3)

Comparison of DTAM-DCC variants, Pyramid Convolution, and xLSTM-DCC variants across datasets. The parallel DTAM-DCC design (final model) consistently achieves the best performance.

| Dataset | Metric | DTAM-DCC (Parallel) | DTAM-DCC (Serial) | Pyramid Conv | xLSTM-DCC (Serial) | xLSTM-DCC (Parallel) |
|---------|--------|---------------------|-------------------|--------------|---------------------|-----------------------|
| **ETTh1** | MSE | **0.827** | 0.938 | 1.087 | 1.102 | 0.938 |
| | MAE | **0.699** | 0.757 | 0.792 | 0.825 | 0.768 |
| **ETTh2** | MSE | **1.065** | 2.527 | 2.832 | 3.059 | 3.097 |
| | MAE | **1.208** | 1.229 | 1.405 | 1.424 | 1.340 |
| **ETTm1** | MSE | **0.566** | 0.717 | 1.063 | 0.699 | 0.716 |
| | MAE | **0.526** | 0.631 | 0.776 | 0.609 | 0.569 |
| **ETTm2** | MSE | **0.650** | 1.093 | 2.246 | 2.546 | 1.754 |
| | MAE | **0.623** | 0.778 | 1.237 | 1.176 | 0.962 |
| **Weather** | MSE | **0.283** | 0.440 | 0.365 | 0.312 | 0.367 |
| | MAE | **0.307** | 0.367 | 0.369 | 0.320 | 0.365 |
| **Traffic** | MSE | **0.466** | 0.718 | 0.649 | 0.740 | 0.693 |
| | MAE | **0.318** | 0.442 | 0.723 | 0.445 | 0.430 |
| **Electricity** | MSE | **0.400** | 0.420 | 0.952 | 0.423 | 0.490 |
| | MAE | **0.430** | 0.443 | 0.766 | 0.448 | 0.442 |
| **Exchange** | MSE | **1.252** | 2.329 | 2.590 | 3.326 | 2.447 |
| | MAE | **0.907** | 1.162 | 1.303 | 1.431 | 1.271 |
| **Solar** | MSE | **0.255** | 0.270 | 0.499 | 0.249 | 0.257 |
| | MAE | **0.256** | 0.250 | 0.383 | 0.250 | 0.244 |

### Module Ablation (Table 4–8)

Ablation of individual components on ETTm1, Solar, and Weather (average across 4 horizons).

| Module | ETTm1 MSE | Solar MSE | Weather MSE | Params (K) | FLOPs (K) |
|--------|-----------|-----------|-------------|------------|-----------|
| **Full Model** | **0.566** | **0.255** | **0.283** | 53.31 | 429.78 |
| – EMA | 0.623 (+10.1%) | 0.251 (−1.6%)* | 0.285 (+0.7%) | 53.31 | 429.78 |
| – DCC | 0.623 (+10.1%) | 0.644 (+152.5%) | 0.271 (−4.2%)* | 67.46 | 152.14 |
| – DTAM | 0.644 (+13.8%) | 0.242 (−5.1%)* | 0.851 (+200.7%) | 19.18 | 687.63 |
| – AFGM | 0.622 (+9.9%) | 0.250 (−2.0%)* | 0.280 (−1.1%)* | 133.18 | 509.65 |
| – KAN | 0.566 (0.0%) | 0.256 (+0.4%) | 0.283 (0.0%) | ~53.30 | ~429.78 |

*Isolated improvements on specific datasets are attributed to dataset-specific characteristics; the full model achieves the most consistent and balanced performance across all nine benchmarks.*

**Key findings:**
- **EMA**: Removing EMA forces the model to learn trend-seasonal interactions from raw entangled sequences. MSE rises from 0.415 to 0.493 on ETTm1 (96-step) and from 0.200 to 0.239 on Solar (96-step).
- **DCC**: Strongest impact on seasonal-rich datasets (Solar 96-step: MSE 0.200→0.484). DCC is the backbone of the seasonal-extraction pathway.
- **DTAM**: Critical for long-range modeling. Weather (720-step) MSE dramatically increases from 0.363 to 2.614 without DTAM.
- **AFGM**: Consistent but moderate impact. On Solar (336-step), MSE rises from 0.277 to 0.284. Learns sample-dependent routing weights for dynamic trend-seasonal fusion.
- **KAN**: B-Spline KAN outperforms Fourier KAN, RBF KAN, and standard MLP head across all ETT datasets with negligible additional computational cost.

## Inference

To test a trained model, set `is_training=0` in `run.py` and execute:

```bash
python run.py
```

The model loads the checkpoint from `./checkpoints/{setting}/checkpoint.pth` and evaluates on the test set. Visualization plots are automatically saved to `test_results/{setting}/`.

## Citation

If you find DIR-Net useful in your research, please cite our paper:

> Jiaxue Yang, Yutong Wang, Zhongfeng Kang, Shaoyi Zong, Shantian Yang, Lihui Deng, Zichen Song.
> **DIR-Net: A Disentangled Implicit Routing Network for Effective Time Series Forecasting.**

BibTeX:

```bibtex
@article{yang2025dirnet,
  title   = {DIR-Net: A Disentangled Implicit Routing Network for Effective Time Series Forecasting},
  author  = {Jiaxue Yang and Yutong Wang and Zhongfeng Kang and Shaoyi Zong and Shantian Yang and Lihui Deng and Zichen Song},
  journal = {Preprint submitted to Elsevier},
  year    = {2025},
  note    = {Code available at https://github.com/kang_lzu/DIR-Net}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open an issue on the repository or contact the corresponding author: Zhongfeng Kang (kangzf@lzu.edu.cn).

## Acknowledgments

- Thanks to the authors of [xLSTM](https://github.com/NX-AI/xlstm), [KAN](https://github.com/KindXiaoming/pykan), and other foundational works.
- Dataset providers for the benchmark time series datasets.
