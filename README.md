# green-ai-trajectory-benchmark
Master's thesis benchmarking code for energy-aware trajectory forecasting.
# Energy-Aware Benchmarking of Trajectory Forecasting Architectures

Code accompanying the MSc thesis *"Αξιολόγηση της κατανάλωσης ενέργειας αλγορίθμων τεχνητής νοημοσύνης σε δεδομένα κίνησης"*
(University of Piraeus, MSc in Applied Statistics).

The project benchmarks four time-series forecasting architectures on aviation (ADS-B) trajectory data, measuring both
predictive accuracy (MSE / MAE) and energy consumption / carbon footprint (via
[CodeCarbon](https://github.com/mlco2/codecarbon)) under an identical training protocol, and identifies the empirical
accuracy–energy Pareto front between them.

| Tier | Model | Idea | Complexity in sequence length L |
|---|---|---|---|
| Lightweight | **DLinear** | Trend/seasonal decomposition + linear projections (Zeng et al., 2023) | O(L) |
| Lightweight | **SOFTS** | Series-core fusion, no attention (Han et al., 2024) | O(L·d) |
| Medium | **LSTM** | 2-layer recurrent baseline (Hochreiter & Schmidhuber, 1997) | O(L) sequential |
| Heavy | **Informer** | ProbSparse self-attention (Zhou et al., 2021) | O(L log L) |

## Experimental protocol

All four models are trained and evaluated under the same conditions:

- **Dataset:** London Landing Sync ADS-B flight telemetry (423,701 observations), 6 features per step:
  latitude, longitude, altitude, ground speed, heading, vertical rate. Full dataset description: thesis Section 4.1.
- **Task:** one-step-ahead prediction of position (latitude, longitude) from a sliding window of L = 10 steps.
- **Split:** strict **chronological 60% / 20% / 20%** train / validation / test (no shuffling, no future leakage) – thesis Section 3.3.3.
- **Training:** Adam, learning rate 1e-3, batch size 64, 5 epochs, fixed random seed.
- **Metrics:** MSE, MAE; energy (kWh) split into GPU / RAM / CPU; CO₂-equivalent (g); peak VRAM; training time;
  Energy Efficiency Index EEI = 1 / (MSE × kWh).
- **Hardware:** NVIDIA Tesla T4 (16 GB), 2 vCPU Intel Xeon @ 2.00 GHz, 12.67 GB RAM, energy measured with CodeCarbon 3.3.0
  (GPU via NVML, CPU via RAPL when available otherwise load-based TDP estimate, RAM via a static W/GB model).

## Main findings

Exact numbers are in the result CSVs of this repository and in Table 5.1 of the thesis. In short:

- **Informer** is Pareto-dominated: it consumes more energy than the other models without an accuracy benefit.
- **DLinear** is the cheapest model in energy and training time, with competitive error.
- **LSTM** achieves the best accuracy (lowest MSE and MAE) at a moderate energy cost and the largest VRAM footprint.
- **SOFTS** sits between DLinear and LSTM and does not lie on the Pareto front once DLinear is included.

So the genuine accuracy–energy trade-off is between **DLinear** (energy) and **LSTM** (accuracy). See thesis Chapter 5 for the analysis.

## Repository structure

```
.
├── script_four_architectures.py     # main benchmark script: trains and measures the models  
├── figure_1_script.py                  # regenerates the thesis figures 
├── figure_2_script.py                  # regenerates the thesis figures 
├── figure_3_script.py                  # regenerates the thesis figures 
├── requirements.txt
└── data/                            # place the London Landing Sync CSV here (not distributed)
```

## Setup

```bash
git clone https://github.com/nikolioudakism14/green-ai-trajectory-benchmark.git
cd green-ai-trajectory-benchmark
pip install -r requirements.txt
```

A CUDA GPU is recommended. On CPU-only machines the code still runs, but the energy figures will not be comparable to the thesis.

## Usage

1. Put the London Landing Sync CSV in `data/`.
   The raw file has no header; columns are `flight_id, seq_id, timestamp, lon, lat, alt`.
2. Train and measure all models:
   ```bash
   python script_four_architectures.py
   ```
   Nothing is hard-coded: every row comes from a fresh training run.
3. Regenerate the figures:
   ```bash
   figure_1_script.py
   figure_2_script.py
   figure_3_script.py
   ```

**Energy and time measurements depend on the hardware and on background load, so re-running will give values close to,
but not identical to, those reported in the thesis.**

## Scope note

These are deliberately simplified, single-block implementations of each architecture's core mechanism
(DLinear's decomposition + linear layers, SOFTS' series-core fusion, LSTM's recurrent gating, Informer's ProbSparse attention),
built for a controlled like-for-like energy comparison, not as reproductions of the full original models.
Hyperparameters follow the source literature at reduced scale and were not tuned; each model was trained once
(no multi-seed confidence intervals). See thesis Sections 3.7 and 4.2.2.

## Citation

If you use this code, please cite the thesis: Nikolioudakis, M. (2026). *Αξιολόγηση της κατανάλωσης ενέργειας αλγορίθμων τεχνητής νοημοσύνης σε δεδομένα κίνησης*.
MSc thesis, University of Piraeus.
