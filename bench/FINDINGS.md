# MiniMax H3 on 2x RTX 5060 Ti — measured findings

_Generated 2026-08-04 14:07 from 52 benchmark runs._

## Tuned configuration

**Launch flags:** `--use-sage-attention --fast`

| phase | winner | config | server s | stage |
|---|---|---|---:|---|
| A Attention backend | A2 | `--use-sage-attention` | 74.3 | probe |
| B --fast optimizations | B4 | `--use-sage-attention --fast` | 354.8 | probe |

Artefacts: `configs/comfyui-server.ini`, `configs/launch-tuned.sh`, `workflows/h3/h3-tuned.json`, `bench/tuned.json`.


## Results — confirm stage


### A. Attention backend

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| A0 | `(defaults)` | 1250.6 | 62.53 | +0.0% | cuda:0 13.1G, cuda:1 0.2G |

## Results — probe stage


### A. Attention backend

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| A2 | `--use-sage-attention` | 74.3 | 7.43 | +0.0% | cuda:0 15.0G, cuda:1 0.2G |
| A0 | `(defaults)` | 83.3 | 8.34 | +12.2% | cuda:0 15.1G, cuda:1 0.2G |
| A1 | `--use-pytorch-cross-attention` | 89.4 | 8.94 | +20.4% | cuda:0 15.0G, cuda:1 0.2G |
| A4 | `--use-split-cross-attention` | 130.4 | 13.04 | +75.6% | cuda:0 13.1G, cuda:1 0.2G |

### B. --fast optimizations

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| B4 | `--use-sage-attention --fast` | 354.8 | 35.48 | +0.0% | cuda:0 14.3G, cuda:1 0.2G |
| B2 | `--use-sage-attention --fast fp16_accumulation cublas_ops` | 356.4 | 35.64 | +0.5% | cuda:0 13.8G, cuda:1 0.2G |
| B1 | `--use-sage-attention --fast fp16_accumulation` | 359.2 | 35.92 | +1.3% | cuda:0 13.7G, cuda:1 0.2G |
| B3 | `--use-sage-attention --fast autotune` | 373.1 | 37.31 | +5.1% | cuda:0 14.3G, cuda:1 0.2G |
| B0 | `--use-sage-attention` | 373.4 | 37.34 | +5.2% | cuda:0 13.6G, cuda:1 0.2G |

### K. Step caching, compile & sage3  _(not auto-ranked: rows differ in work done or output, so fastest != best)_

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| K6 | `--use-sage-attention` | 227.6 | 22.76 | +0.0% | cuda:0 11.5G, cuda:1 0.2G |
| K7 | `--use-sage-attention` | 252.2 | 25.22 | +10.8% | cuda:0 14.9G, cuda:1 0.2G |
| K5 | `--use-sage-attention` | 253.0 | 25.30 | +11.2% | cuda:0 13.7G, cuda:1 0.2G |
| K8 | `--use-sage-attention` | 253.1 | 25.31 | +11.2% | cuda:0 14.9G, cuda:1 0.2G |
| K2 | `--use-sage-attention` | 253.5 | 25.35 | +11.4% | cuda:0 15.0G, cuda:1 0.2G |
| K4 | `--use-sage-attention` | 255.0 | 25.50 | +12.0% | cuda:0 14.9G, cuda:1 0.2G |
| K3 | `--use-sage-attention` | 283.2 | 28.32 | +24.4% | cuda:0 15.0G, cuda:1 0.2G |
| K1 | `--use-sage-attention` | 284.7 | 28.47 | +25.1% | cuda:0 11.3G, cuda:1 0.2G |
| K10 | `--use-sage-attention` | 336.4 | 33.64 | +47.8% | cuda:0 13.7G, cuda:1 0.2G |
| K0 | `--use-sage-attention` | 366.7 | 36.67 | +61.1% | cuda:0 14.0G, cuda:1 0.2G |
| K11 | `--use-sage-attention` | — | — | **failed** | — |
| K9 | `--use-sage-attention` | — | — | **failed** | — |

### L. Resolution scaling  _(not auto-ranked: rows differ in work done or output, so fastest != best)_

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| L4 | `--use-sage-attention` | 87.6 | 8.76 | +0.0% | cuda:0 15.1G, cuda:1 0.2G |
| L3 | `--use-sage-attention` | 106.9 | 10.69 | +22.1% | cuda:0 15.0G, cuda:1 0.2G |
| L2 | `--use-sage-attention` | 141.7 | 14.16 | +61.7% | cuda:0 15.2G, cuda:1 0.2G |
| L1 | `--use-sage-attention` | 246.1 | 24.61 | +181.0% | cuda:0 13.7G, cuda:1 0.2G |

### M. DisTorch2 sharding

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| M1 | `--use-sage-attention` | 348.0 | 34.80 | +0.0% | cuda:0 11.5G, cuda:1 0.2G |
| M2 | `--use-sage-attention` | 359.9 | 35.99 | +3.4% | cuda:0 13.0G, cuda:1 0.2G |
| M0 | `--use-sage-attention` | 376.0 | 37.60 | +8.1% | cuda:0 13.7G, cuda:1 0.2G |

### N. N

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| N0 | `--use-sage-attention` | 112.2 | 112.23 | +0.0% | cuda:0 13.8G, cuda:1 0.2G |
| N2 | `--use-sage-attention` | 144.8 | 48.27 | +29.0% | cuda:0 11.3G, cuda:1 0.2G |
| N1 | `--use-sage-attention` | — | — | **failed** | — |

### P. Pipeline split (shared server)

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| P1 | `--use-sage-attention --disable-cuda-malloc --disable-async-offload` | — | — | **failed** | — |
| P0 | `--use-sage-attention --disable-cuda-malloc --disable-async-offload` | — | — | **failed** | — |

### Q. Pipeline split (isolated server)

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| Q1 | `--use-sage-attention` | — | — | **failed** | — |
