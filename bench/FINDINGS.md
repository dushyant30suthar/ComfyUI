# MiniMax H3 on 2x RTX 5060 Ti — measured findings

_Generated 2026-08-04 23:18 from 74 benchmark runs._

## Tuned configuration

**Launch flags:** `--use-sage-attention --fast`

**Placement:** `{'model_device': 'gpu:0', 'clip_device': 'gpu:1', 'vae_device': 'gpu:0'}`

| phase | winner | config | server s | stage |
|---|---|---|---:|---|
| A Attention backend | A2 | `--use-sage-attention` | 74.3 | probe |
| B --fast optimizations | B4 | `--use-sage-attention --fast` | 354.8 | probe |
| H Encoder & VAE precision | H0 | `--use-sage-attention --fast` | 356.1 | probe |
| D Device placement | D3 | `model=gpu:0 clip=gpu:1 vae=gpu:0` | 351.7 | probe |

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

### D. Device placement

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| D1 | `--use-sage-attention --fast --reserve-vram 0.5` | 348.3 | 34.83 | +0.0% | cuda:0 10.1G, cuda:1 6.8G |
| D3 | `--use-sage-attention --fast` | 351.7 | 35.17 | +1.0% | cuda:0 14.4G, cuda:1 1.1G |
| D0 | `--use-sage-attention --fast` | 355.8 | 35.58 | +2.2% | cuda:0 14.3G, cuda:1 0.2G |
| D4 | `--use-sage-attention --fast` | 356.0 | 35.60 | +2.2% | cuda:0 15.1G, cuda:1 10.3G |
| D2 | `--use-sage-attention --fast` | 420.1 | 42.01 | +20.6% | cuda:0 10.3G, cuda:1 6.8G |

### H. Encoder & VAE precision

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| H5 | `--use-sage-attention --fast --fp16-intermediates` | 353.4 | 35.34 | +0.0% | cuda:0 14.3G, cuda:1 0.2G |
| H2 | `--use-sage-attention --fast --lowvram` | 355.3 | 35.53 | +0.5% | cuda:0 14.3G, cuda:1 0.2G |
| H0 | `--use-sage-attention --fast` | 356.1 | 35.60 | +0.8% | cuda:0 14.3G, cuda:1 0.2G |
| H1 | `--use-sage-attention --fast --fp8_e4m3fn-text-enc` | 359.7 | 35.97 | +1.8% | cuda:0 14.3G, cuda:1 0.2G |
| H4 | `--use-sage-attention --fast --bf16-vae` | 373.8 | 37.38 | +5.8% | cuda:0 14.2G, cuda:1 0.2G |
| H3 | `--use-sage-attention --fast --cpu-vae` | — | — | **failed** | — |

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
| L4 | `--use-sage-attention --fast` | 109.6 | 10.96 | +0.0% | cuda:0 15.1G, cuda:1 0.2G |
| L3 | `--use-sage-attention --fast` | 129.2 | 12.93 | +18.0% | cuda:0 15.1G, cuda:1 0.2G |
| L2 | `--use-sage-attention --fast` | 156.5 | 15.65 | +42.8% | cuda:0 15.1G, cuda:1 0.2G |
| L1 | `--use-sage-attention --fast` | 234.2 | 23.43 | +113.8% | cuda:0 15.1G, cuda:1 0.2G |

### M. DisTorch2 sharding

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| M1 | `--use-sage-attention` | 348.0 | 34.80 | +0.0% | cuda:0 11.5G, cuda:1 0.2G |
| M2 | `--use-sage-attention` | 359.9 | 35.99 | +3.4% | cuda:0 13.0G, cuda:1 0.2G |
| M0 | `--use-sage-attention` | 376.0 | 37.60 | +8.1% | cuda:0 13.7G, cuda:1 0.2G |

### N. N

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| N1 | `--use-sage-attention --fast` | 29.8 | 29.76 | +0.0% | cuda:0 12.4G, cuda:1 0.2G |
| N0 | `--use-sage-attention --fast` | 103.1 | 103.06 | +246.3% | cuda:0 14.3G, cuda:1 0.2G |
| N2 | `--use-sage-attention` | 144.8 | 48.27 | +386.6% | cuda:0 11.3G, cuda:1 0.2G |

### P. Pipeline split (shared server)

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| P1 | `--use-sage-attention --disable-cuda-malloc --disable-async-offload` | — | — | **failed** | — |
| P0 | `--use-sage-attention --disable-cuda-malloc --disable-async-offload` | — | — | **failed** | — |

### Q. Pipeline split (isolated server)

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| Q1 | `--use-sage-attention` | — | — | **failed** | — |

### R. R

| row | config | server s | s/step | vs best | peak VRAM |
|---|---|---:|---:|---:|---|
| R2 | `--use-sage-attention --fast` | 271.1 | 27.11 | +0.0% | cuda:0 14.2G, cuda:1 9.3G |
| R3 | `--use-sage-attention --fast` | 276.8 | 27.68 | +2.1% | cuda:0 13.9G, cuda:1 8.3G |
| R1 | `--use-sage-attention --fast` | 284.2 | 28.42 | +4.9% | cuda:0 13.9G, cuda:1 12.4G |
