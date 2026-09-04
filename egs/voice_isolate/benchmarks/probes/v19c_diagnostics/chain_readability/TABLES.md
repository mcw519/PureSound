# chain_readability -- full tables (2026-09-04)

Scripts in this directory; cache at `<scratch>/agcache/{v8,v16,v11b}`.
v11b ckpt = `egs/voice_isolate/exp/dpcrn_v11b_compinv/lightning_logs/version_0/checkpoints/epoch=31-step=16000.ckpt`
(ep31 is the latest that exists; training was stopped mid-cosine by hand -- see `probes/v11b_VERDICT.md`).

## 1. `anchor_gate_sim.py readability`, all three tags (pooled frames)

| tag | chain | scope | W | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | nK | nS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8 | device | clip | 0.5 | 0.986 | 0.996 | 0.585 | 1.053 | 10.59 | -0.90 | 7743 | 11533 |
| v8 | device | clip | 1.0 | 0.989 | 0.997 | 0.591 | 1.043 | 10.62 | -0.55 | 7743 | 11533 |
| v8 | device | clip | 2.0 | 0.990 | 0.997 | 0.595 | 1.030 | 10.47 | 0.01 | 7743 | 11533 |
| v8 | device | session | 0.5 | 0.697 | 0.766 | 0.733 | 0.878 | 9.12 | 5.86 | 16514 | 11857 |
| v8 | device | session | 1.0 | 0.695 | 0.757 | 0.732 | 0.863 | 9.04 | 5.88 | 16514 | 11857 |
| v8 | device | session | 2.0 | 0.687 | 0.742 | 0.739 | 0.856 | 9.04 | 6.47 | 16514 | 11857 |
| v8 | qvf | clip | 0.5 | 0.272 | 0.808 | 0.701 | 0.489 | 10.98 | 5.10 | 2989 | 1718 |
| v8 | qvf | clip | 1.0 | 0.265 | 0.799 | 0.678 | 0.483 | 11.73 | 5.14 | 2989 | 1718 |
| v8 | qvf | clip | 2.0 | 0.256 | 0.787 | 0.678 | 0.483 | 12.30 | 5.63 | 2989 | 1718 |
| v8 | qvf | session | 0.5 | 0.340 | 0.518 | 1.225 | 0.797 | 4.56 | 3.60 | 3481 | 1012 |
| v8 | qvf | session | 1.0 | 0.357 | 0.517 | 1.240 | 0.825 | 4.08 | 3.90 | 3481 | 1012 |
| v8 | qvf | session | 2.0 | 0.395 | 0.491 | 1.215 | 0.849 | 3.81 | 4.30 | 3481 | 1012 |
| v16 | device | clip | 0.5 | 0.970 | 0.988 | 0.644 | 1.045 | 9.63 | 0.48 | 7743 | 11533 |
| v16 | device | clip | 1.0 | 0.977 | 0.990 | 0.653 | 1.040 | 9.61 | 0.30 | 7743 | 11533 |
| v16 | device | clip | 2.0 | 0.980 | 0.993 | 0.653 | 1.028 | 9.43 | 0.46 | 7743 | 11533 |
| v16 | device | session | 0.5 | 0.712 | 0.788 | 0.719 | 0.857 | 10.02 | 6.79 | 16514 | 11857 |
| v16 | device | session | 1.0 | 0.709 | 0.781 | 0.715 | 0.852 | 10.07 | 6.95 | 16514 | 11857 |
| v16 | device | session | 2.0 | 0.702 | 0.770 | 0.714 | 0.842 | 10.08 | 7.19 | 16514 | 11857 |
| v16 | qvf | clip | 0.5 | 0.258 | 0.702 | 0.818 | 0.547 | 9.02 | 4.92 | 2989 | 1718 |
| v16 | qvf | clip | 1.0 | 0.255 | 0.674 | 0.841 | 0.576 | 9.14 | 5.24 | 2989 | 1718 |
| v16 | qvf | clip | 2.0 | 0.248 | 0.660 | 0.851 | 0.576 | 9.14 | 5.45 | 2989 | 1718 |
| v16 | qvf | session | 0.5 | 0.341 | 0.452 | 1.426 | 0.780 | 3.75 | 4.57 | 3481 | 1012 |
| v16 | qvf | session | 1.0 | 0.357 | 0.441 | 1.463 | 0.809 | 3.29 | 4.57 | 3481 | 1012 |
| v16 | qvf | session | 2.0 | 0.373 | 0.420 | 1.471 | 0.819 | 3.41 | 4.55 | 3481 | 1012 |
| v11b | device | clip | 0.5 | 0.949 | 0.994 | 0.773 | 0.956 | 9.05 | -0.21 | 7743 | 11533 |
| v11b | device | clip | 1.0 | 0.962 | 0.994 | 0.767 | 0.951 | 9.12 | -0.19 | 7743 | 11533 |
| v11b | device | clip | 2.0 | 0.968 | 0.995 | 0.764 | 0.950 | 8.99 | -0.00 | 7743 | 11533 |
| v11b | device | session | 0.5 | 0.737 | 0.922 | 0.848 | 0.979 | 10.06 | 0.90 | 16514 | 11857 |
| v11b | device | session | 1.0 | 0.750 | 0.903 | 0.846 | 0.976 | 10.19 | 0.95 | 16514 | 11857 |
| v11b | device | session | 2.0 | 0.763 | 0.873 | 0.843 | 0.970 | 10.32 | 1.47 | 16514 | 11857 |
| v11b | qvf | clip | 0.5 | 0.273 | 0.847 | 0.851 | 0.751 | 9.58 | 3.65 | 2989 | 1718 |
| v11b | qvf | clip | 1.0 | 0.258 | 0.841 | 0.843 | 0.742 | 10.51 | 3.76 | 2989 | 1718 |
| v11b | qvf | clip | 2.0 | 0.245 | 0.841 | 0.813 | 0.730 | 10.63 | 4.28 | 2989 | 1718 |
| v11b | qvf | session | 0.5 | 0.341 | 0.519 | 1.147 | 0.815 | 7.80 | 8.29 | 3481 | 1012 |
| v11b | qvf | session | 1.0 | 0.377 | 0.528 | 1.147 | 0.808 | 7.91 | 8.21 | 3481 | 1012 |
| v11b | qvf | session | 2.0 | 0.392 | 0.508 | 1.168 | 0.816 | 7.58 | 8.42 | 3481 | 1012 |

## 2. Prefix (anchor) reads, W = 1 s

| tag | chain | prefix condition | n | median m | median DRR dB | p10 m | p90 m |
|---|---|---|---|---|---|---|---|
| v8 | device | event | 38 | 0.556 | 10.89 | 0.556 | 0.557 |
| v8 | device | far_speech | 37 | 0.875 | 1.86 | 0.875 | 1.012 |
| v8 | device | floor | 37 | 0.618 | 2.96 | 0.618 | 0.619 |
| v8 | device | near_speech | 38 | 0.526 | 11.55 | 0.526 | 0.583 |
| v8 | device | stream | 34 | 0.762 | 4.90 | 0.558 | 0.932 |
| v8 | qvf | far_speech | 20 | 0.715 | 5.58 | 0.660 | 0.905 |
| v8 | qvf | floor | 12 | 0.704 | 4.12 | 0.613 | 0.731 |
| v8 | qvf | near_speech | 20 | 0.869 | 6.11 | 0.786 | 1.167 |
| v8 | qvf | stream | 12 | 0.991 | 6.45 | 0.593 | 1.169 |
| v16 | device | event | 38 | 0.575 | 10.24 | 0.574 | 0.575 |
| v16 | device | far_speech | 37 | 1.004 | -1.09 | 0.960 | 1.004 |
| v16 | device | floor | 37 | 0.606 | 1.36 | 0.606 | 0.608 |
| v16 | device | near_speech | 38 | 0.561 | 10.44 | 0.561 | 0.611 |
| v16 | device | stream | 34 | 0.812 | 3.76 | 0.602 | 0.988 |
| v16 | qvf | far_speech | 20 | 0.768 | 4.47 | 0.698 | 1.132 |
| v16 | qvf | floor | 12 | 0.665 | 3.15 | 0.623 | 0.682 |
| v16 | qvf | near_speech | 20 | 1.150 | 3.18 | 1.038 | 1.476 |
| v16 | qvf | stream | 12 | 1.254 | 4.88 | 0.653 | 1.588 |
| v11b | device | event | 38 | 0.721 | 10.71 | 0.721 | 0.721 |
| v11b | device | far_speech | 37 | 1.032 | 2.85 | 0.820 | 1.032 |
| v11b | device | floor | 37 | 0.702 | 5.95 | 0.702 | 0.702 |
| v11b | device | near_speech | 38 | 0.744 | 8.70 | 0.680 | 0.744 |
| v11b | device | stream | 34 | 0.830 | 4.67 | 0.718 | 0.938 |
| v11b | qvf | far_speech | 20 | 0.771 | 5.54 | 0.750 | 1.037 |
| v11b | qvf | floor | 12 | 0.532 | 4.55 | 0.500 | 0.604 |
| v11b | qvf | near_speech | 20 | 0.988 | 6.97 | 0.961 | 1.077 |
| v11b | qvf | stream | 12 | 1.146 | 7.05 | 0.772 | 1.367 |

## 3. Per-recording-group breakdown (W = 1 s, condition `none`)

| tag | group | scope | chain | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | nK | nS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8 | 180d | clip | device | 0.999 | 0.999 | 0.524 | 1.036 | 10.63 | -0.54 | 1848 | 3621 |
| v8 | 180d | session | device | 0.863 | 0.843 | 0.660 | 0.952 | 9.28 | 5.18 | 4654 | 3716 |
| v8 | 270d | clip | device | 0.998 | 0.995 | 0.568 | 1.033 | 11.30 | 0.56 | 1645 | 4390 |
| v8 | 270d | session | device | 0.608 | 0.717 | 0.790 | 0.841 | 9.19 | 5.92 | 4260 | 4481 |
| v8 | 90d | clip | device | 0.993 | 0.998 | 0.606 | 1.072 | 10.70 | -1.66 | 3649 | 3522 |
| v8 | 90d | session | device | 0.601 | 0.719 | 0.748 | 0.793 | 8.86 | 6.70 | 7600 | 3660 |
| v8 | qvf_gym | clip | qvf | 0.191 | 0.324 | 1.008 | 0.846 | 6.56 | 7.27 | 582 | 120 |
| v8 | qvf_gym | session | qvf | 0.388 | 0.570 | 1.494 | 1.299 | 1.00 | 0.18 | 1365 | 122 |
| v8 | qvf_plumbing | clip | qvf | 0.170 | 0.761 | 0.774 | 0.708 | 4.06 | 1.93 | 167 | 193 |
| v8 | qvf_price | clip | qvf | 0.047 | 0.644 | 1.211 | 0.810 | 6.26 | 3.28 | 485 | 456 |
| v8 | qvf_price | session | qvf | 0.171 | 0.559 | 1.334 | 1.012 | 2.85 | 2.44 | 1134 | 494 |
| v8 | qvf_scenario3 | clip | qvf | 0.041 | 1.000 | 0.571 | 0.369 | 13.08 | 5.31 | 960 | 365 |
| v8 | qvf_scenario3 | session | qvf | 0.835 | 0.974 | 0.550 | 0.638 | 12.73 | 9.76 | 982 | 396 |
| v16 | 180d | clip | device | 0.992 | 0.994 | 0.553 | 1.021 | 9.76 | 0.24 | 1848 | 3621 |
| v16 | 180d | session | device | 0.852 | 0.836 | 0.640 | 0.865 | 10.25 | 6.61 | 4654 | 3716 |
| v16 | 270d | clip | device | 0.981 | 0.984 | 0.634 | 1.033 | 10.46 | 0.74 | 1645 | 4390 |
| v16 | 270d | session | device | 0.635 | 0.782 | 0.755 | 0.855 | 10.25 | 6.91 | 4260 | 4481 |
| v16 | 90d | clip | device | 0.982 | 0.999 | 0.674 | 1.078 | 9.30 | 0.03 | 3649 | 3522 |
| v16 | 90d | session | device | 0.667 | 0.742 | 0.745 | 0.836 | 9.81 | 7.37 | 7600 | 3660 |
| v16 | qvf_gym | clip | qvf | 0.087 | 0.106 | 1.303 | 0.993 | 3.48 | 5.32 | 582 | 120 |
| v16 | qvf_gym | session | qvf | 0.604 | 0.862 | 1.614 | 1.794 | 2.05 | -1.10 | 1365 | 122 |
| v16 | qvf_plumbing | clip | qvf | 0.074 | 0.506 | 0.934 | 0.823 | 0.71 | -0.01 | 167 | 193 |
| v16 | qvf_price | clip | qvf | 0.000 | 0.506 | 1.590 | 0.902 | 3.51 | 2.65 | 485 | 456 |
| v16 | qvf_price | session | qvf | 0.120 | 0.274 | 1.594 | 1.014 | 2.04 | 4.18 | 1134 | 494 |
| v16 | qvf_scenario3 | clip | qvf | 0.035 | 0.990 | 0.639 | 0.404 | 11.70 | 4.82 | 960 | 365 |
| v16 | qvf_scenario3 | session | qvf | 0.843 | 0.934 | 0.600 | 0.698 | 11.88 | 9.84 | 982 | 396 |
| v11b | 180d | clip | device | 0.986 | 0.994 | 0.732 | 0.911 | 8.67 | -0.38 | 1848 | 3621 |
| v11b | 180d | session | device | 0.790 | 0.930 | 0.814 | 0.937 | 10.28 | 1.07 | 4654 | 3716 |
| v11b | 270d | clip | device | 0.987 | 0.999 | 0.789 | 0.959 | 9.78 | 0.01 | 1645 | 4390 |
| v11b | 270d | session | device | 0.772 | 0.910 | 0.883 | 1.002 | 10.19 | 0.60 | 4260 | 4481 |
| v11b | 90d | clip | device | 0.942 | 0.996 | 0.774 | 0.968 | 9.56 | -0.08 | 3649 | 3522 |
| v11b | 90d | session | device | 0.699 | 0.868 | 0.847 | 0.966 | 10.16 | 1.58 | 7600 | 3660 |
| v11b | qvf_gym | clip | qvf | 0.091 | 0.529 | 1.137 | 0.995 | 5.56 | 5.63 | 582 | 120 |
| v11b | qvf_gym | session | qvf | 0.782 | 0.592 | 1.171 | 1.319 | 7.25 | 6.83 | 1365 | 122 |
| v11b | qvf_plumbing | clip | qvf | 0.000 | 0.637 | 0.997 | 0.806 | 3.26 | 2.25 | 167 | 193 |
| v11b | qvf_price | clip | qvf | 0.073 | 0.805 | 1.189 | 0.804 | 5.75 | 2.03 | 485 | 456 |
| v11b | qvf_price | session | qvf | 0.188 | 0.529 | 1.264 | 0.885 | 6.64 | 6.37 | 1134 | 494 |
| v11b | qvf_scenario3 | clip | qvf | 0.092 | 0.993 | 0.765 | 0.620 | 11.85 | 3.78 | 960 | 365 |
| v11b | qvf_scenario3 | session | qvf | 0.694 | 0.926 | 0.735 | 0.763 | 12.41 | 10.77 | 982 | 396 |

## 4. Clip-level bootstrap CI (resampling unit = recording/clip, B = 2000)

| tag | chain | scope | readout | AUC | 95% CI | n keep units | n supp units |
|---|---|---|---|---|---|---|---|
| v8 | device | clip | dist | 0.989 | [0.969, 0.999] | 18 | 14 |
| v8 | device | clip | drr | 0.997 | [0.992, 0.999] | 18 | 14 |
| v8 | device | session | dist | 0.695 | [0.594, 0.789] | 3 | 3 |
| v8 | device | session | drr | 0.757 | [0.707, 0.831] | 3 | 3 |
| v8 | qvf | clip | dist | 0.265 | [0.054, 0.668] | 9 | 6 |
| v8 | qvf | clip | drr | 0.799 | [0.586, 0.925] | 9 | 6 |
| v8 | qvf | session | dist | 0.357 | [0.056, 0.935] | 3 | 3 |
| v8 | qvf | session | drr | 0.517 | [0.104, 0.984] | 3 | 3 |
| v16 | device | clip | dist | 0.977 | [0.954, 0.991] | 18 | 14 |
| v16 | device | clip | drr | 0.990 | [0.978, 0.998] | 18 | 14 |
| v16 | device | session | dist | 0.709 | [0.643, 0.813] | 3 | 3 |
| v16 | device | session | drr | 0.781 | [0.744, 0.827] | 3 | 3 |
| v16 | qvf | clip | dist | 0.255 | [0.046, 0.648] | 9 | 6 |
| v16 | qvf | clip | drr | 0.674 | [0.340, 0.887] | 9 | 6 |
| v16 | qvf | session | dist | 0.357 | [0.043, 0.903] | 3 | 3 |
| v16 | qvf | session | drr | 0.441 | [0.101, 0.964] | 3 | 3 |
| v11b | device | clip | dist | 0.962 | [0.923, 0.988] | 18 | 14 |
| v11b | device | clip | drr | 0.994 | [0.986, 0.999] | 18 | 14 |
| v11b | device | session | dist | 0.750 | [0.682, 0.813] | 3 | 3 |
| v11b | device | session | drr | 0.903 | [0.872, 0.935] | 3 | 3 |
| v11b | qvf | clip | dist | 0.258 | [0.052, 0.516] | 9 | 6 |
| v11b | qvf | clip | drr | 0.841 | [0.646, 0.959] | 9 | 6 |
| v11b | qvf | session | dist | 0.377 | [0.081, 0.862] | 3 | 3 |
| v11b | qvf | session | drr | 0.528 | [0.084, 0.971] | 3 | 3 |

## 5. PAIRED bootstrap on AUC differences (same resampled clips for every tag, B = 4000)

| chain | scope | readout | pair | AUC v8 | AUC v16 | AUC v11b | delta | 95% CI | p (two-sided) |
|---|---|---|---|---|---|---|---|---|---|
| device | clip | dist | v11b-v8 | 0.989 | 0.977 | 0.962 | -0.027 | [-0.065, +0.005] | 0.102 |
| device | clip | dist | v11b-v16 | 0.989 | 0.977 | 0.962 | -0.015 | [-0.052, +0.019] | 0.411 |
| device | clip | dist | v16-v8 | 0.989 | 0.977 | 0.962 | -0.012 | [-0.030, -0.001] | 0.033 |
| device | clip | drr | v11b-v8 | 0.997 | 0.990 | 0.994 | -0.002 | [-0.007, +0.001] | 0.198 |
| device | clip | drr | v11b-v16 | 0.997 | 0.990 | 0.994 | +0.004 | [-0.004, +0.014] | 0.349 |
| device | clip | drr | v16-v8 | 0.997 | 0.990 | 0.994 | -0.006 | [-0.017, -0.000] | 0.040 |
| device | session | dist | v11b-v8 | 0.695 | 0.709 | 0.750 | +0.055 | [-0.080, +0.142] | 0.389 |
| device | session | dist | v11b-v16 | 0.695 | 0.709 | 0.750 | +0.041 | [-0.044, +0.117] | 0.455 |
| device | session | dist | v16-v8 | 0.695 | 0.709 | 0.750 | +0.014 | [-0.056, +0.085] | 0.683 |
| device | session | drr | v11b-v8 | 0.757 | 0.781 | 0.903 | +0.145 | [+0.102, +0.188] | 0.000 |
| device | session | drr | v11b-v16 | 0.757 | 0.781 | 0.903 | +0.122 | [+0.081, +0.152] | 0.000 |
| device | session | drr | v16-v8 | 0.757 | 0.781 | 0.903 | +0.024 | [-0.010, +0.055] | 0.165 |
| qvf | clip | dist | v11b-v8 | 0.265 | 0.255 | 0.258 | -0.006 | [-0.161, +0.115] | 0.893 |
| qvf | clip | dist | v11b-v16 | 0.265 | 0.255 | 0.258 | +0.003 | [-0.138, +0.107] | 0.953 |
| qvf | clip | dist | v16-v8 | 0.265 | 0.255 | 0.258 | -0.009 | [-0.045, +0.015] | 0.526 |
| qvf | clip | drr | v11b-v8 | 0.799 | 0.674 | 0.841 | +0.042 | [-0.017, +0.138] | 0.167 |
| qvf | clip | drr | v11b-v16 | 0.799 | 0.674 | 0.841 | +0.167 | [+0.051, +0.337] | 0.000 |
| qvf | clip | drr | v16-v8 | 0.799 | 0.674 | 0.841 | -0.125 | [-0.276, -0.026] | 0.003 |
| qvf | session | dist | v11b-v8 | 0.357 | 0.357 | 0.377 | +0.020 | [-0.077, +0.219] | 0.707 |
| qvf | session | dist | v11b-v16 | 0.357 | 0.357 | 0.377 | +0.020 | [-0.082, +0.102] | 0.795 |
| qvf | session | dist | v16-v8 | 0.357 | 0.357 | 0.377 | +0.001 | [-0.039, +0.171] | 0.961 |
| qvf | session | drr | v11b-v8 | 0.517 | 0.441 | 0.528 | +0.011 | [-0.129, +0.131] | 0.980 |
| qvf | session | drr | v11b-v16 | 0.517 | 0.441 | 0.528 | +0.087 | [-0.154, +0.267] | 0.576 |
| qvf | session | drr | v16-v8 | 0.517 | 0.441 | 0.528 | -0.076 | [-0.201, +0.111] | 0.302 |

## 6. BROKEN offline chain probe (`comp_readability.py`) -- kept as the record of the defect

`compressor_gain` RETURNS a gain curve; this run (and `benchmarks/probes/eq_probe.py`) never multiplied it into the signal, and on these clips the curve is all-ones, so the model was fed a CONSTANT DC signal. Identical numbers across three operating points are the signature.

| tag | chain | cond | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | GR mean dB | GR max dB |
|---|---|---|---|---|---|---|---|---|---|---|
| v8 | device | none | 0.989 | 0.997 | 0.591 | 1.043 | 10.62 | -0.55 | nan | nan |
| v8 | device | comp_thr-34_r2 | 0.559 | 0.575 | 0.582 | 0.608 | -3.80 | -4.11 | nan | nan |
| v8 | device | comp_thr-28_r4 | 0.559 | 0.582 | 0.582 | 0.608 | -3.79 | -4.11 | nan | nan |
| v8 | device | comp_thr-22_r6 | 0.559 | 0.582 | 0.582 | 0.608 | -3.79 | -4.11 | nan | nan |
| v8 | device | broadcast | 0.974 | 0.987 | 0.661 | 1.014 | 10.22 | 1.39 | nan | nan |
| v8 | device | comp+broadcast | 0.380 | 0.451 | 0.472 | 0.457 | 5.43 | 5.66 | nan | nan |
| v8 | device | dark_tilt(ctrl) | 0.987 | 0.997 | 0.572 | 1.096 | 10.42 | -1.83 | nan | nan |
| v8 | qvf | none | 0.265 | 0.799 | 0.678 | 0.483 | 11.73 | 5.14 | nan | nan |
| v8 | qvf | comp_thr-34_r2 | 0.500 | 0.464 | 0.505 | 0.504 | -2.12 | -1.97 | nan | nan |
| v8 | qvf | comp_thr-28_r4 | 0.455 | 0.416 | 0.514 | 0.498 | -2.40 | -1.72 | nan | nan |
| v8 | qvf | comp_thr-22_r6 | 0.517 | 0.470 | 0.494 | 0.501 | -1.62 | -1.77 | nan | nan |
| v8 | qvf | broadcast | 0.246 | 0.841 | 0.732 | 0.475 | 11.91 | 5.20 | nan | nan |
| v8 | qvf | comp+broadcast | 0.255 | 0.566 | 0.607 | 0.434 | 6.66 | 7.02 | nan | nan |
| v8 | qvf | dark_tilt(ctrl) | 0.268 | 0.772 | 0.671 | 0.485 | 11.18 | 5.40 | nan | nan |
| v16 | device | none | 0.977 | 0.990 | 0.653 | 1.040 | 9.61 | 0.30 | nan | nan |
| v16 | device | comp_thr-34_r2 | 0.586 | 0.544 | 0.744 | 0.748 | -0.33 | -0.38 | nan | nan |
| v16 | device | comp_thr-28_r4 | 0.567 | 0.545 | 0.745 | 0.748 | -0.34 | -0.38 | nan | nan |
| v16 | device | comp_thr-22_r6 | 0.567 | 0.546 | 0.745 | 0.748 | -0.34 | -0.38 | nan | nan |
| v16 | device | broadcast | 0.918 | 0.983 | 0.717 | 1.001 | 9.51 | 2.00 | nan | nan |
| v16 | device | comp+broadcast | 0.438 | 0.464 | 0.790 | 0.780 | 2.17 | 2.19 | nan | nan |
| v16 | device | dark_tilt(ctrl) | 0.980 | 0.988 | 0.648 | 1.103 | 9.11 | -0.33 | nan | nan |
| v16 | qvf | none | 0.255 | 0.674 | 0.841 | 0.576 | 9.14 | 5.24 | nan | nan |
| v16 | qvf | comp_thr-34_r2 | 0.476 | 0.325 | 0.777 | 0.775 | -1.38 | -0.97 | nan | nan |
| v16 | qvf | comp_thr-28_r4 | 0.402 | 0.181 | 0.785 | 0.777 | -1.70 | -0.95 | nan | nan |
| v16 | qvf | comp_thr-22_r6 | 0.469 | 0.199 | 0.778 | 0.770 | -1.20 | -0.76 | nan | nan |
| v16 | qvf | broadcast | 0.265 | 0.746 | 0.840 | 0.588 | 9.60 | 5.01 | nan | nan |
| v16 | qvf | comp+broadcast | 0.244 | 0.770 | 0.974 | 0.737 | 3.81 | 2.04 | nan | nan |
| v16 | qvf | dark_tilt(ctrl) | 0.254 | 0.646 | 0.844 | 0.591 | 8.88 | 5.31 | nan | nan |
| v11b | device | none | 0.962 | 0.994 | 0.767 | 0.951 | 9.12 | -0.19 | nan | nan |
| v11b | device | comp_thr-34_r2 | 0.563 | 0.565 | 1.388 | 1.498 | -5.34 | -7.05 | nan | nan |
| v11b | device | comp_thr-28_r4 | 0.561 | 0.560 | 1.389 | 1.498 | -5.38 | -7.05 | nan | nan |
| v11b | device | comp_thr-22_r6 | 0.561 | 0.560 | 1.389 | 1.498 | -5.38 | -7.05 | nan | nan |
| v11b | device | broadcast | 0.949 | 0.994 | 0.792 | 0.973 | 8.99 | 0.08 | nan | nan |
| v11b | device | comp+broadcast | 0.581 | 0.501 | 0.962 | 1.018 | 5.78 | 5.79 | nan | nan |
| v11b | device | dark_tilt(ctrl) | 0.972 | 0.993 | 0.781 | 0.987 | 8.92 | -0.61 | nan | nan |
| v11b | qvf | none | 0.259 | 0.841 | 0.843 | 0.742 | 10.51 | 3.76 | nan | nan |
| v11b | qvf | comp_thr-34_r2 | 0.512 | 0.512 | 1.157 | 1.158 | -0.55 | -0.88 | nan | nan |
| v11b | qvf | comp_thr-28_r4 | 0.516 | 0.522 | 1.176 | 1.158 | -0.33 | -0.91 | nan | nan |
| v11b | qvf | comp_thr-22_r6 | 0.516 | 0.504 | 1.147 | 1.156 | -0.55 | -0.84 | nan | nan |
| v11b | qvf | broadcast | 0.186 | 0.856 | 0.827 | 0.666 | 11.38 | 3.80 | nan | nan |
| v11b | qvf | comp+broadcast | 0.528 | 0.314 | 0.850 | 0.874 | 4.21 | 7.47 | nan | nan |
| v11b | qvf | dark_tilt(ctrl) | 0.287 | 0.784 | 0.866 | 0.775 | 9.82 | 4.48 | nan | nan |

## 7. FIXED offline chain probe (`comp_readability2.py`)

Gain multiplied in, clip normalised to the recipe's `gain_normalized_to: -28` dBFS RMS before the compressor and restored after, so the stage operates in its training range; `GRmean/GRmax` is the gain reduction it actually applied. `wshape_*` is the |x|^p waveshaper of `compression_probe.py` (a different operator -- the one whose causality was established in 2026-08-21).

| tag | chain | cond | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | GR mean dB | GR max dB |
|---|---|---|---|---|---|---|---|---|---|---|
| v8 | device | none | 0.989 | 0.997 | 0.591 | 1.043 | 10.62 | -0.55 | 0.00 | 0.00 |
| v8 | device | comp_thr-34_r2 | 0.987 | 0.998 | 0.578 | 1.043 | 10.90 | -1.01 | 0.18 | 8.05 |
| v8 | device | comp_thr-28_r4 | 0.990 | 0.998 | 0.591 | 1.051 | 10.70 | -0.74 | 0.07 | 9.44 |
| v8 | device | comp_thr-22_r6 | 0.989 | 0.997 | 0.591 | 1.045 | 10.64 | -0.47 | 0.00 | 5.91 |
| v8 | device | broadcast | 0.974 | 0.987 | 0.661 | 1.014 | 10.22 | 1.39 | 0.00 | 0.00 |
| v8 | device | comp+broadcast | 0.978 | 0.991 | 0.657 | 1.017 | 10.31 | 1.03 | 0.07 | 9.44 |
| v8 | device | dark_tilt(ctrl) | 0.987 | 0.997 | 0.572 | 1.096 | 10.42 | -1.83 | 0.00 | 0.00 |
| v8 | device | wshape_p0.8 | 0.981 | 0.988 | 0.600 | 1.025 | 10.33 | 0.70 | 0.00 | 0.00 |
| v8 | device | wshape_p0.6 | 0.957 | 0.955 | 0.679 | 0.974 | 9.69 | 3.42 | 0.00 | 0.00 |
| v8 | device | wshape_p0.4 | 0.880 | 0.908 | 0.775 | 0.950 | 9.02 | 5.00 | 0.00 | 0.00 |
| v8 | device | comp_r6_hardknee | 0.983 | 0.998 | 0.623 | 1.104 | 10.95 | -1.53 | 1.05 | 16.22 |
| v8 | qvf | none | 0.265 | 0.799 | 0.678 | 0.483 | 11.73 | 5.14 | 0.00 | 0.00 |
| v8 | qvf | comp_thr-34_r2 | 0.255 | 0.787 | 0.659 | 0.470 | 11.93 | 5.35 | 0.18 | 8.05 |
| v8 | qvf | comp_thr-28_r4 | 0.256 | 0.813 | 0.664 | 0.474 | 11.95 | 5.12 | 0.07 | 9.44 |
| v8 | qvf | comp_thr-22_r6 | 0.261 | 0.802 | 0.680 | 0.483 | 11.65 | 5.13 | 0.00 | 5.91 |
| v8 | qvf | broadcast | 0.246 | 0.841 | 0.732 | 0.475 | 11.91 | 5.20 | 0.00 | 0.00 |
| v8 | qvf | comp+broadcast | 0.235 | 0.836 | 0.724 | 0.444 | 12.31 | 5.31 | 0.07 | 9.44 |
| v8 | qvf | dark_tilt(ctrl) | 0.268 | 0.772 | 0.671 | 0.485 | 11.18 | 5.40 | 0.00 | 0.00 |
| v8 | qvf | wshape_p0.8 | 0.291 | 0.899 | 0.692 | 0.539 | 12.15 | 4.96 | 0.00 | 0.00 |
| v8 | qvf | wshape_p0.6 | 0.319 | 0.919 | 0.674 | 0.608 | 13.10 | 5.65 | 0.00 | 0.00 |
| v8 | qvf | wshape_p0.4 | 0.276 | 0.905 | 0.731 | 0.625 | 12.24 | 6.72 | 0.00 | 0.00 |
| v8 | qvf | comp_r6_hardknee | 0.248 | 0.845 | 0.666 | 0.500 | 11.86 | 5.67 | 1.05 | 16.22 |
| v16 | device | none | 0.977 | 0.990 | 0.653 | 1.040 | 9.61 | 0.30 | 0.00 | 0.00 |
| v16 | device | comp_thr-34_r2 | 0.984 | 0.995 | 0.643 | 1.043 | 9.92 | -1.00 | 0.18 | 8.05 |
| v16 | device | comp_thr-28_r4 | 0.981 | 0.993 | 0.654 | 1.037 | 9.69 | -0.23 | 0.07 | 9.44 |
| v16 | device | comp_thr-22_r6 | 0.977 | 0.990 | 0.656 | 1.034 | 9.63 | 0.33 | 0.00 | 5.91 |
| v16 | device | broadcast | 0.918 | 0.983 | 0.717 | 1.001 | 9.51 | 2.00 | 0.00 | 0.00 |
| v16 | device | comp+broadcast | 0.927 | 0.987 | 0.718 | 0.998 | 9.58 | 1.45 | 0.07 | 9.44 |
| v16 | device | dark_tilt(ctrl) | 0.980 | 0.988 | 0.648 | 1.103 | 9.11 | -0.33 | 0.00 | 0.00 |
| v16 | device | wshape_p0.8 | 0.980 | 0.981 | 0.653 | 1.050 | 9.56 | 0.57 | 0.00 | 0.00 |
| v16 | device | wshape_p0.6 | 0.957 | 0.960 | 0.693 | 1.028 | 9.17 | 2.58 | 0.00 | 0.00 |
| v16 | device | wshape_p0.4 | 0.902 | 0.956 | 0.787 | 0.996 | 8.21 | 4.39 | 0.00 | 0.00 |
| v16 | device | comp_r6_hardknee | 0.974 | 0.995 | 0.686 | 1.119 | 9.74 | -1.70 | 1.05 | 16.22 |
| v16 | qvf | none | 0.255 | 0.674 | 0.841 | 0.576 | 9.14 | 5.24 | 0.00 | 0.00 |
| v16 | qvf | comp_thr-34_r2 | 0.253 | 0.662 | 0.796 | 0.549 | 9.48 | 5.07 | 0.18 | 8.05 |
| v16 | qvf | comp_thr-28_r4 | 0.251 | 0.666 | 0.806 | 0.555 | 9.54 | 5.30 | 0.07 | 9.44 |
| v16 | qvf | comp_thr-22_r6 | 0.253 | 0.669 | 0.844 | 0.570 | 9.12 | 5.24 | 0.00 | 5.91 |
| v16 | qvf | broadcast | 0.265 | 0.746 | 0.840 | 0.588 | 9.60 | 5.01 | 0.00 | 0.00 |
| v16 | qvf | comp+broadcast | 0.251 | 0.720 | 0.808 | 0.548 | 10.16 | 5.22 | 0.07 | 9.44 |
| v16 | qvf | dark_tilt(ctrl) | 0.254 | 0.646 | 0.844 | 0.591 | 8.88 | 5.31 | 0.00 | 0.00 |
| v16 | qvf | wshape_p0.8 | 0.251 | 0.795 | 0.831 | 0.604 | 9.38 | 4.68 | 0.00 | 0.00 |
| v16 | qvf | wshape_p0.6 | 0.262 | 0.908 | 0.772 | 0.689 | 10.83 | 5.15 | 0.00 | 0.00 |
| v16 | qvf | wshape_p0.4 | 0.223 | 0.914 | 0.767 | 0.682 | 10.63 | 5.64 | 0.00 | 0.00 |
| v16 | qvf | comp_r6_hardknee | 0.247 | 0.685 | 0.789 | 0.565 | 9.40 | 5.23 | 1.05 | 16.22 |
| v11b | device | none | 0.962 | 0.994 | 0.767 | 0.951 | 9.12 | -0.19 | 0.00 | 0.00 |
| v11b | device | comp_thr-34_r2 | 0.962 | 0.996 | 0.767 | 0.932 | 9.18 | -0.30 | 0.18 | 8.05 |
| v11b | device | comp_thr-28_r4 | 0.962 | 0.995 | 0.770 | 0.943 | 9.15 | -0.25 | 0.07 | 9.44 |
| v11b | device | comp_thr-22_r6 | 0.963 | 0.994 | 0.769 | 0.951 | 9.13 | -0.21 | 0.00 | 5.91 |
| v11b | device | broadcast | 0.949 | 0.994 | 0.792 | 0.973 | 8.99 | 0.08 | 0.00 | 0.00 |
| v11b | device | comp+broadcast | 0.949 | 0.995 | 0.791 | 0.963 | 9.04 | -0.05 | 0.07 | 9.44 |
| v11b | device | dark_tilt(ctrl) | 0.972 | 0.993 | 0.781 | 0.987 | 8.92 | -0.61 | 0.00 | 0.00 |
| v11b | device | wshape_p0.8 | 0.952 | 0.987 | 0.780 | 0.978 | 8.46 | 0.10 | 0.00 | 0.00 |
| v11b | device | wshape_p0.6 | 0.917 | 0.981 | 0.754 | 0.908 | 7.52 | 0.81 | 0.00 | 0.00 |
| v11b | device | wshape_p0.4 | 0.865 | 0.960 | 0.756 | 0.881 | 7.14 | 2.90 | 0.00 | 0.00 |
| v11b | device | comp_r6_hardknee | 0.948 | 0.996 | 0.780 | 0.932 | 9.18 | -0.38 | 1.05 | 16.22 |
| v11b | qvf | none | 0.259 | 0.841 | 0.843 | 0.742 | 10.51 | 3.76 | 0.00 | 0.00 |
| v11b | qvf | comp_thr-34_r2 | 0.262 | 0.848 | 0.837 | 0.740 | 10.43 | 3.73 | 0.18 | 8.05 |
| v11b | qvf | comp_thr-28_r4 | 0.263 | 0.848 | 0.842 | 0.739 | 10.53 | 3.67 | 0.07 | 9.44 |
| v11b | qvf | comp_thr-22_r6 | 0.260 | 0.845 | 0.846 | 0.742 | 10.50 | 3.75 | 0.00 | 5.91 |
| v11b | qvf | broadcast | 0.186 | 0.856 | 0.827 | 0.666 | 11.38 | 3.80 | 0.00 | 0.00 |
| v11b | qvf | comp+broadcast | 0.189 | 0.867 | 0.823 | 0.642 | 11.32 | 3.76 | 0.07 | 9.44 |
| v11b | qvf | dark_tilt(ctrl) | 0.287 | 0.784 | 0.866 | 0.775 | 9.82 | 4.48 | 0.00 | 0.00 |
| v11b | qvf | wshape_p0.8 | 0.232 | 0.949 | 0.844 | 0.733 | 10.64 | 3.00 | 0.00 | 0.00 |
| v11b | qvf | wshape_p0.6 | 0.131 | 0.955 | 0.845 | 0.674 | 11.01 | 3.48 | 0.00 | 0.00 |
| v11b | qvf | wshape_p0.4 | 0.122 | 0.905 | 0.783 | 0.636 | 9.59 | 4.60 | 0.00 | 0.00 |
| v11b | qvf | comp_r6_hardknee | 0.269 | 0.865 | 0.837 | 0.750 | 10.35 | 3.72 | 1.05 | 16.22 |

## 8. Fresh linear probe on the SAME cached bottleneck frames (`linear_probe.py`)

Logistic regression (C = 0.1, standardised) on the 128-d pooled bottleneck window the DistHead reads.
Holdout unit = recording group, so no group is in both fit and test. This separates *is the near/far
information in the representation* from *does the DistHead's learned mapping point the right way*.

| tag | arm | AUC | detail |
|---|---|---|---|
| v8 | device-fit -> device-test (LOGO) | 0.999 | 0.999, 1.000, 0.997 |
| v8 | device-fit -> QVF-test (pooled) | 0.789 | nK=3160 nS=1832 |
| v8 | device-fit -> QVF-test (per group) | n/a | qvf_gym=0.686, qvf_plumbing=1.000, qvf_price=0.978, qvf_scenario3=0.951 |
| v8 | QVF-fit -> QVF-test (LOGO) | 0.999 | qvf_gym=1.000, qvf_plumbing=0.999, qvf_price=0.997, qvf_scenario3=1.000 |
| v8 | QVF within-group in-sample (info ceiling) | 1.000 | qvf_gym=1.000, qvf_plumbing=1.000, qvf_price=1.000, qvf_scenario3=1.000 |
| v16 | device-fit -> device-test (LOGO) | 0.999 | 0.999, 1.000, 0.998 |
| v16 | device-fit -> QVF-test (pooled) | 0.827 | nK=3160 nS=1832 |
| v16 | device-fit -> QVF-test (per group) | n/a | qvf_gym=0.672, qvf_plumbing=0.991, qvf_price=0.955, qvf_scenario3=0.965 |
| v16 | QVF-fit -> QVF-test (LOGO) | 0.998 | qvf_gym=0.999, qvf_plumbing=0.999, qvf_price=0.995, qvf_scenario3=1.000 |
| v16 | QVF within-group in-sample (info ceiling) | 1.000 | qvf_gym=1.000, qvf_plumbing=1.000, qvf_price=1.000, qvf_scenario3=1.000 |
| v11b | device-fit -> device-test (LOGO) | 1.000 | 0.999, 1.000, 0.999 |
| v11b | device-fit -> QVF-test (pooled) | 0.680 | nK=3160 nS=1832 |
| v11b | device-fit -> QVF-test (per group) | n/a | qvf_gym=0.855, qvf_plumbing=0.970, qvf_price=0.401, qvf_scenario3=0.953 |
| v11b | QVF-fit -> QVF-test (LOGO) | 0.986 | qvf_gym=1.000, qvf_plumbing=0.999, qvf_price=0.947, qvf_scenario3=1.000 |
| v11b | QVF within-group in-sample (info ceiling) | 1.000 | qvf_gym=1.000, qvf_plumbing=1.000, qvf_price=1.000, qvf_scenario3=1.000 |

## 9. All three DistHead output slots (`slot2_auc.py`, W = 1 s)

Slot 2 is the interferer-distance regression, never used by `readability` before.

| tag | chain | scope | AUC DRR | AUC fg dist | AUC itf dist | itf keep m | itf supp m |
|---|---|---|---|---|---|---|---|
| v8 | device | clip | 0.994 | 0.985 | 0.703 | 2.525 | 2.049 |
| v8 | device | session | 0.756 | 0.696 | 0.698 | 2.284 | 2.097 |
| v8 | qvf | clip | 0.787 | 0.273 | 0.168 | 1.820 | 2.643 |
| v8 | qvf | session | 0.518 | 0.362 | 0.146 | 1.734 | 2.472 |
| v16 | device | clip | 0.983 | 0.965 | 0.808 | 2.609 | 1.971 |
| v16 | device | session | 0.779 | 0.709 | 0.564 | 2.024 | 2.013 |
| v16 | qvf | clip | 0.663 | 0.266 | 0.129 | 1.768 | 2.735 |
| v16 | qvf | session | 0.443 | 0.361 | 0.183 | 1.491 | 2.334 |
| v11b | device | clip | 0.983 | 0.957 | 0.710 | 2.203 | 2.000 |
| v11b | device | session | 0.902 | 0.750 | 0.588 | 1.730 | 1.664 |
| v11b | qvf | clip | 0.823 | 0.275 | 0.189 | 1.914 | 2.350 |
| v11b | qvf | session | 0.529 | 0.381 | 0.276 | 1.749 | 2.018 |
