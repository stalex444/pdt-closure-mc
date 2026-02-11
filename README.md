# PDT Closure Monte Carlo — Statistical Validation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stalex444/pdt-closure-mc/blob/main/notebooks/pdt_statistical_tests.ipynb)

**Four-layer GPU-accelerated Monte Carlo test of Pisot Dimensional Theory**

This repository contains the statistical validation suite for the framework described in:

> S. Alexander, "There Is No Hierarchy," submitted to the Gravity Research Foundation 2026 Awards for Essays on Gravitation.

The code tests whether the observed agreement between PDT's predictions and 18 experimentally measured quantities could arise by chance.

## Results Summary

| Layer | Test | Trials | Result |
|-------|------|--------|--------|
| **Baseline** | PDT predictions vs. experiment | — | 18/18 within 3%, mean error 0.297% |
| **Layer 1** | Random (r,q) with PDT formulas | 2,000,000,000 | All ≥14 matches cluster near (ρ,Q); island fraction ~10⁻⁴ |
| **Layer 2** | Exclude polynomial neighborhood | 500,000,000 | Best: 11/18 (mean error 6.9%). Zero trials ≥12. p < 6×10⁻⁹ (>5.7σ) |
| **Layer 3** | Random exponents at fixed (ρ,Q) | 125,000 exhaustive + 1M random | Only 1 triple of 125,000 hits 3 key targets: PDT's (15,29,19). Random best: 16/18 |
| **Layer 4** | Permutation (formula↔observable) | 1,000,000 | Best shuffled: 10/18 vs. PDT 18/18. Gap: 8 predictions |

**Bottom line:** Outside a ~10⁻⁴ neighborhood of the roots of x³ = x + 1 and x⁴ = x + 1, no random pair (r,q) in 500 million trials matched more than 11 of 18 predictions. The exponent set and formula-to-observable mapping are each independently unique.

## Quick Start

### Google Colab (recommended)
!pip install cupy-cuda12x
%run pdt_closure_mc_gpu_v3.py
Runtime: ~3 minutes on A100, ~10 minutes on T4.

### Local (CPU fallback)
```bash
pip install numpy scipy
python pdt_closure_mc_gpu_v3.py
```
Runtime: ~2–4 hours without GPU.

### Quick verification mode
Set `QUICK_MODE = True` at line 36 of the script for a ~2 minute verification run (10M trials) that confirms the structure works before committing to the full 2.5B run.

## Files

| File | Description |
|------|-------------|
| `notebooks/pdt_statistical_tests.ipynb` | Colab notebook (click badge above to run) |
| `pdt_closure_mc_gpu_v3.py` | Main validation script (GPU/CPU) |
| `mc_results_summary_v3.txt` | Full output from 2.5B-trial production run |
| `colab_output.txt` | Complete Colab console output |
| `TARGETS.md` | Experimental targets with sources |
| `requirements.txt` | Python dependencies |
| `LICENSE` | CC-BY 4.0 |

## Experimental Targets

All predictions are compared against published experimental values:

| Observable | Target | Source |
|-----------|--------|--------|
| α⁻¹ | 137.035999177 | CODATA 2022 (Rev. Mod. Phys. 2024) |
| sin²θ_W | 0.23122 | PDG 2024 (MS-bar, M_Z) |
| α_s(M_Z) | 0.1180 | PDG 2024 world average |
| Y_p | 0.2449 | Aver et al. 2021 |
| n_s | 0.9649 | Planck 2018 TT,TE,EE+lowE |
| m_τ/m_e | 3477.23 | PDG 2024 |
| m_μ/m_e | 206.7682830 | CODATA 2022 |
| Tsirelson bound | 2√2 | Exact (Cirel'son 1980) |
| \|γ_halo\| | 0.82 | SPARC median |
| He/H | 0.3252 | BBN + CMB |
| \|V_us\| | 0.22500 | PDG 2024 Cabibbo |
| r (tensor-to-scalar) | 0.033 | PDT prediction (testable by BICEP Array / Simons Observatory) |
| sin²θ₂₃ | 0.546 | PDG 2024 (NO) |
| sin²θ₁₂ | 0.307 | PDG 2024 |
| sin²θ₁₃ | 0.02200 | PDG 2024 |
| H₀ ratio | 1.0831 | SH0ES / Planck tension |
| S₈ ratio | 0.919 | DES Y3 / Planck |
| log₁₀(α/α_G) | 42.620 | CODATA 2022 |

## Exponent Provenance

Every integer exponent in the framework corresponds to a group-theoretic dimension, not a free parameter:

| Exponent | Value | Origin |
|----------|-------|--------|
| e₁ | 15 | dim SO(4,2), conformal group of 3+1 spacetime |
| e₆ | 29 | dim SU(2) × SO(4,2) (electroweak × conformal) |
| e₇ | 19 | dim SO(4,2) + rank SU(3) × dim(adjoint) |
| e₈ | 209 | 11 × 19, where 11 = dim SO(3,2) anti-de Sitter |
| ψ³ | 3 | rank of SU(3) color |
| λ₄/5 | 5 | dim SU(2) + rank SU(3) |

12 of 18 predictions are exact algebraic functions of λ₃ = 1−1/ρ, λ₄ = 1−1/Q, ψ = Q/ρ with no exponents to adjust.

## Reproducibility

The script uses fixed random seeds (42, 137, 99, 2026) for full reproducibility. Results are deterministic given the same hardware and library versions. Minor floating-point differences between GPU and CPU are expected but do not affect match counts.

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- CuPy (optional, for GPU acceleration)

## Citation

If you use this code, please cite:
S. Alexander, "There Is No Hierarchy,"
Gravity Research Foundation Essay Competition 2026.

## License

This work is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
