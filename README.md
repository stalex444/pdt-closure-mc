# PDT Closure Monte Carlo

**Statistical validation of Pisot Dimensional Theory (PDT)**
*Companion code for the Gravity Research Foundation 2026 essay submission*

## What This Tests

PDT predicts 18 fundamental constants from two polynomial roots:
- **ρ** = 1.32472… (real root of x³ = x + 1)
- **Q** = 1.22074… (real root of x⁴ = x + 1)

The Monte Carlo asks: could any other pair of constants (r, q) match nature this well using the same formulas? The answer is no.

## Results Summary (v3.1)

| Layer | Test | Trials | Result |
|-------|------|--------|--------|
| **PDT** | Baseline | — | **18/18** within 3%, mean error **0.297%** |
| 1 | Random (r,q) | 2,000,000,000 | Best rival: 15/18 (near ρ,Q only) |
| 2 | Exclude island | 500,000,000 | Wall at 11/18 → **>5.7σ** |
| 3 | Random exponents | 125,000 exhaustive | Only (15,29,19) works = group dimensions |
| 4 | Permutation | 1,000,000 | Best shuffled: 9/18 |

**Empirical significance:** 0 of 2 billion random trials achieved ≥16/18 matches. p < 1.5 × 10⁻⁹ (>5.7σ).

## Quick Start

### Full analysis (GPU, ~3–5 min on A100)
```bash
# On Google Colab with GPU runtime:
!pip install cupy-cuda12x
%run pdt_closure_mc_gpu_v3.py
```

### Quick verification (CPU, ~30 sec)
```bash
python quick_verify.py
```

## Experimental Targets

All predictions are compared against the latest experimental values:

| Source | Values Used |
|--------|-------------|
| CODATA 2022 | α⁻¹, m_μ/m_e, log₁₀(α/α_G) |
| PDG 2024 | sin²θ_W, α_s(M_Z), m_τ/m_e, |V_us|, neutrino mixing angles |
| Aver et al. 2026 | Y_p (primordial helium) |
| Planck 2018 | n_s (spectral index) |
| SPARC | |γ_halo| (galaxy rotation) |

### v3.1 Update (February 2026)
- Y_p target updated: 0.2449 (Aver 2021) → **0.2458** (Aver 2026)
- α_s(M_Z) confirmed at 0.1180 (PDG 2024/2025)

## What the Four Layers Test

**Layer 1 — Uniqueness of (ρ, Q):** Draw 2 billion random pairs from [1.01, 2.50]². Plug them into PDT's 18 formulas. No pair outside a tiny neighborhood of (ρ, Q) achieves more than 11/18 matches.

**Layer 2 — No alternative islands:** Exclude the polynomial neighborhood (|r−ρ| < 0.02 and |q−Q| < 0.02) and repeat. The wall stays at 11/18. There is no second solution anywhere in parameter space.

**Layer 3 — Exponent uniqueness:** Fix (ρ, Q) and randomize the exponents. Only 1 of 125,000 integer triples simultaneously hits α⁻¹, m_τ/m_e, and m_μ/m_e — and that triple is PDT's (15, 29, 19), which are the dimensions of the conformal and electroweak symmetry groups. 12 of 18 predictions have no exponent freedom at all (pure algebraic functions of λ₃, λ₄, ψ).

**Layer 4 — Permutation test:** Keep the formulas and constants but randomly reassign which formula maps to which observable. In 1 million shuffles, no random assignment exceeds 9/18 matches. The mapping is unique.

### The Threshold Sweep
| Threshold | PDT | Wall (excl.) | Gap |
|-----------|-----|-------------|-----|
| 1% | 16 | 6 | **10** |
| 2% | 18 | 11 | 7 |
| 3% | 18 | 11 | 7 |
| 5% | 18 | 13 | 5 |
| 10% | 18 | 15 | 3 |

The gap **widens** under tighter scrutiny — the opposite of numerology.

## File Structure

```
pdt-closure-mc/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── pdt_closure_mc_gpu_v3.py      # Full 4-layer MC (v3.1, GPU/CPU)
├── quick_verify.py               # 30-second CPU verification
└── mc_results_summary_v3.txt     # Output from full run
```

## Reproducing the Results

The script auto-detects GPU/CPU and outputs `mc_results_summary_v3.txt`. Expected output on A100:

```
Layer 1: 2,000,000,000 trials → 18/18 only at (ρ, Q)
Layer 2: 500,000,000 trials (excl) → wall at 11/18, >5.7σ
Layer 3: 1/125,000 exponent triples → PDT's (15,29,19)
Layer 4: 1,000,000 shuffles → best 9/18
```

## Citation

```bibtex
@misc{alexander2026hierarchy,
  author = {Alexander, Stephanie},
  title = {There Is No Hierarchy: The Gauge Hierarchy as Dimensional Arithmetic},
  year = {2026},
  note = {Submitted to Gravity Research Foundation 2026 Awards for Essays on Gravitation}
}
```

## License

MIT License. Use freely. Verify independently.
