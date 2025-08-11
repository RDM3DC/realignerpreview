## Quantum-Control Benchmark (MIMO-ARP vs Gaussian-DRAG)

**Settings:**
- T₁ = 50 μs, Tφ = 20 μs
- 1/f + white amplitude noise
- Static crosstalk = 5%

**Results:**
| Strategy | EPC (proxy) | XT | Amp-noise | Gate time |
|----------|--------------|----|-----------|-----------|
| Gaussian + DRAG | baseline | — | — | ~44.6 ns |
| Diagonal ARP (τ≈6 ns) | −12% | −61% | −12% | +~56% |
| MIMO-ARP (τ=2 ns, SPSA-tuned) | **−15.5%** | −33% | −15% | +~5% |

**Takeaway:** MIMO-ARP offers the best tradeoff—reducing EPC most effectively with minimal duration increase.

Benchmark data resides in `benchmark_gaussian_arp.csv`; generate plots from this data as needed.

_These are phenomenological pulse-level sims for quick iteration. Fixed-seed for reproducibility._
