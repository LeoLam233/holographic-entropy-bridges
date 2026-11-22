# Minimum-cut bridges and unavoidable cycles in holographic graph entropies

Dehao Lin, School of Physics, Sun Yat-sen University, Guangzhou, China.

[lindh9@mail2.sysu.edu.cn](mailto:lindh9@mail2.sysu.edu.cn)

The paper gives a bridge criterion for graph entropy vectors, proves the C1 and
C2 lifting assertions, and certifies that the six-party ray 110 has no finite
forest realization. Gluing copies of this ray yields extreme rays with
arbitrarily large minimum cycle rank. The manuscript relates its six-party
examples for C3 and C4 to [Ju and Czech's seven-party result](https://arxiv.org/abs/2611.21726v1).

- [Manuscript PDF](paper/C1234_bridge_manuscript.pdf)
- [LaTeX source](paper/C1234_bridge_manuscript.tex)
- [Version 1.0.0 release](https://github.com/LeoLam233/holographic-entropy-bridges/releases/tag/v1.0.0)

## Reproduce the checks

From the repository root, run the exact ray-110 checks with Python 3.10 or
later. The two scripts cover the finite forest-exclusion certificate, the
integer cut and rank checks, and the cycle-cover examples.

```sh
python -B reproduction/C3/verify_c3.py
python -B reproduction/C3/verify_ray110.py
```

Install the tested NumPy and SciPy versions for the bridge and C1 checks.
The bridge verifier checks all 4,096 two-block masks and the ranks in the
marked-port example. The C1/C2 scripts replay the split and fine-lifting
checks on the included Fletcher graphs and their built-in controls.

```sh
python -m pip install -r reproduction/requirements.txt
python -B reproduction/verify_bridge_examples.py
cd reproduction/C1_C2
python -B verify_quartic_c1.py
python -B verify_c2.py fletcher_graphs.json
```

The C1 script can also check rows 5, 55, 294 and 532 of the catalogue snapshot
described in the [IRSI note](https://irsi.ai/notes/moon-holo-lean/).
From `reproduction/C1_C2`, pass `--catalog /path/to/irsi_catalog_graphs.json`
if you have that snapshot. The SHA-256 of the replayed input is
`2d09da4fec8710d3a2b21f73825cb4d0ffe7374c30ffab8022b80b3cfa9c2b98`.

To rebuild the manuscript from `paper/`, run
`latexmk -pdf -interaction=nonstopmode -halt-on-error C1234_bridge_manuscript.tex`
with TeX Live. The figure is defined in the LaTeX source.

## Cite and reuse

See [CITATION.cff](CITATION.cff) for citation metadata. The original manuscript
and documentation use [CC BY 4.0](LICENSE-CONTENT.md). Original verification
code uses the [MIT license](LICENSE).

The ray-110, ray-145 and ray-168 graph inputs reproduce data from
[Austen M. Fletcher's deposit](https://doi.org/10.5281/zenodo.22020364).
The original [Fletcher MIT notice](reproduction/C1_C2/LICENSE-FLETCHER)
accompanies the graph data, including the witness transcribed in the ray-110
verifier.
