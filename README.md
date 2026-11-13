<p align="center">
  <img src="https://raw.githubusercontent.com/hydrogeophysicsgroup/anemone/main/docs/images/logo.png" alt="anEMone logo" width="200">
</p>

# anEMone

[![PyPI](https://img.shields.io/pypi/v/anemone-core.svg)](https://pypi.org/project/anemone-core/)
[![Python](https://img.shields.io/pypi/pyversions/anemone-core.svg)](https://pypi.org/project/anemone-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/hydrogeophysicsgroup/anemone/blob/main/LICENSE)

**Fast, readable 1D transient electromagnetic (TEM) forward modelling in Python, built on PyTorch.**

Anemone simulates the gated dB/dt response of layered-earth models for real TEM systems. After a few lines of setup for your instrument, a forward call looks like this:

```python
dbdt = fwr(resistivity, thickness)
```

The same code runs on CPUs and GPUs, and evaluates thousands of models in a single batched call.

## Why anemone?

- **Fast.** Anemone uses digital-filter Hankel and Fourier transforms, geometric operators for the loop integration, and a matrix form of the waveform convolution and gating. Combined with batched PyTorch tensors, it runs about 30× faster than AarhusInv on the CPU for large batches, with a further 2–3× on a GPU.
- **Realistic systems.** It supports arbitrary horizontal polygonal or circular transmitter loops, finite receiver areas, offset and elevated receivers, piecewise-linear waveforms, Butterworth receiver filters, gating, multi-moment systems, and import of AarhusInv `.gex` system files.
- **Induced polarisation.** It includes Cole-Cole and Maximum Phase Angle (MPA) parameterisations for frequency-dependent conductivity.
- **Accurate.** Responses agree with analytic halfspace solutions to within 3% (mostly below 1%) for times from 1 ns to 1 s, and with AarhusInv for realistic ground-based and airborne systems.
- **Readable.** The code follows the mathematics of the forward problem, and docstrings cite the corresponding equations in Ward & Hohmann (1988), so it also works as a teaching resource.

## Installation

```console
pip install anemone-core
```

The package is published on PyPI as `anemone-core` but is imported as `anemone`:

```python
import anemone
```

Python 3.9 or newer is required. If you want GPU support, install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) **before** installing anemone. Otherwise pip installs the default (usually CPU-only) build.

To install from source for development:

```console
git clone https://github.com/hydrogeophysicsgroup/anemone.git
cd anemone
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from anemone.system import Loop, Receiver, Waveform
from anemone.forwards import Forward

tx = Loop.square(40)              # 40 m x 40 m transmitter loop at z = 0
rx = Receiver()                   # dipole receiver at the origin (central loop)
wv = Waveform.example()           # trapezoidal example waveform

t = torch.logspace(-5, -3, 32)    # times [s] at which dB/dt is evaluated

fwr = Forward(tx, rx, t, waveform=wv)

rho   = torch.tensor([40., 80., 40.])   # layer resistivities [Ohm m]
thick = torch.tensor([20., 30.])        # layer thicknesses [m]; the last layer is a halfspace

dbdt = fwr(rho, thick)                  # shape: (32,)
```

Setting up a `Forward` does the expensive work once: it trims filter coefficients, builds geometric operators, and assembles waveform and gating matrices. Every later call is cheap.

### Many models at once

Layers go along the first dimension and models along the second. One call evaluates the whole batch:

```python
n_models = 1000
rho   = 10**(0.5 + 2*torch.rand(3, n_models))   # (n_layers, n_models)
thick = torch.full((2, n_models), 20.)          # (n_layers - 1, n_models)

dbdt = fwr(rho, thick)                          # shape: (n_models, n_times)
```

### Realistic and dual-moment systems

Define the geometry, waveforms and gates for each moment, then add the forwards together:

```python
tx = Loop.square(3, z=0.97)        # elevated 3 m x 3 m transmitter
rx = Receiver(x=-9, z=0.43)        # offset receiver (tTEM-like geometry)

lm_fwr = Forward(tx, rx, lm_gate_times, waveform=Waveform(lm_wv_time, lm_wv_amp))
hm_fwr = Forward(tx, rx, hm_gate_times, waveform=Waveform(hm_wv_time, hm_wv_amp))

dual_fwr = lm_fwr + hm_fwr
dbdt = dual_fwr(rho, thick)
```

Receiver filters are passed with `filterfunc=` (for example `ButterworthFilter` or `FilterChain` from `anemone.system`), and gate start and end times with `times=` and `t_end=`. You can also build the whole system from an AarhusInv `.gex` file:

```python
dual_fwr = Forward.from_gex("examples/TX11_3x3_1_turn.gex")
```

### Induced polarisation

```python
t   = torch.logspace(-6, -2, 62)
fwr = Forward.colecole(Loop.square(40), Receiver(), t)

rho = torch.tensor([200., 80., 300.])
m   = torch.tensor([0.6, 0.6, 0.6])     # chargeability
tau = torch.tensor([1e-2, 1e-2, 1e-2])  # time constant [s]
C   = torch.tensor([0.6, 0.6, 0.6])     # frequency exponent
thk = torch.tensor([20., 20.])

dbdt = fwr(rho, m, tau, C, thk)
```

`Forward.mpa(...)` works the same way, with `(rho, phi_max, tau_phi, C, thickness)`.

### Running on a GPU

`Forward` uses the device of the `times` tensor you give it. To run on a GPU, create the times there and pass your model tensors on the same device:

```python
device = "cuda"
t   = torch.logspace(-5, -3, 32, device=device)
fwr = Forward(tx, rx, t, waveform=wv)
dbdt = fwr(rho.to(device), thick.to(device))
```

The GPU pays off for large batches (hundreds of models or more). For small batches the CPU is usually faster.

## Package layout

| Module | Contents |
|---|---|
| `anemone.physics` | Wavenumbers, TE reflection coefficients and their derivatives, Cole-Cole and MPA conductivity models |
| `anemone.transforms` | Hankel transform, (inverse) Fourier transform, inverse Laplace transform (Gaver-Stehfest), cubic spline integration |
| `anemone.system` | `Loop`, `Receiver`, `Waveform`, `ButterworthFilter`, `FilterChain`, plus waveform and gating matrices |
| `anemone.forwards` | `Forward` (fast, for production), `BaseForward` (a slow version that mirrors the equations, for learning), plus IP and dual-moment variants |
| `anemone.inversion` | Inversion models and `Inversion`, `LCI` and `SCI` inversion classes |

Functions in `physics` broadcast over their inputs, so they can be written much like their mathematical form:

```python
from anemone.physics import calc_wavenumber

sigma = torch.logspace(0, 3, 100)
omega = torch.logspace(-6, 6, 601)
k = calc_wavenumber(sigma[:, None], omega[None, :])   # shape: (100, 601)
```

## Examples

The [`examples/`](https://github.com/hydrogeophysicsgroup/anemone/tree/main/examples) folder contains Jupyter notebooks:

- `compare_TEM_forwards_with_analytic_responses.ipynb`: validation against analytic halfspace solutions
- `sensitivity_analysis_for_tTEM.ipynb`: sensitivity analysis for a tTEM-style system
- `import_usf_and_invert_it.ipynb`: loading field data and inverting it

## Testing

Every module has a test suite. Many of the tests compare numerical results against analytic solutions:

```console
pip install -e ".[dev]"
pytest tests
```

## Documentation

The documentation is built from the markdown files in `docs/` with MkDocs:

```console
pip install -e ".[docs]"
mkdocs serve      # preview locally
mkdocs build      # build the static site
```

To render curated Jupyter notebooks with `mkdocs-jupyter`, place them under `docs/` and add them to `mkdocs.yml`.

## Citing

A paper describing Anemone is under review. Until it is published, please cite this conference abstract: 
> Griffiths, M.P., E. Möller, I. Vela, A.V. Christiansen and D. Grombacher (2025) anEMone: A Python package for parallelized 1-dimensional transient electromagnetic forward modelling. _EAGE Near-surface Geoscience Conference_, Naples, Italy

The waveform and gating formulation is described in:

> Griffiths, M. P. (2026). Convolution-free electromagnetic system response in the time domain. *Geophysics*, 91(1), A1–A5.


## Key references

- Ward, S. H., & Hohmann, G. W. (1988). Electromagnetic theory for geophysical applications. In *Electromagnetic Methods in Applied Geophysics, Vol. 1*, 130–311. [doi:10.1190/1.9781560802631.ch4](https://doi.org/10.1190/1.9781560802631.ch4)
- Johansen, H. K., & Sørensen, K. (1979). Fast Hankel transforms. *Geophysical Prospecting*, 27(4), 876–901. [doi:10.1111/j.1365-2478.1979.tb01005.x](https://doi.org/10.1111/j.1365-2478.1979.tb01005.x)
- Christensen, N. B. (1990). Optimized fast Hankel transform filters. *Geophysical Prospecting*, 38(5), 545–568. [doi:10.1111/j.1365-2478.1990.tb01861.x](https://doi.org/10.1111/j.1365-2478.1990.tb01861.x)
- Boerner, D. E., & West, G. F. (1989). A generalized representation of the electromagnetic fields in a layered earth. *Geophysical Journal International*, 97(3), 529–547.
- Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace transforms. *Communications of the ACM*, 13(1), 47–49. [doi:10.1145/361953.361969](https://doi.org/10.1145/361953.361969)

## License

Anemone is released under the [MIT License](https://github.com/hydrogeophysicsgroup/anemone/blob/main/LICENSE).
