# 🔭 RomanPower: Power Spectrum Estimation for Roman

**`RomanPower`** provides **auto** and **cross power spectrum** (and bispectrum) estimation for the
**Nancy Grace Roman Space Telescope** High Latitude Spectroscopic Survey.

It is built on top of [`jax-power`](https://github.com/adematti/jax-power) by *Arnaud de Mattia*, and inherits
its full feature set: periodic boxes and survey geometries, global and local lines of sight, window matrices
and covariances, and distributed multi-GPU computation with JAX.

> **Relation to `jax-power`**
> This repository started as a fork of `jax-power`; the Python package is renamed `jaxpower` → `romanpower`,
> so imports read `from romanpower import ...`. Everything else — the API, the module layout, the
> conventions — follows upstream, and upstream fixes can be ported over directly.

---

## 📦 Installation

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/willlake/RomanPower.git
```

Or, for development:

```bash
git clone https://github.com/willlake/RomanPower.git
cd RomanPower
pip install -e .
```

### Requirements

- Python ≥ 3.10
- `jax`, `jaxlib` (with GPU or TPU support, if applicable)
- `numpy`
- [`jaxdecomp`](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) — for distributed FFT and halo exchange
- [`lsstypes`](https://github.com/adematti/lsstypes) — observable data structures

We recommend following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
to ensure correct setup for your hardware (CPU/GPU/TPU).

---

## 🧪 Quick Example: Auto Power Spectrum with Multi-GPU

```python
import jax
# Initialize JAX distributed environment
jax.distributed.initialize()

from jax import numpy as jnp
from romanpower import (
    get_mesh_attrs,
    compute_mesh2_spectrum,
    ParticleField,
    FKPField,
    create_sharding_mesh,
    BinMesh2SpectrumPoles,
    compute_fkp2_normalization,
    compute_fkp2_shotnoise
)

with create_sharding_mesh() as sharding_mesh:  # distribute mesh and particles

    # Create MeshAttrs from positions (assumed already sharded across processes)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=2., meshsize=128)

    # Define FKP field = data - randoms
    data = ParticleField(data_positions, data_weights, attrs=mattrs, exchange=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=mattrs, exchange=True)
    fkp = FKPField(data, randoms)
    # Warning!
    # If you access data.positions, data.weights (same for randoms) in a distributed context,
    # You'll see they are reordered w.r.t. data_positions, data_weights
    # This is such that the local portion of the 3D mesh receives the relevant particles

    # Define k-bin edges and multipoles
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=(0, 2, 4))

    # Compute normalization and shot noise terms
    norm = compute_fkp2_normalization(fkp, bin=bin)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)

    # Paint FKP field to mesh
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    del fkp  # cleanup

    # JIT the power spectrum function
    compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])

    # Compute P(k)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

    # Save result
    spectrum.write('spectrum.h5')

# Shut down distributed environment
jax.distributed.shutdown()
```

📝 Example notebooks are available in the `nb/` directory, and tests in `tests/`
(each test file runs as a plain script via its `__main__` block).

---

## 📚 Citation

If you use `RomanPower`, please cite the upstream package it is based on:

> **`jax-power`** — [https://github.com/adematti/jax-power](https://github.com/adematti/jax-power)
> Developed by *Arnaud de Mattia*.

Multi-GPU 3D FFT and halo exchange support is provided by:

> **`jaxdecomp`** — [https://github.com/DifferentiableUniverseInitiative/jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)
> Developed by *Wassim Kabalan* and *François Lanusse*.
> 📄 *Publication incoming!*

---

## 🙏 Acknowledgments

`RomanPower` is a derivative of [`jax-power`](https://github.com/adematti/jax-power) — thanks to
**Arnaud de Mattia** for the original package.

---
