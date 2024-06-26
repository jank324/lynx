<img src="images/logo.png" align="right" width="25%"/>

![format](https://github.com/jank324/lynx/actions/workflows/format.yaml/badge.svg)
![pytest](https://github.com/jank324/lynx/actions/workflows/pytest.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/lynx-accelerator/badge/?version=latest)](https://lynx-accelerator.readthedocs.io/en/latest/?badge=latest)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!-- [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) -->

# Lynx

**🚧 This repository is currently under construction. Do not expect this code to work yet. I recommend using _Cheetah_ for now. 🚧**

Lynx is a JAX port of [Cheetah](https://github.com/desy-ml/cheetah).
Cheetah is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

## Installation

Simply install _Lynx_ from PyPI by running the following command.

```bash
pip install lynx-accelerator
```

## How To Use

A sequence of accelerator elements (or a lattice) is called a `Segment` in _Lynx_. You can create a `Segment` as follows

```python
import jax.numpy as jnp
from lynx import BPM, Drift, HorizontalCorrector, Segment, VerticalCorrector

segment = Segment(
    elements=[
        BPM(name="BPM1SMATCH"),
        Drift(length=jnp.array([1.0])),
        BPM(name="BPM6SMATCH"),
        Drift(length=jnp.array([1.0])),
        VerticalCorrector(length=jnp.array([0.3]), name="V7SMATCH"),
        Drift(length=jnp.array([0.2])),
        HorizontalCorrector(length=jnp.array([0.3]), name="H10SMATCH"),
        Drift(length=jnp.array([7.0])),
        HorizontalCorrector(length=jnp.array([0.3]), name="H12SMATCH"),
        Drift(length=jnp.array([0.05])),
        BPM(name="BPM13SMATCH"),
    ]
)
```

Alternatively you can create a segment from an Ocelot cell by running

```python
segment = Segment.from_ocelot(cell)
```

All elements can be accesses as a property of the segment via their name. The strength of a quadrupole named _AREAMQZM2_ for example, may be set by running

```python
segment.AREAMQZM2.k1 = jnp.array([4.2])
```

In order to track a beam through the segment, simply call the segment like so

```python
outgoing_beam = segment.track(incoming_beam)
```

You can choose to track either a beam defined by its parameters (fast) or by its particles (precise). _Lynx_ defines two different beam classes for this purpose and beams may be created by

```python
beam1 = ParameterBeam.from_parameters()
beam2 = ParticleBeam.from_parameters()
```

It is also possible to load beams from Ocelot `ParticleArray` or Astra particle distribution files for both types of beam

```python
ocelot_beam = ParticleBeam.from_ocelot(parray)
astra_beam = ParticleBeam.from_astra(filepath)
```

You may plot a segment with reference particle traces bay calling

```python
segment.plot_overview(beam=beam)
```

![Overview Plot](images/misalignment.png)

where the optional keyword argument `beam` is the incoming beam represented by the reference particles. Lynx will use a default incoming beam, if no beam is passed.

## Cite Lynx

If you use Lynx, please cite the following two papers:

```bibtex
@article{kaiser2024cheetah,
    title        = {Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations},
    author       = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = {May},
    journal      = {Phys. Rev. Accel. Beams},
    publisher    = {American Physical Society},
    volume       = 27,
    pages        = {054601},
    doi          = {10.1103/PhysRevAccelBeams.27.054601},
    url          = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.27.054601},
    issue        = 5,
    numpages     = 17
}
@inproceedings{stein2022accelerating,
    title        = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
    author       = {Stein, Oliver and Kaiser, Jan and Eichler, Annika},
    year         = 2022,
    booktitle    = {Proceedings of the 13th International Particle Accelerator Conference}
}
```

## For Developers

Activate your virtual environment. (Optional)

Install the Lynx package as editable

```sh
pip install -e .
```

We suggest installing pre-commit hooks to automatically conform with the code formatting in commits:

```sh
pip install pre-commit
pre-commit install
```

## Acknowledgements

We acknowledge the contributions of the following people to the development of Lynx and Cheetah: Jan Kaiser, Chenran Xu, Oliver Stein, Annika Eichler, Andrea Santamaria Garcia and others.

The work to develop Lynx and Cheetah has in part been funded by the IVF project InternLabs-0011 (HIR3X) and the Initiative and Networking Fund by the Helmholtz Association (Autonomous Accelerator, ZT-I-PF-5-6).
In addition, we acknowledge support from DESY (Hamburg, Germany) and KIT (Karlsruhe, Germany), members of the Helmholtz Association HGF.
