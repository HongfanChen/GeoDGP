# Overview

Accurately predicting the horizontal component of ground magnetic field perturbations is crucial for the risk management of natural hazards caused by space weather events, such as geomagnetically induced currents (GICs). **GeoDGP** is a data-driven model that probabilistically forecasts local geomagnetic perturbations over the globe at 1-minute cadence and at most 1-hour ahead. The model is trained on 30 years of (1995-2022) NASA/GSFC’s [OMNI dataset](https://omniweb.gsfc.nasa.gov/) and [SuperMAG](https://supermag.jhuapl.edu/) ground magnetometer measurements.

![example](docs/example.png)

Check out our [real-time global prediction](https://csem.engin.umich.edu/GeoDGP/).

## Structure

```
GeoDGP/
├── data/               # this can be downloaded from Deep Blue Data, see data section below
├── figure/             # this can be downloaded from Deep Blue Data, see data section below
├── docs/               # documents
├── scripts/            # scripts to reproduce results in the paper
├── src/                # Source code
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HongfanChen/GeoDGP.git
cd your-repo
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Features

### Input
- Solar wind measurements from the first Lagrangian point (L1).
- The location input.
| Variable Name | Description |
| :----: | :---- |
| $B_x, B_y, B_z$ | Interplanetary Magnetic Field (IMF) |
| $V_x$ | Solar wind velocity |
| $N_p$| Proton number density |
| $T$ | Plasma temperature |
| Dst | Disturbance storm time index |
| $\theta$ | Dipole tilt angle |
| $\lambda$ | Geomagnetic latitude |
| $\phi$ | Geomagnetic longitude in Solar Magnetic (SM) corrdinates |

### Output
- The north, east, and horizontal components of geomagnetic perturbations ($dBH$, $dBN$, and $dBE$) with a lead time corresponding to the solar wind propagation time from L1.

## Data

The data used in the paper is deposited in Deep Blue Data repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Hongfan Chen, Gabor Toth, Yang Chen, et al. <i>GeoDGP: One-Hour Ahead Global Probabilistic Geomagnetic Perturbation Forecasting using Deep Gaussian Process.</i> ESS Open Archive. December 23, 2024. DOI:
<a href=“https://doi.org/10.22541/essoar.173499121.15272711/v1” target=“_blank”> 10.22541/essoar.173499121.15272711/v1</a>

