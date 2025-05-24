Copyright © <2024> The Regents of the University of Michigan

## GeoDGP Overview

Accurately predicting the horizontal component of ground magnetic field perturbations is crucial for the risk management of natural hazards caused by space weather events, such as geomagnetically induced currents (GICs). **GeoDGP** is a data-driven model that probabilistically forecasts local geomagnetic perturbations over the globe at 1-minute cadence and at most 1-hour ahead. The model is trained on 30 years of (1995-2022) NASA/GSFC’s [OMNI dataset](https://omniweb.gsfc.nasa.gov/) and [SuperMAG](https://supermag.jhuapl.edu/) ground magnetometer measurements.

![example](docs/example.png)

Check out our [real-time global prediction](https://csem.engin.umich.edu/GeoDGP/).


## Features

### Input

- Solar wind measurements from the first Lagrangian point (L1).
- The location input.

| Variable Name | Description |
| :----: | :---- |
| $B_x, B_y, B_z$ | Interplanetary Magnetic Field (IMF) |
| $V_x$ | Solar wind velocity |
| $N_p$ | Proton number density |
| $T$ | Plasma temperature |
| Dst | Disturbance storm time index |
| $\theta$ | Dipole tilt angle |
| $\lambda$ | Geomagnetic latitude |
| $\phi$ | Geomagnetic longitude in Solar Magnetic (SM) coordinates |

### Output

- The north, east, and horizontal components of geomagnetic perturbations ($dB_H$, $dB_N$, and $dB_E$) with a lead time corresponding to 1 hour plus the solar wind propagation time from L1.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HongfanChen/GeoDGP.git
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

## Data Access

To reproduce the results in the original paper, please refer to the dataset available at the [Deep Blue Data repository](https://doi.org/10.7302/6brp-0y03), and find the permanantly archived release of code at [![DOI](https://zenodo.org/badge/928522177.svg)](https://doi.org/10.5281/zenodo.15505660).

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0 License - see the [LICENSE](LICENSE.txt) file for details. If you are interested in using the code for commercial purposes, please reach out to Hongfan Chen (chenhf@umich.edu).

## Citation

Hongfan Chen, Gabor Toth, Yang Chen, et al. <i>GeoDGP: One-Hour Ahead Global Probabilistic Geomagnetic Perturbation Forecasting using Deep Gaussian Process.</i> ESS Open Archive. December 23, 2024. DOI: <a href=“https://doi.org/10.22541/essoar.173499121.15272711/v1” target=“_blank”> 10.22541/essoar.173499121.15272711/v1</a>

Chen, H., Chen, Y., Huang, Z., Zou, S., Huan, X., Toth, G. <i>Data and Analysis for Global Probabilistic Geomagnetic Perturbation Forecasting Using the Data-Driven Model GeoDGP</i> [Data set], University of Michigan - Deep Blue Data. https://doi.org/10.7302/6brp-0y03

## Credits

Hongfan Chen: Conceptualization, Data Curation, Methodology, Formal Analysis, Software.

Gabor Toth: Conceptualization, Supervision, Project Administration.

Yang Chen: Methodology, Supervision.

Shasha Zou: Formal Analysis, Supervision.

Zhenguang Huang: Methodology.

Xun Huan: Methodology, Supervision.

Aniket Jivani: Website Development.

## Contacts

Please contact Hongfan Chen (chenhf@umich.edu) for any questions.

## Acknowledgement

This work is supported by the National Science Foundation (NSF) under Grant No. 2027555. NextGen Space Weather Modeling Framework Using Data, Physics and Uncertainty Quantification. We acknowledge use of NASA/GSFC’s Space Physics Data Facility’s OMNIWeb and CDAWeb service, and OMNI data. We gratefully acknowledge the SuperMAG collaborators. The Dst data are provided by the WDC for Geomagnetism, Kyoto.