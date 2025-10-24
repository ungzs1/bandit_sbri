# Code Repository for *Neural Dynamics Reveal Foraging-like Computations in the Frontal Cortex*

This repository contains the code used to reproduce the figures and analyses presented in the manuscript *“Neural dynamics reveal foraging-like computations in the frontal cortex.”* It includes preprocessing, modeling, and visualization scripts as well as example data to demonstrate the workflow.

The provided source code was used to generate results in the manuscript. The example dataset included in this repository is a **small subset** of the full dataset used in the manuscript. Therefore, results are not exactly matching those in the manuscript.


---

## 1. System Requirements

The code was developed and tested on **macOS Sonoma (Version 14.5)**, but it should run on any major operating system (macOS, Linux, or Windows) with Python ≥ 3.9 and the required packages installed.

All software dependencies are specified in the [`environment.yml`](environment.yml) file. These can be installed automatically using [Conda](https://docs.conda.io/) or manually based on the package list.

---

## 2. Installation Guide

### Option 1 — Using Conda (Recommended)

1. Clone this repository:

   ```bash
   git clone https://github.com/ungzs1/bandit_sbri.git
   cd bandit_sbri
   ```

2. Create and activate a virtual environment from the provided file:

   ```bash
   conda env create -f environment.yml
   conda activate popy_published
   ```

   Installation typically takes a few minutes.

3. Install the local `popy` package in editable mode:

   ```bash
   pip install -e .
   ```

   This installs the repository’s internal Python package (`popy/`), allowing all notebooks and scripts to import it directly from anywhere in the project (e.g., from the `demos` or `analysis` folders). After this step, modules can be imported normally, for example:
   
   ```python
   import popy
   ```

4. Download demo neural data.
   
   Demo neural data containing 4 sessions recordings data can be downloaded from the following link: [Data Download Link](https://sdrive.cnrs.fr/s/yBL5aYG94G8KKsx) as a zip file. Data needs to be extracted to the `data/processed/neural_data/` folder.

### Option 2 — Manual Setup

You can also install the dependencies manually using `pip` or another virtual environment manager. However, only the Conda setup has been tested and is guaranteed to reproduce the environment used in the manuscript.

---

## 3. Repository Structure

```
bandit_sbri/
│
├── data/
│   ├── processed/       # Example preprocessed sessions, ready to use
│   │   ├── behavior/    # Folder for demo behavioral data
│   │   └── neural_data/ # Folder for demo neural data (**extract neural data files here**)
│   └── results/         # Folder for saving analysis outputs
│
├── popy/                # Core package with model, analysis, and plotting functions
│   ├── config.py        # **Set your local data and results paths here**
│   └── etc...           # Other analysis scripts
│
├── analysis/            # Scripts for computationally intensive analyses
│
├── demos/               # Jupyter notebooks for reproducing manuscript figures
│
├── environment.yml      # Environment specification file
├── setup.py             # Installer for the popy package
└── README.md            # This file
```

---

## 4. Data structure

The provided demo dataset contains behavioral data from **two monkeys**, each recorded across **two experimental sessions** (four sessions total). 

### Behavioral data

Behavioral data is stored in a pandas DataFrame, located `data/processed/behavior/behavior_kapo.pkl`. Each row corresponds to a single trial, containing information about session context, choices, outcomes, and switching behavior.

The DataFrame has the following attributes:

| Column          | Description                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **monkey**      | Identifier for the monkey (e.g., `"ka"`, `"po"`).                                                                                           |
| **session**     | Session identifier (e.g., `"020622"`). Each monkey has two sessions.                                                                        |
| **trial_id**    | Sequential trial number within a session.                                                                                                   |
| **block_id**    | Identifier for experimental block within a session.                                                                                         |
| **best_target** | Target with high payoff for that trial/block.                                                                                  |
| **target**      | Target chosen by the monkey during the trial.                                                                                               |
| **feedback**    | Feedback or reward indicator (e.g., `1.0` = rewarded, `0.0` = not rewarded).                                                                |
| **switch**      | Indicates if the monkey switched choices compared to the previous trial (e.g., `1.0` = switch, `0.0` = stay). May be `NaN` on first trials. |

Demo behavioral data file is included in the repository.

### Neural data

Neural data is stored in individual files for each session, located in `data/processed/neural_data/<MONKEY>_<SESSION>_spikes.nc`. The data are stored in NetCDF format (.nc), structured as an ```xarray``` ```Dataset``` containing spiking activity aligned to task events (in short, xarray is a Python library for working with labeled numpy arrays). The main data array contains binary spike trains (1 = spike, 0 = no spike) for multiple recorded units over time. 

The Dataset has the following structure:

| Component       | Description                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| **Dimensions**  | (`unit`, `time`)                                                                                              |
| **Variables**   | `__xarray_dataarray_variable__` — main spike data array (binary, shape `[unit, time]`) |
| **Coordinates** | Metadata aligned with each `unit` or `time` dimension                                                       |

The coordinates provide additional context for each dimension:

| Coordinate           | Dimension | Description                                                 |
| -------------------- | --------- | ----------------------------------------------------------- |
| **unit**             | `unit`    | Index of recorded neural units.                            |
| **time**             | `time`    | Continuous time points of recording.                       |
| **unit_id_original** | `unit`    | Original identifier for the unit from acquisition software. |
| **channel**          | `unit`    | Electrode channel number associated with the unit.          |
| **monkey**           | `unit`    | Monkey identifier (e.g., `"po"`).                           |
| **session**          | `unit`    | Session identifier (e.g., `"240921"`).                      |
| **area**             | `unit`    | Brain area of the recording site (e.g., MCC or LPFC).       |
| **subregion**        | `unit`    | Subdivision within the recorded area (e.g., vLPFC or dLPFC).|
| **trial_id**         | `time`    | Behavioral trial associated with each time segment.         |
| **epoch_id**         | `time`    | Epoch in the trial    |

For inspection, the neural data can be loaded as follows:

```python
import xarray as xr

ds = xr.load_dataset('/<YOUR>/<PATH>/bandit_sbri/data/processed/neural_data/ka_210322_spikes.nc')
print(ds)
```

Demo neural data files are available for download as described in the Installation Guide.

## 5. Instructions for Use

### Generating Figures

Figures from the manuscript can be reproduced using the notebooks in the `demos` directory.
Each notebook includes:

* A **Summary** cell at the top, describing the purpose of the notebook and which manuscript figure it corresponds to.
* A **Setup** section with imports and helper functions (these can be collapsed for readability).
* The basic data processing and figure generation section, which generates figure panels.

> Some notebooks require running specific analysis scripts beforehand. These dependencies are clearly indicated in the Summary cell at the beginning of each notebook.

### Running Analyses

Longer analyses are implemented as standalone Python scripts in the `analysis` folder.
To execute a specific analysis:

```bash
python analysis/model_optimizing.py
python analysis/model_fitting.py
python analysis/glm.py
python analysis/time_resolved_decoding.py
python analysis/neural_value_extraction.py
python analysis/neural_value_target_dependence.py
```

Results will be saved automatically in `data/results/`, from where the corresponding notebook reads it.

### Expected Runtime

* All figure notebooks should execute within tens of seconds.
* Analysis scripts may take longer (typically from a few minutes to several hours).

---

## 6. Citation

If you use this code or reproduce analyses in your own work, please cite the following:

> *Ungvarszki, Z., Goussi-Denjean, C., Szekely, A., Orban, G., Di Volo, M. and Procyk, E., 2025. Neural dynamics reveal foraging-like computations in the frontal cortex. bioRxiv, pp.2025-09.*

---

## 7. Contact

For questions, bug reports, or requests for the full dataset, please contact:
**Zsombor Ungvaszki**
Email: ungvaszki.zsombor [at] gmail [dot] com
Institution: Inserm-SBRI, Lyon, France