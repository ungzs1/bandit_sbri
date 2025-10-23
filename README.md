# Code Repository for *Neural Dynamics Reveal Foraging-like Computations in the Frontal Cortex*

This repository contains the code used to reproduce the figures and analyses presented in the manuscript *“Neural dynamics reveal foraging-like computations in the frontal cortex.”* It includes preprocessing, modeling, and visualization scripts as well as example data to demonstrate the workflow.

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
   
   Demo neural data containing 4 sessions recordings data can be downloaded from the following link: [Data Download Link](https://drive.google.com/drive/folders/1rYJ3Y9bX6K4v5X1a2b3c4d5e6f7g8h9i?usp=sharing) as a zip file. Data needs to be extracted to the `data/processed/neural_data/` folder.

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

## 4. Instructions for Use

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

* All figure notebooks should execute within **tens of seconds**.
* Analysis scripts may take longer (typically from a few minutes to several hours).

The provided source code was used to generate results in the manuscript. The example dataset included in this repository is a **small subset** of the full dataset used in the manuscript. Therefore, results are not exactly matching those in the manuscript.

---

## 5. Citation

If you use this code or reproduce analyses in your own work, please cite the following:

> *Ungvarszki, Z., Goussi-Denjean, C., Szekely, A., Orban, G., Di Volo, M. and Procyk, E., 2025. Neural dynamics reveal foraging-like computations in the frontal cortex. bioRxiv, pp.2025-09.*

---

## 6. Contact

For questions, bug reports, or requests for the full dataset, please contact:
**Zsombor Ungvaszki**
Email: ungvaszki.zsombor [at] gmail [dot] com
Institution: Inserm-SBRI, Lyon, France