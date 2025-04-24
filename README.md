# Antibody-Antigen Predictions Ranker

This project provides a framework for processing a set of h5 data files, training a deep set network to rank antibody-antigen interactions, and running jobs on SLURM.

Project Structure:

```
.
├── data/                # Raw and processed data
├── src/                 # Source code for data processing, model, and training
├── configs/             # Configuration files for experiments
├── scripts/             # Utility scripts to launch processing and training
├── slurm/               # SLURM job scripts
└── README.md            # Project overview
```

## Install on Berzelius Cluster

```bash
module load Mambaforge
mamba create -n abag python=3.11 pip
mamba activate abag
pip install -r requirements.txt
```


## Usage

### Local Execution

```bash
bash scripts/run_preprocess.sh
bash scripts/run_train.sh
```

### SLURM Execution

```bash
sbatch slurm/process_data.slurm
sbatch slurm/train.slurm
```
