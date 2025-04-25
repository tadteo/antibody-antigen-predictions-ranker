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

## Dataset info
### Sample Weighting (Importance Sampling + DockQ Balancing)

When we train, we use a **Per‐Complex Sampler** that draws exactly \(M\) models _per complex_ each epoch.  That introduces a bias in the sampling probability \(p_{ij}\) of sample \(j\) from complex \(i\).  We correct for two things:


C = number of complexes

N_i = total number of samples in complex 

M = samples_per_complex

1. **Complex‐level over/under–sampling**  
   - Let \(N_i\) = number of models for complex \(i\).  
   - Because we pick exactly \(M\) from each complex:
     \[
       p_{ij}\;\propto\;\frac{1}{N_i}\quad\Longrightarrow\quad
       w_{ij}^{\rm complex}\;\propto\;\frac{1}{p_{ij}}\;\propto\;\frac{N_i}{M}.
     \]

2. **Global DockQ–bin balancing**  
   - Each sample falls into one of four global DockQ buckets \(b\in\{0,1,2,3\}\).  
   - Let \(n_b\) = total samples in bucket \(b\).  
   - We up‐weight rare buckets by
     \[
       w_{ij}^{\rm bucket} \;=\;\frac{1}{n_{\,b_{ij}}}.
     \]

Putting it together, the **final per‐sample weight** is
\[
w_{ij}
\;=\;
\underbrace{\frac{N_i}{M}}_{\displaystyle w_{ij}^{\rm complex}}
\;\times\;
\underbrace{\frac{1}{n_{b_{ij}}}}_{\displaystyle w_{ij}^{\rm bucket}}
\;\quad(\text{optionally normalized so }\sum w_{ij}=1).
\]

```python
# M = samples_per_complex (e.g. 10)
# df is your manifest DataFrame with 'complex_id' and 'bucket' columns

# count models per complex & samples per bucket
complex_counts = df['complex_id'].value_counts().to_dict()
bucket_counts  = df['bucket'].value_counts().to_dict()

# complex‐level correction
df['weight_complex'] = df['complex_id'].map(lambda c: complex_counts[c] / M)
# bucket‐level correction
df['weight_bucket']  = df['bucket']    .map(lambda b: 1.0 / bucket_counts[b])

# final weight
df['weight'] = df['weight_complex'] * df['weight_bucket']

# (optional) normalize to sum to 1
df['weight'] /= df['weight'].sum()
```
