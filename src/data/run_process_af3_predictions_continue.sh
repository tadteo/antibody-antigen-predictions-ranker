#! /bin/bash

# setting up the environment
module load Mambaforge
mamba activate abag

# setting up working directory
cd /proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/src/data/

# running the script
python process_af3_predictions.py \
    --metadata /proj/berzelius-2021-29/Database/simple_sabdab_db/training/training_metadata.csv \
    --af3_folder /proj/berzelius-2021-29/users/x_malud/alphafold3/matteo/models-training/ \
    --reference_folder /proj/berzelius-2021-29/Database/simple_sabdab_db/training/pdb \
    --resume \
    --output /proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/20250806_151257
