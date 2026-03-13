AI-based stability assessment of smart grid
========================================

Automated pipeline for generating datasets and training transformer models
to discover Lyapunov functions for power system

Based on: "Global Lyapunov functions: a long-standing open problem in
mathematics, with symbolic transformers" (NeurIPS 2024, Meta FAIR).

Automation scripts written by Anqi Xu


DIRECTORY STRUCTURE
-------------------

/mnt/scratch/groups/PSLY/
    Lyapunov/                          Research repository (Meta FAIR)
        train.py                       Entry point for generation and training
        02create.py&create_dataset.py  Step 2: Generate the cleaned datasets for train/valid/test

    Lyapunov_data/                     The directory for the all cleaned datasets

    Lyapunov_model/                    The directory for the Replication training results

    Lyfuntion/                         The directory for generating the datasets of power system
        01.py                          Step 1: Dataset generation
        ...
        07.py
        01_extract.py                  The python code for extracting the Lyapunov pairs
        ...
        07_CL_extract.py
        PS_..._all                     The extraction datasets
        PS_...                         The original power system datasets     

    train.sbatch                       The trianing file for replication 
    PS-train.sbatch                    Step 3: Train Power-System Models
    slurm_train_....err                SLURM output and error logs
    slurm_train_....out

REFERENCE
---------

@article{alfarano2024global,
  title={Global Lyapunov functions: a long-standing open problem in
         mathematics, with symbolic transformers},
  author={Alfarano, Alberto and Charton, Francois and Hayat, Amaury},
  journal={arXiv preprint arXiv:2410.08304},
  year={2024}
}
