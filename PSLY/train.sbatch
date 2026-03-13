# The code for replication

#!/bin/bash

#SBATCH -J Lyapunov_Train
#SBATCH -p gpu-a-lowsmall
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -o slurm_train_%j.out
#SBATCH -e slurm_train_%j.err

echo "======== Slurm Job started on $(hostname) ========"

source ~/.bashrc
conda activate Lyapunov
cd /users/sgaxu2/Lyapunov

MODEL_OUTPUT_DIR="/users/sgaxu2/Lyapunov_models"
mkdir -p $MODEL_OUTPUT_DIR

python train.py \
--dump_path "$MODEL_OUTPUT_DIR" \
--n_enc_layers 6 \
--n_dec_layers 6 \
--emb_dim 640 \
--n_heads 10 \
--batch_size 4 \
--batch_size_eval 16 \
--max_src_len 0 \
--max_len 1024 \
--max_output_len 512 \
--optimizer "adam_inverse_sqrt,warmup_updates=10000,lr=0.0001" \
--epoch_size 300000 \
--max_epoch 10000 \
--num_workers 1 \
--export_data false \
--eval_size 200 \
--eval_verbose 0 \
--beam_eval true \
--lyap_polynomial_H true \
--lyap_basic_functions_num true \
--lyap_pure_polynomial true \
--lyap_SOS_checker true \
--lyap_SOS_fwd_gen false \
--stopping_criterion "valid_ode_lyapunov_beam_acc,100" \
--validation_metrics "valid_ode_lyapunov_beam_acc" \
--reload_size -1 \
--reload_data "ode_lyapunov,/users/sgaxu2/Lyapunov_data/final_datasetdata.prefix.100.data.prefix.100.cleaned.train,/users/sgaxu2/Lyapunov_data/final_datasetdata.prefix.100.data.prefix.100.cleaned.valid.final"

echo "======== Slurm Job finished ========"
