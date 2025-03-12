#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c4
#SBATCH --mem-per-cpu=6G
#SBATCH --output=tmp/tdmpc2_%j.log

export MUJOCO_GL=egl
export PYTHONPATH=$HOME/SlotWM/tdmpc2:$PYTHONPATH

SEED=0
ENV="maniskill"
TASK="pick-cube"
RUN_NAME="tdmpc2_${ENV}_${TASK}_seed_${SEED}"
srun python -m tdmpc2.train task=$TASK \
                       obs="rgb" \
                       exp_name=$RUN_NAME \
                       wandb_group=$ENV \
                       seed=$SEED