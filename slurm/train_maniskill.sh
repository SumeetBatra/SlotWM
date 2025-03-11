#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c4
#SBATCH --mem-per-cpu=6G
#SBATCH --output=tmp/tdmpc2_maniskill_%j.log

export MUJOCO_GL=egl

SEED=0
TASK="pick-cube"
RUN_NAME="tdmpc2_maniskill_${TASK}_seed${SEED}"
srun python -m tdmpc2.train --task=$TASK \
                            --exp_name=$RUN_NAME \
                            --wandb_group="maniskill" \
                            --seed=$SEED