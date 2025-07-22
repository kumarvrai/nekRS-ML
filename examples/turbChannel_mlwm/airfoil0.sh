#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand
#PBS -l walltime=00:30:00
##PBS -q preemptable
#PBS -q debug
#PBS -A EnergyApps
#PBS -o out.o
#PBS -e error.e

INS_DIR=/home/kumarv/install_nrsml/
module use /soft/modulefiles
module load conda/2024-04-29
conda activate base
source $INS_DIR/smartnekrs-env/bin/activate

export HYDRA_FULL_ERROR=1
export CRAY_ACCEL_TARGET=nvidia80
export NEKRS_HOME=/home/kumarv/.local/nekrs-ml-bsc
#export NEKRS_HOME=/home/kumarv/.local/nekrs-ml-vish
export PATH=$NEKRS_HOME/bin:$PATH
#export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
#export TORCH_PATH=/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INS_DIR/SmartRedis/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INS_DIR/SmartRedis/install/lib64:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR

${PWD}/run_train_colocated.sh
#${PWD}/run_inference_colocated.sh
