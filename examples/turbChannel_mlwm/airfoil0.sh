#!/bin/bash
### Job name on queue
#SBATCH --job-name=nekrs_mlwm

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH --output=out.o
#SBATCH --error=error.e

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

### Queue and account
##SBATCH --qos=acc_bsccase
##SBATCH --time=2-00:00:00
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --account=bsc21
##sbatch --dependency=afterany:9878054

### MN% modules
BASE_DIR=/home/bsc/bsc021712/install_nrsml/
module purge
#module getdefault nekrs
##module getdefault nekrs-cuda11
#module load miniforge
#source activate smartsim
##module load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx cmake
##module load miniforge
##source activate smartsim-nvhpc
module getdefault sod2d
module unload python
module load miniforge
source activate smartsim-nvhpc

export NEKRS_HOME=$HOME/.local/nekrs_v23_ml
export PATH=$NEKRS_HOME/bin:$PATH

export SMARTREDIS_DIR=/apps/ACC/MINIFORGE/24.3.0-0/envs/smartsim-nvhpc/
export SMARTSIM_REDISAI=1.2.7
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
#export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
#export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib64:$LD_LIBRARY_PATH

#echo $SLURM_JOBID 
#echo $SLURM_SUBMIT_DIR 
#echo $SLURM_SUBMIT_HOST 
#echo $SLURM_JOB_NODELIST 
#echo $SLURM_ARRAY_TASK_ID

./run_train_colocated.sh
#./run_inference_colocated.sh
