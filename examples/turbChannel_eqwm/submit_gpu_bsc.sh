#!/bin/bash
### Job name on queue
#SBATCH --job-name=nekrs_wm

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
#SBATCH --time=00:20:00
#SBATCH --account=bsc21
##sbatch --dependency=afterany:9878054

### MN% modules
BASE_DIR=/home/bsc/bsc021712/install_nrsml/
module load bsc/1.0 cuda/11.8 ucx/1.16.0 gcc/11.4.0 cmake/3.30.5 cudnn/9.0.0-cuda11 openmpi/4.1.5-ucx1.16-gcc

##module load miniforge
##source activate smartsim
##
##export SMARTREDIS_DIR=$BASE_DIR/SmartRedis/install
###export RAI_PATH=$PWD_DIR/redisai/install-gpu/redisai.so
##export SMARTSIM_REDISAI=1.2.7
##export PATH=$SMARTREDIS_DIR/bin:$PATH
##export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib64:$LD_LIBRARY_PATH

export NEKRS_HOME=$HOME/.local/nekrs_v23
export PATH=$NEKRS_HOME/bin:$PATH
#export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
#export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

mpirun $NEKRS_HOME/bin/nekrs --setup turbChannel_inference
#nrspre turbChannel_inference 1
