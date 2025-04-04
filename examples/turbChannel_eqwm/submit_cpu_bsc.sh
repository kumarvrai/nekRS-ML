#!/bin/bash
#SBATCH --account=bsc21
#SBATCH --job-name=nk_aoa10
#SBATCH --ntasks=112
#SBATCH --time=00:20:00
#SBATCH --qos=gp_debug
##SBATCH --time=0-00:20:00
##SBATCH --qos=gp_bsccase
#SBATCH --output=out.o
#SBATCH --error=error.e

### MN% modules
module purge 
module load openmpi/4.1.5-gcc gcc/12.3.0

export NEKRS_HOME=$HOME/.local/nekrs_v23
export PATH=$NEKRS_HOME/bin:$PATH

#export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
#export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

mpirun $NEKRS_HOME/bin/nekrs --setup turbChannel

