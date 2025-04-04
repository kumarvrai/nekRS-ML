PWD_DIR=$HOME/install_nrsml/
module load bsc/1.0 cuda/11.8 ucx/1.16.0 gcc/11.4.0 cmake/3.30.5 cudnn/9.0.0-cuda11 openmpi/4.1.5-ucx1.16-gcc
module load miniforge
source activate smartsim

export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH
export TORCH_CMAKE_PATH=$( python -c 'import torch;print(torch.utils.cmake_prefix_path)' )
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
#export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" #added by vishal

export SMARTREDIS_DIR=$PWD_DIR/SmartRedis/install
#export RAI_PATH=$PWD_DIR/redisai/install-gpu/redisai.so
export SMARTSIM_REDISAI=1.2.7
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH


git clone https://github.com/CrayLabs/SmartRedis.git
cd SmartRedis
make lib DEP_CC=cc DEP_CXX=CC
pip install -e .
cd ..

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

git clone https://github.com/kumarvrai/nekRS-ML.git
cd nekRS-ML
git checkout ml-wall-model

export NEKRS_HOME=$HOME/.local/nekrs-ml-bsc
export PATH=$NEKRS_HOME/bin:$PATH
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install_nrsml/SmartRedis/install/lib64:$LD_LIBRARY_PATH

CC=mpicc CXX=mpic++ FC=mpif77 ./nrsconfig -DCMAKE_INSTALL_PREFIX=$NEKRS_HOME -DENABLE_SMARTREDIS=1 -DSMARTREDIS_PATH=$HOME/install_nrsml/SmartRedis
