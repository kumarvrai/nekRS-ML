#!/bin/bash
# Build and install nekRS-ML (branch wm-sod2d-compat) on MareNostrum 5 ACC:
# nvhpc 24.3 + parallel HDF5 (same toolchain as SOD2D), CUDA backend (the
# CPU/Serial backend is in the same binary), SmartRedis, HDF5 witness files.
#
#   ./install.sh                 configure (if needed) + build + install
#   NJ=16 ./install.sh           parallel build jobs (default 32)
#   RECONFIGURE=1 ./install.sh   re-run cmake configure (after changing options
#                                or adding/removing files)
#
# Overridable: NRS_INSTALL_PREFIX (install prefix), BUILD_DIR, SMARTREDIS_PATH,
# ENABLE_HDF5 (ON/OFF), NEKRS_GPU_MPI (ON/OFF). NEKRS_HOME is deliberately not
# used as the prefix: ~/.bashrc exports it for an older install, which this
# build would otherwise overwrite.
#
# Works from the ACC login (modules) and from the GPP login, where the ACC
# modules do not load: the toolchain is then set from its absolute paths.
# Run jobs with the same modules (module getdefault sod2d no longer restores:
# load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx by name) and submit
# ACC jobs with "#SBATCH --export=NONE" + "#!/bin/bash -l".
set -e

SCRATCH=${SCRATCH:-/gpfs/scratch/bsc21/bsc021712}
PREFIX=${NRS_INSTALL_PREFIX:-$SCRATCH/.local/nekrs_v23_wm}
SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-$SRC_DIR/build}
SMARTREDIS_PATH=${SMARTREDIS_PATH:-$HOME/install_nrsml/SmartRedis}
ENABLE_HDF5=${ENABLE_HDF5:-ON}
NEKRS_GPU_MPI=${NEKRS_GPU_MPI:-ON}
NJ=${NJ:-32}

# ---- toolchain: nvhpc 24.3 (+ its OpenMPI/HPC-X) and parallel HDF5 ---------
NVH=/apps/ACC/NVIDIA-HPC-SDK/24.3/Linux_x86_64/24.3
H5=/apps/ACC/HDF5/1.14.1-2/NVIDIA/NVHPCX
module purge > /dev/null 2>&1 || true
if module load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx > /dev/null 2>&1 && command -v nvc++ > /dev/null; then
  echo "toolchain: modules nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx"
else
  echo "toolchain: ACC modules unavailable (GPP login?), using absolute paths"
  export NVHPC=/apps/ACC/NVIDIA-HPC-SDK/24.3 NVHPC_ROOT=$NVH
  export PATH=$H5/bin:$NVH/comm_libs/mpi/bin:$NVH/compilers/extras/qd/bin:$NVH/compilers/ucx/bin:$NVH/compilers/bin:$NVH/cuda/bin:/apps/GPP/CMAKE/3.30.5/bin:$PATH
  export LD_LIBRARY_PATH=$H5/lib:$NVH/comm_libs/nvshmem/lib:$NVH/comm_libs/ucx/lib:$NVH/comm_libs/nccl/lib:$NVH/math_libs/lib64:$NVH/compilers/lib:$NVH/compilers/extras/qd/lib:$NVH/cuda/extras/CUPTI/lib64:$NVH/cuda/lib64:$LD_LIBRARY_PATH
  export CPATH=$NVH/math_libs/include:$NVH/comm_libs/nccl/include:$NVH/comm_libs/ucx/include:$NVH/comm_libs/nvshmem/include:$NVH/compilers/extras/qd/include/qd:$CPATH
fi
command -v cmake > /dev/null || export PATH=/apps/GPP/CMAKE/3.30.5/bin:$PATH
export HDF5_ROOT=$H5 HDF5_DIR=$H5
unset LDFLAGS
export CC=mpicc CXX=mpic++ FC=mpif77
export LD_LIBRARY_PATH=$SMARTREDIS_PATH/install/lib64:$SMARTREDIS_PATH/install/lib:$LD_LIBRARY_PATH

echo "source  : $SRC_DIR"
echo "build   : $BUILD_DIR"
echo "install : $PREFIX"
echo "options : SmartRedis ($SMARTREDIS_PATH)  HDF5=$ENABLE_HDF5  GPU-aware MPI=$NEKRS_GPU_MPI"
echo "compiler: $(command -v mpic++) -> $(mpic++ --version 2>/dev/null | grep -m1 -i nvc)"

# ---- configure --------------------------------------------------------------
if [ -f "$BUILD_DIR/CMakeCache.txt" ] && [ -z "$RECONFIGURE" ]; then
  cached=$(grep "^CMAKE_INSTALL_PREFIX:" "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)
  if [ "$cached" != "$PREFIX" ]; then
    echo "ERROR: $BUILD_DIR is configured for prefix $cached, not $PREFIX (use RECONFIGURE=1)" >&2
    exit 1
  fi
fi
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ -n "$RECONFIGURE" ]; then
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" -Wfatal-errors \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DNEKRS_GPU_MPI=$NEKRS_GPU_MPI \
    -DENABLE_SMARTREDIS=ON -DSMARTREDIS_PATH="$SMARTREDIS_PATH" \
    -DENABLE_HDF5=$ENABLE_HDF5
fi

# ---- build + install --------------------------------------------------------
cmake --build "$BUILD_DIR" --target install -j "$NJ"

echo
echo "installed nekRS in $PREFIX"
echo "  export NEKRS_HOME=$PREFIX"
echo "  export PATH=\$NEKRS_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=$SMARTREDIS_PATH/install/lib64:\$LD_LIBRARY_PATH   # libsmartredis (not in RUNPATH)"
echo "  export NEKRS_GPU_MPI=0     # at run time on MN5 ACC: GPU-aware MPI segfaults in UCX"
