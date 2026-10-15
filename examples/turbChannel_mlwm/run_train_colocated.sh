#!/bin/bash
# Training-data run: resolved LES (no-slip walls) streams witness samples to
# the co-located DB; wm_witness_sink.py writes them to wm_witness.h5 in the
# SOD2D layout. Settings in turbChannel_train.par ([WALLMODEL], [SGSMODEL]).
#   BACKEND=CUDA (MN5 ACC, default) or BACKEND=CPU (OCCA serial, needs a CPU build)
#   NSIM      MPI ranks for nekRS (default 4)
#   SINK_ARGS extra sink options, e.g. "--max-samples 2000 --skip-samples 100"
BACKEND=${BACKEND:-CUDA}
NSIM=${NSIM:-4}
SINK_ARGS=${SINK_ARGS:-}

for ext in box usr par re2; do cp turbChannel_train.$ext turbChannel.$ext; done

if [ "$BACKEND" = "CUDA" ]; then
  sim_arguments="--setup turbChannel.par --backend CUDA --device-id 0"
  sim_affinity="sim.affinity=./affinity_nrs.sh"
else
  sim_arguments="--setup turbChannel.par --backend CPU"
  sim_affinity=""
fi

python ssim_driver_marenostrum.py \
  database.load_exp=True \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=$NSIM run_args.simprocs_pn=$NSIM \
  sim.arguments="${sim_arguments}" $sim_affinity \
  sim.link_files=["./turbChannel.oudf","./turbChannel.usr","./turbChannel.par","./turbChannel.udf","./turbChannel.re2","./turbChannel.box","./my_drag.oudf","./affinity_nrs.sh","./restart.fld"] \
  train.executable="${PWD}/wm_witness_sink.py --h5file ${PWD}/wm_witness.h5 ${SINK_ARGS}" \
  run_args.mlprocs=1 run_args.mlprocs_pn=1 \
  train.device=cpu train.affinity=./affinity_ml.sh

for ext in box usr par re2; do rm turbChannel.$ext; done
