#!/bin/bash
# A-posteriori WMLES: shl walls, traction from the model in the co-located
# RedisAI. Settings in turbChannel_inference.par ([WALLMODEL], [SGSMODEL]).
# The model takes the raw input columns (7 per off-wall point) and returns
# u_tau (trainer_wall_model_offline.py export), the same file as SOD2D.
#   BACKEND=CUDA: nekRS on GPU, TORCH model on GPU (MODEL=model_jit.pt)
#   BACKEND=CPU : nekRS CPU, ONNX model on CPU (MODEL=model.onnx); CPU
#                 RedisAI on MN5 has no torch backend
#   NSIM  MPI ranks for nekRS (default 2)
BACKEND=${BACKEND:-CUDA}
NSIM=${NSIM:-2}

for ext in box usr par re2; do cp turbChannel_inference.$ext turbChannel.$ext; done

if [ "$BACKEND" = "CUDA" ]; then
  MODEL=${MODEL:-./model_jit.pt}
  sim_arguments="--setup turbChannel.par --backend CUDA"
  infer="inference.backend=TORCH inference.device=GPU inference.devices_per_node=${NSIM}"
else
  MODEL=${MODEL:-./model.onnx}
  sim_arguments="--setup turbChannel.par --backend CPU"
  infer="inference.backend=ONNX inference.device=CPU inference.devices_per_node=1"
fi

python ssim_driver_marenostrum.py \
  database.load_exp=True \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=$NSIM run_args.simprocs_pn=$NSIM \
  sim.arguments="${sim_arguments}" \
  sim.copy_files=["./turbChannel.oudf","./turbChannel.usr","./turbChannel.par","./turbChannel.udf","./turbChannel.re2","./turbChannel.box","./my_drag.oudf","./affinity_nrs.sh","./restart.fld"] \
  inference.model_path=${MODEL} $infer

for ext in box usr par re2; do rm turbChannel.$ext; done
