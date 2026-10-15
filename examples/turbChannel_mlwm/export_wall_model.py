#!/usr/bin/env python3
"""Re-export a wall model trained by trainer_wall_model_offline.py (SOD2D or
NekRS data) for another case: same network, the case's nu and y_s.

The exported model (raw wall-model inputs -> u_tau) carries the training
file's nu and y_s as constants of its derived features (e.g. Strategy I:
Re_s = |u_t| y_s/nu, u_tau = pred nu/y_s). The network works in wall units,
so moving it to a case with another nu (SOD2D channel nu = 5.357e-5, NekRS
turbChannel 1/19000) means re-exporting with that nu, not retraining.

Steps: rebuild the trainer's WallModelExport from <name>_params.json and the
weights in <name>.pt; check it reproduces the original <name>_jit.pt with the
training constants; re-trace with the target constants (from --witness, a
wm_witness.h5 of the target case, or --nu/--ys) in float32 as SOD2D. Run with
the torch of the RedisAI backend that will load the model (smartsim-nvhpc:
torch 2.3.1); a TorchScript file from a newer torch may not load there.

  MPI4PY_RC_INITIALIZE=false python export_wall_model.py \\
      --trainer <dir with trainer_wall_model_offline.py> \\
      --model <dir>/model_strategyI_inst --witness <nekRS case>/wm_witness.h5 \\
      --out model_strategyI_inst_nekrs_jit.pt
"""
import argparse
import json
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", required=True, help="directory of trainer_wall_model_offline.py")
    ap.add_argument("--model", required=True,
                    help="path prefix of the trained model: <prefix>.pt, <prefix>_jit.pt, <prefix>_params.json")
    ap.add_argument("--witness", help="wm_witness.h5 of the target case (nu, y_s)")
    ap.add_argument("--nu", type=float, help="target kinematic viscosity (overrides --witness)")
    ap.add_argument("--ys", type=float, nargs="*", help="target y_s per off-wall point (overrides --witness)")
    ap.add_argument("--out", required=True, help="output TorchScript file")
    args = ap.parse_args()

    sys.path.insert(0, args.trainer)
    import trainer_wall_model_offline as T

    with open(args.model + "_params.json") as fh:
        params = json.load(fh)
    meta, best = params["meta"], params["best_params"]
    old_const = dict(meta["export_constants"])

    # target constants
    new_const = dict(old_const)
    if args.witness:
        import h5py
        with h5py.File(args.witness, "r") as f:
            new_const["nu"] = float(f["nu"][0])
            ys = f["y_s"][:]
            for k in range(ys.shape[1]):
                if not np.allclose(ys[:, k], ys[0, k], rtol=1e-6):
                    raise SystemExit(f"y_s of point {k+1} varies along the wall: cannot be a model constant")
                new_const[f"ys_{k+1:02d}"] = float(ys[0, k])
    if args.nu is not None:
        new_const["nu"] = args.nu
    if args.ys:
        for k, v in enumerate(args.ys):
            new_const[f"ys_{k+1:02d}"] = v
    missing = set(old_const) - set(new_const)
    if missing:
        raise SystemExit(f"target constants missing {sorted(missing)}")

    def build(constants):
        core = T.WallModelFCN(meta["ndIn"], meta["ndOut"], best["nNeurons"], best["numHiddenLayers"])
        derived = T.DerivedInputFeatures(meta["all_raw_input_names"], meta["input_names"], constants)
        m = T.WallModelExport(core, derived, meta["inverse_names"])
        m.load_state_dict(torch.load(args.model + ".pt", map_location="cpu"))
        return m.eval()

    nraw = len(meta["all_raw_input_names"])
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.05, (4096, nraw)).astype(np.float32)
    x[:, 0] = rng.uniform(0.2, 1.3, len(x))       # streamwise velocity range at y_s
    xt = torch.from_numpy(x)

    # 1) the rebuild reproduces the original export
    ref = torch.jit.load(args.model + "_jit.pt", map_location="cpu").eval()
    with torch.no_grad():
        err = (build(old_const)(xt) - ref(xt)).abs().max().item() / ref(xt).abs().max().item()
    print(f"rebuild vs {args.model}_jit.pt with the training constants: max rel diff {err:.2e}")
    if err > 1e-6:
        raise SystemExit("rebuilt model does not reproduce the original export")

    # 2) re-trace with the target constants, float32 as SOD2D
    model = build(new_const)
    trace_x = torch.from_numpy(rng.uniform(0.1, 2.0, (8, nraw)).astype(np.float32))
    with torch.no_grad():
        jit = torch.jit.trace(model, trace_x)
    torch.jit.save(jit, args.out)
    with torch.no_grad():
        a, b = ref(xt).numpy().ravel(), jit(xt).numpy().ravel()
    print(f"constants  training: {old_const}\n           target:   {new_const}")
    print(f"mean u_tau on the test inputs: training constants {a.mean():.6e}, target {b.mean():.6e} "
          f"(ratio {b.mean()/a.mean():.4f})")
    print(f"saved {args.out} (torch {torch.__version__}, float32, "
          f"{nraw} raw inputs -> {meta['physical_names']})")


if __name__ == "__main__":
    main()
