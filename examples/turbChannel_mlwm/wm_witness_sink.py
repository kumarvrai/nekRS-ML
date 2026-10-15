#!/usr/bin/env python3
"""Witness sink: collect wall-model training samples from the SmartSim
database and write them to wm_witness.h5 with the SOD2D schema, so
trainer_wall_model_offline.py and the SOD2D analysis scripts read NekRS
data unchanged.

Runs as the "train" component of the SmartSim driver (1 process), next to
nekRS with [WALLMODEL] wm_mode = train. Protocol (src/plugins/smartRedis.hpp):
  wm_meta, tensorInfo.<rank>, wm_static.<rank>   once, at setup
  x.<rank>.<step> then wm_sample.<i> = [step, time]  per sample
  wm_done = [n_samples]                           at the end of the run
The sink reads the samples in order, appends them, deletes them from the
database, and stops the simulation through check-run once --max-samples
are written.

HDF5 layout (as SOD2D wm_witness.h5): input_XX/output_XX (npts, nt), time
(nt), istep (nt), nu (1), y_s (npts, K), wall_xyz/wall_normal/wall_tangent1/
wall_tangent2 (npts, 3), nwit_per_wall (1) = K, sample_instantaneous (1),
nwitPar/nwitOffset (nranks). Points are ordered by rank, then by local wall
node. File attributes record the producer and the data contract.
"""
import argparse
import os
import sys
import time as wallclock

import h5py
import numpy as np
from smartredis import Client

META_FIELDS = ["contract_version", "nu", "K", "n_inputs", "n_outputs",
               "sample_instantaneous", "period_walave", "project_normal",
               "nranks", "witness_dist"]
COLS_PER_POINT = ["u_1", "u_2", "u_3", "dpdx_1", "dpdx_2", "dpdx_3", "utau_reichardt"]


def log(msg):
    print(f"[wm_sink] {msg}", flush=True)


def wait_for(client, key, poll_ms, timeout):
    t0 = wallclock.time()
    while not client.tensor_exists(key):
        if wallclock.time() - t0 > timeout:
            return False
        wallclock.sleep(poll_ms / 1000.0)
    return True


def read_layout(client, poll_ms, timeout):
    if not wait_for(client, "wm_meta", poll_ms, timeout):
        raise RuntimeError(f"wm_meta did not appear in the database within {timeout} s")
    meta = dict(zip(META_FIELDS, client.get_tensor("wm_meta").tolist()))
    nranks = int(meta["nranks"])
    K = int(meta["K"])
    n_in, n_out = int(meta["n_inputs"]), int(meta["n_outputs"])
    npts = np.zeros(nranks, dtype=np.int64)
    for r in range(nranks):
        info = client.get_tensor(f"tensorInfo.{r}")
        npts[r] = int(info[0])
        if int(info[4]) != n_in or int(info[5]) != n_out:
            raise RuntimeError(f"tensorInfo.{r} disagrees with wm_meta: {info}")
    static = [client.get_tensor(f"wm_static.{r}") for r in range(nranks) if npts[r] > 0]
    static = np.concatenate(static, axis=0)
    if static.shape != (npts.sum(), 12 + K):
        raise RuntimeError(f"wm_static shape {static.shape}, expected {(npts.sum(), 12 + K)}")
    return meta, npts, static


def create_file(path, meta, npts, static, dtype):
    ntot = int(npts.sum())
    K = int(meta["K"])
    n_in, n_out = int(meta["n_inputs"]), int(meta["n_outputs"])
    f = h5py.File(path, "w")
    chunk = (min(ntot, 1 << 17), 1)
    for j in range(n_in):
        f.create_dataset(f"input_{j + 1:02d}", (ntot, 0), maxshape=(ntot, None), dtype=dtype, chunks=chunk)
    for j in range(n_out):
        f.create_dataset(f"output_{j + 1:02d}", (ntot, 0), maxshape=(ntot, None), dtype=dtype, chunks=chunk)
    f.create_dataset("time", (0,), maxshape=(None,), dtype=dtype, chunks=(1,))
    f.create_dataset("istep", (0,), maxshape=(None,), dtype=np.int32, chunks=(1,))
    f["nu"] = np.array([meta["nu"]], dtype=dtype)
    f["nwit_per_wall"] = np.array([K], dtype=np.int32)
    f["sample_instantaneous"] = np.array([int(meta["sample_instantaneous"])], dtype=np.int32)
    f["nwitPar"] = npts.astype(np.int32)
    f["nwitOffset"] = (np.cumsum(npts) - npts).astype(np.int32)
    f["wall_xyz"] = static[:, 0:3].astype(dtype)
    f["wall_normal"] = static[:, 3:6].astype(dtype)
    f["wall_tangent1"] = static[:, 6:9].astype(dtype)
    f["wall_tangent2"] = static[:, 9:12].astype(dtype)
    f["y_s"] = static[:, 12:12 + K].astype(dtype)

    names = [f"{c}@{k + 1}" for k in range(K) for c in COLS_PER_POINT]
    f.attrs["code"] = "nekRS"
    f.attrs["input_columns"] = np.array(names, dtype="S")
    frame = "wall(t1,n,t2)" if meta["project_normal"] else "global(x,y,z)"
    f.attrs["output_columns"] = np.array([f"tau_w_{i}" for i in (1, 2, 3)], dtype="S")
    f.attrs["vector_frame"] = frame
    for k, v in meta.items():
        f.attrs[k] = v
    return f


def append_sample(f, data, step, t, n_in, n_out):
    nt = f["time"].shape[0]
    for j in range(n_in):
        d = f[f"input_{j + 1:02d}"]
        d.resize(nt + 1, axis=1)
        d[:, nt] = data[:, j]
    for j in range(n_out):
        d = f[f"output_{j + 1:02d}"]
        d.resize(nt + 1, axis=1)
        d[:, nt] = data[:, n_in + j]
    f["time"].resize(nt + 1, axis=0)
    f["time"][nt] = t
    f["istep"].resize(nt + 1, axis=0)
    f["istep"][nt] = step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5file", default="wm_witness.h5")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="stop the simulation (check-run = 0) after this many samples; 0 = until wm_done")
    ap.add_argument("--skip-samples", type=int, default=0,
                    help="discard this many initial samples (start-up transient)")
    ap.add_argument("--poll-ms", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds without a new sample before giving up")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="float32 as SOD2D (rp=4)")
    ap.add_argument("--flush-every", type=int, default=20)
    # The SmartSim driver appends trainer options (--dbnodes, --device, --ppn, ...)
    args, _ = ap.parse_known_args()

    # Only one writer: under mpirun with several ML ranks, rank > 0 exits.
    for var in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK", "SLURM_PROCID"):
        if int(os.environ.get(var, "0")) > 0:
            return

    client = Client(cluster=False, logger_name="wm_sink")
    meta, npts, static = read_layout(client, args.poll_ms, args.timeout)
    n_in, n_out = int(meta["n_inputs"]), int(meta["n_outputs"])
    ranks = np.nonzero(npts)[0]
    log(f"{len(npts)} ranks, {npts.sum()} wall points, K={int(meta['K'])}, "
        f"{n_in} inputs, {n_out} outputs, nu={meta['nu']:.6g}, "
        f"{'instantaneous' if meta['sample_instantaneous'] else 'walave'}")

    f = create_file(args.h5file, meta, npts, static, np.dtype(args.dtype))
    i = 0
    written = 0
    last = wallclock.time()
    try:
        while True:
            key = f"wm_sample.{i}"
            if not client.tensor_exists(key):
                if client.tensor_exists("wm_done") and i >= int(client.get_tensor("wm_done")[0]):
                    log("simulation finished (wm_done)")
                    break
                if wallclock.time() - last > args.timeout:
                    log(f"no sample for {args.timeout} s, stopping")
                    break
                wallclock.sleep(args.poll_ms / 1000.0)
                continue
            step, t = client.get_tensor(key).tolist()
            step = int(step)
            parts = []
            for r in ranks:
                xkey = f"x.{r}.{step}"
                parts.append(client.get_tensor(xkey))
                client.delete_tensor(xkey)
            client.delete_tensor(key)
            data = np.concatenate(parts, axis=0)
            if data.shape != (npts.sum(), n_in + n_out):
                raise RuntimeError(f"sample {i}: shape {data.shape}")
            if i >= args.skip_samples:
                append_sample(f, data, step, t, n_in, n_out)
                written += 1
                if written % args.flush_every == 0:
                    f.flush()
                    log(f"{written} samples written (step {step}, t={t:.4f})")
            i += 1
            last = wallclock.time()
            if args.max_samples and written >= args.max_samples:
                log(f"reached --max-samples {args.max_samples}, telling nekRS to stop")
                client.put_tensor("check-run", np.zeros(2, dtype=np.int32))
                break
    finally:
        f.close()
        log(f"closed {args.h5file} with {written} samples")


if __name__ == "__main__":
    sys.exit(main())
