"""
public_panel_pipeline.py -- run every public-book experiment of the article, in order, on one run
directory, with the working directory set to the output directory so each script's CSVs land there.

usage: python research/public_panel_pipeline.py <run_dir> <out_dir> [--skip=step1,step2] [--only=step,...]
steps (in order): stack, phi, blend, offdiag, snapshots, voltarget, voltarget_snapshots,
                  lambda, regime, state_conditional, psd, hidden_shorts, geometry,
                  allocator34, allocator252
Each step writes <out_dir>/NN_<step>.log. A failing step is reported and the pipeline continues.
"""
import os
import subprocess
import sys
import time

RUN = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
RES = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(OUT, "preqp_weights_stacked.csv")
skip = set(); only = None
for a in sys.argv[3:]:
    if a.startswith("--skip="):
        skip |= set(a.split("=", 1)[1].split(","))
    if a.startswith("--only="):
        only = set(a.split("=", 1)[1].split(","))

STEPS = [
    ("stack", ["stack_intent_weights.py", RUN, W]),
    ("phi", ["phi_persistence.py", RUN, W]),
    ("blend", ["blend_falsification.py", RUN, "252", W]),
    ("offdiag", ["offdiag_falsification.py", RUN, "252", W]),
    ("snapshots", ["phi_from_snapshots.py", RUN, W]),
    ("voltarget", ["voltarget_prescription.py", RUN, W, "252"]),
    ("voltarget_snapshots", ["voltarget_snapshots.py", RUN, W, "252"]),
    ("lambda", ["lambda_regime.py", RUN, W, "252"]),
    ("regime", ["regime_indicator.py", RUN, W, "252"]),
    ("state_conditional", ["phi_state_conditional.py", RUN, W]),
    ("psd", ["blend_psd_check.py", RUN, W, "252"]),
    ("hidden_shorts", ["hidden_shorts_check.py", RUN, W]),
    ("geometry", ["phi_geometry_check.py", RUN, W]),
    ("allocator34", ["allocator_risk_parity.py", RUN, W, "34", "1"]),
    ("allocator252", ["allocator_risk_parity.py", RUN, W, "252", "1"]),
]

for i, (name, args) in enumerate(STEPS):
    if name in skip or (only is not None and name not in only):
        continue
    log = os.path.join(OUT, f"{i:02d}_{name}.log"); t0 = time.time()
    cmd = [sys.executable, "-u", os.path.join(RES, args[0])] + args[1:]
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=OUT, stdout=f, stderr=subprocess.STDOUT)
    print(f"{name:20s} rc={rc} {time.time() - t0:6.0f}s -> {os.path.basename(log)}", flush=True)
print("done")
