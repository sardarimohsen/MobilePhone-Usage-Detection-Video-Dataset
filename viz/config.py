"""viz/config.py - run registry and data locations (portable version).

Paths are resolved relative to the repository, and can be overridden with
environment variables:

    PHONE_VAULT  folder that contains the data, e.g.
                   <VAULT>/Dataset/Dataset/<scene>/<video>.mp4
                   <VAULT>/phone_cache_v3/        (crop cache, training only)
                 default: <repo>/vault
    PHONE_RUNS   folder that contains the run directories, e.g.
                   <RUNS>/runs_c_s384_mid/C_s384_mid_f0_s0/head.pt
                 default: <repo>
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAULT = Path(os.environ.get("PHONE_VAULT", REPO / "vault"))
V4 = Path(os.environ.get("PHONE_RUNS", REPO))

RUNS = [
    dict(key="A",          root=V4/"runs_final",
         tag="A_final",    folds=[0,1,2,3,4], seeds=[0,1,2]),
    dict(key="C_gate",     root=V4/"runs_c_gate_final",
         tag="C_gate",     folds=[0,1,2,3,4], seeds=[0,1,2]),
    dict(key="C_s384_mid", root=V4/"runs_c_s384_mid",
         tag="C_s384_mid", folds=[0,1,2,3,4], seeds=[0,1,2]),
]

NARRATIVE_ORDER = ["A", "C_gate", "C_s384_mid"]
