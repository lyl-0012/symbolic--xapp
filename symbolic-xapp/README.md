# Executable Symbolic xApp Controller

This repository reproduces one reported point of the proposed method:

- `N=10` UEs;
- Full channel features;
- `D_max=7`, `n_min=10`, and 5 cross-fitting folds;
- teacher and rule seed `42`;
- chronological 80% final-fit / 20% test split.

## Run

Use Python 3.9. On Linux or macOS:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_teacher_labels.py --root .
python reproduce_point.py
```

On Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\generate_teacher_labels.py --root .
python reproduce_point.py
```

The first command generates frozen RL-teacher labels. It is the longest stage
and can be resumed by rerunning the same command. The second fits the proposed
cross-fitted symbolic classifier chain, exports executable IF--THEN rules, and
evaluates the test suffix. Generated files are written to
`DROO_labels/` and `results/`; these directories need not be committed.

For a diagnostic run after label generation:

```bash
python reproduce_point.py --quick-test-frames 100
```

Only output with `formal_single_point=true` is the formal point. A valid formal
run must also report `rule_tree_joint_fidelity=1.0`.

## Files

```text
configs/experiment_protocol.json    locked experiment specification
data/data_10.mat                    channel-gain matrix only
scripts/generate_teacher_labels.py  offline RL-teacher label generation
src/symbolic_xapp/memory_dnn.py     offline teacher component
src/symbolic_xapp/rule_model.py     proposed symbolic controller
src/symbolic_xapp/resource_allocator.py
reproduce_point.py                  one-point training and evaluation
requirements.txt                    pinned dependencies
```

The workflow checks the data hash, fixed experiment settings, chronological index separation, binary teacher actions, allocation feasibility, and exact
rule--tree agreement. It outputs the single-point metrics, executable rules,
and per-frame predictions and utilities under `results/`.

## Data, attribution, and license

The channel gains and offline teacher component are derived from the
[official DROO repository](https://github.com/revenol/DROO), released under the
MIT License. The MAT file retains only `input_h`; upstream optimal actions,
allocations, objectives, and comparison code are excluded. See
`THIRD_PARTY_NOTICES.md` .
