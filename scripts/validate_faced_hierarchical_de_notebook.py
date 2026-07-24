from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "faced_hierarchical_de_band_channel_time.ipynb"


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label!r} occurrence, found {count}")
    return source.replace(old, new, 1)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--run-name", default="faced_hierarchical_de_bct_train_smoke_amp")
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--skip-sanity", action="store_true")
arguments = parser.parse_args()
if arguments.epochs < 0:
    raise ValueError("--epochs must be non-negative")

notebook = nbformat.read(SOURCE, as_version=4)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{SOURCE.name}:cell-{index}", "exec")

validation = copy.deepcopy(notebook)
parameters = validation.cells[2]
parameters.source = replace_once(
    parameters.source,
    'RUN_NAME = "faced_hierarchical_de_bct_base_seed42"',
    f'RUN_NAME = "{arguments.run_name}"',
    label="RUN_NAME",
)
parameters.source = replace_once(
    parameters.source,
    "USE_AMP = True",
    f"USE_AMP = {not arguments.no_amp}",
    label="USE_AMP",
)
if arguments.skip_sanity:
    parameters.source = replace_once(
        parameters.source,
        "RUN_SANITY_GATE = True",
        "RUN_SANITY_GATE = False",
        label="RUN_SANITY_GATE",
    )
if arguments.epochs == 0:
    parameters.source = replace_once(
        parameters.source,
        "RUN_TRAINING = True",
        "RUN_TRAINING = False",
        label="RUN_TRAINING",
    )
else:
    parameters.source = replace_once(
        parameters.source,
        "EPOCHS = 80",
        f"EPOCHS = {arguments.epochs}",
        label="EPOCHS",
    )
    warmup = min(5, max(1, arguments.epochs // 4))
    parameters.source = replace_once(
        parameters.source,
        "WARMUP_EPOCHS = 5",
        f"WARMUP_EPOCHS = {warmup}",
        label="WARMUP_EPOCHS",
    )
    parameters.source = replace_once(
        parameters.source,
        "EARLY_STOPPING_PATIENCE = 12",
        "EARLY_STOPPING_PATIENCE = 0",
        label="EARLY_STOPPING_PATIENCE",
    )

output = ROOT / "runs" / arguments.run_name / "executed_validation.ipynb"
client = NotebookClient(
    validation,
    timeout=900,
    kernel_name="cmrd",
    resources={"metadata": {"path": str(ROOT)}},
)
try:
    client.execute()
finally:
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(validation, output)
print(f"validated_cells={len(validation.cells)}")
print(output)
