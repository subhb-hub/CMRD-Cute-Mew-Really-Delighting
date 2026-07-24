from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "faced_hierarchical_frequency_band_channel_time.ipynb"


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label!r} occurrence, found {count}")
    return source.replace(old, new, 1)


parser = argparse.ArgumentParser()
parser.add_argument("--train-smoke", action="store_true")
parser.add_argument("--amp", action="store_true")
parser.add_argument("--default-batches", action="store_true")
parser.add_argument("--sanity-steps", type=int)
parser.add_argument("--sanity-lr", type=float)
arguments = parser.parse_args()

notebook = nbformat.read(SOURCE, as_version=4)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{SOURCE.name}:cell-{index}", "exec")

validation = copy.deepcopy(notebook)
parameter_cell = validation.cells[2]
validation_name = (
    "faced_hierarchical_fbct_train_smoke_amp"
    if arguments.train_smoke and arguments.amp
    else "faced_hierarchical_fbct_train_smoke_float32"
    if arguments.train_smoke
    else "faced_hierarchical_fbct_sanity_float32"
)
parameter_cell.source = replace_once(
    parameter_cell.source,
    'RUN_NAME = "faced_hierarchical_fbct_base_seed42"',
    f'RUN_NAME = "{validation_name}"',
    label="RUN_NAME",
)
parameter_cell.source = replace_once(
    parameter_cell.source,
    "USE_AMP = True",
    f"USE_AMP = {bool(arguments.amp)}",
    label="USE_AMP",
)
if arguments.sanity_steps is not None:
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "SANITY_MAX_STEPS = 300",
        f"SANITY_MAX_STEPS = {arguments.sanity_steps}",
        label="SANITY_MAX_STEPS",
    )
if arguments.sanity_lr is not None:
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "SANITY_LEARNING_RATE = 4e-4",
        f"SANITY_LEARNING_RATE = {arguments.sanity_lr!r}",
        label="SANITY_LEARNING_RATE",
    )
if not arguments.train_smoke:
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "RUN_TRAINING = True",
        "RUN_TRAINING = False",
        label="RUN_TRAINING",
    )
else:
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "EPOCHS = 80",
        "EPOCHS = 2",
        label="EPOCHS",
    )
    if not arguments.default_batches:
        parameter_cell.source = replace_once(
            parameter_cell.source,
            'BATCH_SIZE = MODEL_CFG["batch_size"]',
            "BATCH_SIZE = 16",
            label="BATCH_SIZE",
        )
        parameter_cell.source = replace_once(
            parameter_cell.source,
            "EVAL_BATCH_SIZE = 192",
            "EVAL_BATCH_SIZE = 32",
            label="EVAL_BATCH_SIZE",
        )
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "WARMUP_EPOCHS = 5",
        "WARMUP_EPOCHS = 1",
        label="WARMUP_EPOCHS",
    )
    parameter_cell.source = replace_once(
        parameter_cell.source,
        "EARLY_STOPPING_PATIENCE = 12",
        "EARLY_STOPPING_PATIENCE = 0",
        label="EARLY_STOPPING_PATIENCE",
    )

# A learnability gate must diagnose the model in full precision, independent
# of the production-training AMP choice.
sanity_cell = validation.cells[14]
sanity_cell.source = sanity_cell.source.replace(
    "with autocast_context():\n            sanity_outputs",
    "with contextlib.nullcontext():\n            sanity_outputs",
)
sanity_cell.source = sanity_cell.source.replace(
    "with torch.no_grad(), autocast_context():\n                checked",
    "with torch.no_grad(), contextlib.nullcontext():\n                checked",
)

output = ROOT / "runs" / validation_name / "executed_validation.ipynb"
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
