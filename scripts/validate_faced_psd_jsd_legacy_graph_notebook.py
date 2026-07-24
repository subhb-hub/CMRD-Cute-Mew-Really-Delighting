from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "faced_psd_jsd_legacy_graph_backbone.ipynb"
parser = argparse.ArgumentParser()
parser.add_argument("--sanity", action="store_true")
parser.add_argument("--sanity-lr", type=float)
parser.add_argument("--sanity-steps", type=int)
arguments = parser.parse_args()
validation_name = (
    "faced_sqrt_jsd_legacy_graph_sanity_validation"
    if arguments.sanity
    else "faced_sqrt_jsd_legacy_graph_validation"
)
OUTPUT = ROOT / "runs" / validation_name / "executed_validation.ipynb"


notebook = nbformat.read(SOURCE, as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
for index, cell in enumerate(code_cells, 1):
    compile(cell.source, f"{SOURCE.name}:code-cell-{index}", "exec")

validation = copy.deepcopy(notebook)
parameter_cell = next(
    cell
    for cell in validation.cells
    if cell.cell_type == "code" and "RUN_NAME =" in cell.source
)
parameter_cell.source = (
    parameter_cell.source
    .replace(
        'RUN_NAME = "faced_sqrt_jsd_legacy_graph_seed42"',
        f'RUN_NAME = "{validation_name}"',
    )
    .replace("RUN_TRAINING = True", "RUN_TRAINING = False")
)
if not arguments.sanity:
    parameter_cell.source = parameter_cell.source.replace(
        "RUN_SANITY_GATE = True", "RUN_SANITY_GATE = False"
    )
else:
    if arguments.sanity_lr is not None:
        parameter_cell.source = parameter_cell.source.replace(
            "SANITY_LEARNING_RATE = 3e-4",
            f"SANITY_LEARNING_RATE = {arguments.sanity_lr!r}",
        )
    if arguments.sanity_steps is not None:
        parameter_cell.source = parameter_cell.source.replace(
            "SANITY_MAX_STEPS = 200",
            f"SANITY_MAX_STEPS = {arguments.sanity_steps}",
        )

client = NotebookClient(
    validation,
    timeout=600,
    kernel_name="cmrd",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(validation, OUTPUT)
print(f"validated_cells={len(validation.cells)} code_cells={len(code_cells)}")
print(OUTPUT)
