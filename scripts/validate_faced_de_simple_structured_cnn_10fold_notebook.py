from __future__ import annotations

import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "faced_de_simple_structured_cnn_10fold.ipynb"
RUN_NAME = "faced_de_simple_structured_cnn_fold1_smoke"


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label!r} occurrence, found {count}")
    return source.replace(old, new, 1)


notebook = nbformat.read(SOURCE, as_version=4)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{SOURCE.name}:cell-{index}", "exec")

validation = copy.deepcopy(notebook)
parameters = validation.cells[2]
parameters.source = replace_once(
    parameters.source,
    'RUN_NAME = "faced_de_simple_structured_cnn_10fold_seed42"',
    f'RUN_NAME = "{RUN_NAME}"',
    label="RUN_NAME",
)
parameters.source = replace_once(
    parameters.source,
    "FOLDS_TO_RUN = tuple(range(1, 11))",
    "FOLDS_TO_RUN = (1,)",
    label="FOLDS_TO_RUN",
)
parameters.source = replace_once(
    parameters.source,
    "EPOCHS = 100",
    "EPOCHS = 10",
    label="EPOCHS",
)

output = ROOT / "runs" / RUN_NAME / "executed_validation.ipynb"
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
