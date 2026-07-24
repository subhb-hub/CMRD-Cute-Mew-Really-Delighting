from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "faced_hierarchical_frequency_band_channel_time.ipynb"


def replace_once_or_keep(source: str, old: str, new: str, *, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label!r} occurrence, found {count}")
    return source.replace(old, new, 1)


notebook = nbformat.read(PATH, as_version=4)
if len(notebook.cells) != 24:
    raise RuntimeError(f"Expected 24 notebook cells, found {len(notebook.cells)}")

parameter_cell = notebook.cells[2]
parameter_cell.source = replace_once_or_keep(
    parameter_cell.source,
    "SANITY_LEARNING_RATE = 4e-4",
    "SANITY_LEARNING_RATE = 1e-4  # validated in float32; higher values collapse",
    label="SANITY_LEARNING_RATE",
)

sanity_markdown = notebook.cells[13]
sanity_markdown.source = sanity_markdown.source.replace(
    "The sanity model disables dropout, label smoothing, channel-vote loss\n"
    "and subject adversarial loss.",
    "The sanity model disables AMP, dropout, label smoothing, channel-vote loss\n"
    "and subject adversarial loss. Full precision is deliberate: this gate diagnoses\n"
    "model learnability independently of the production-training AMP choice.",
)

sanity_cell = notebook.cells[14]
sanity_cell.source = replace_once_or_keep(
    sanity_cell.source,
    "with autocast_context():\n            sanity_outputs",
    "with contextlib.nullcontext():\n            sanity_outputs",
    label="sanity training autocast",
)
sanity_cell.source = replace_once_or_keep(
    sanity_cell.source,
    "with torch.no_grad(), autocast_context():\n                checked",
    "with torch.no_grad(), contextlib.nullcontext():\n                checked",
    label="sanity evaluation autocast",
)
sanity_cell.source = replace_once_or_keep(
    sanity_cell.source,
    '"subject_adversarial_loss": 0.0,',
    '"subject_adversarial_loss": 0.0,\n'
    '        "precision": "float32",',
    label="sanity precision audit",
)

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{PATH.name}:cell-{index}", "exec")
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["cmrd_bugfix"] = {
    "reason": "sanity gate inherited bfloat16 AMP and used an unstable learning rate",
    "sanity_precision": "float32",
    "sanity_learning_rate": 1e-4,
    "validated_sanity_steps": 40,
    "validated_sanity_accuracy": 1.0,
    "validated_sanity_loss": 0.013977368362247944,
    "training_smoke": "2 epochs passed with default batch sizes and bfloat16 AMP",
    "outer_target_loaded_during_validation": False,
}
notebook.metadata["kernelspec"] = {
    "display_name": "Python (cmrd)",
    "language": "python",
    "name": "cmrd",
}
nbformat.write(notebook, PATH)
print(PATH)
