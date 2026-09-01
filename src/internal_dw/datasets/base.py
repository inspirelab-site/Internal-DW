from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Minimal protocol for sequence datasets used by Koopman-Gram models.

    This is not a universal data loader. Each dataset should implement its own
    file/key/indexing logic, but return either the new sample dict:

        {
            "state": Tensor [T, ...],
            "external_input": Tensor [T, input_dim] or None,
            "label": int or Tensor,
            "metadata": dict,
        }

    or the legacy tuple:
        (state, external_input, label)

    The only framework-level shape convention is that state is time-first:
        single sample: [T, ...]
        batch:         [B, T, ...]
    """

    dataset_name = "base"
    has_external_input = False
    task_type = "generic_sequence"
    evaluator_name = "generic"

    def __getitem__(self, index):  # pragma: no cover
        raise NotImplementedError


def _stack_or_none(items: List[Any]):
    if all(x is None for x in items):
        return None
    if any(x is None for x in items):
        raise ValueError("A batch mixed None and non-None external_input values.")
    return torch.stack(items, dim=0)


def sequence_collate(batch):
    """Collate dict protocol batches while keeping metadata as a list.

    Backward-compatible with the old (state, external_input, label) tuple return.
    """
    if isinstance(batch[0], dict):
        states = [b["state"] for b in batch]
        exts = [b.get("external_input", None) for b in batch]
        labels = [b.get("label", 0) for b in batch]
        metadata = [b.get("metadata", {}) for b in batch]
        label_tensor = torch.as_tensor(labels, dtype=torch.long) if not torch.is_tensor(labels[0]) else torch.stack(labels, dim=0)
        return {
            "state": torch.stack(states, dim=0),
            "external_input": _stack_or_none(exts),
            "label": label_tensor,
            "metadata": metadata,
        }

    state, external_input, labels = zip(*batch)
    return torch.stack(state, dim=0), _stack_or_none(list(external_input)), torch.as_tensor(labels, dtype=torch.long)
