from __future__ import annotations

from enum import Enum
from typing import Sequence


class Provider(str, Enum):
    """Supported dataset providers (openneuro, dandi, nemar, physionet, neurovault, kaggle, and other)."""
    OPENNEURO = "openneuro"
    DANDI = "dandi"
    PHYSIONET = "physionet"
    KAGGLE = "kaggle"
    NEUROVAULT = "neurovault"
    NEMAR = "nemar"
    OTHER = "other"


class DatasetStatus(str, Enum):
    """Lifecycle state of a registered dataset."""
    INGESTING = "ingesting"
    AVAILABLE = "available"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"


class JobStatus(str, Enum):
    """Lifecycle state of an ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class StorageMode(str, Enum):
    """How local dataset files are managed.

    Parameters
    ----------
    reference : str
        Keep files at their original path and record that path.
    move : str
        Move files into the registry's managed datasets directory.
    copy : str
        Copy files into managed storage while preserving the original source.
    """
    REFERENCE = "reference"
    MOVE = "move"
    COPY = "copy"



class Modality(str, Enum):
    """Controlled modality vocabulary.

    Notes
    -----
    ``ephys`` means electrophysiology;
    ``fnris`` means functional
    near-infrared spectroscopy.
    """
    EEG = "eeg"
    MEG = "meg"
    IEEG = "ieeg"
    FMRI = "fmri"
    FNRIS = "fnris"
    PET = "pet"
    SMRI = "smri"
    DMRI = "dmri"
    EPHYS = "ephys"
    OTHER = "other"


def normalize_modalities(values: Sequence[str | Modality]) -> list[str]:
    """Validate modality names and return unique lowercase values."""
    allowed = {item.value for item in Modality}
    normalized = {value.value if isinstance(value, Modality) else str(value).strip().lower() for value in values}
    invalid = normalized - allowed
    if invalid:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported modality {sorted(invalid)!r}; choose from: {choices}")
    return sorted(normalized)
