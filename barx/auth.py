"""Credential helpers shared by BARX training and evaluation."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_hf_token(value: str | Path | None) -> str | None:
    """Resolve an optional Hugging Face token file or environment variable.

    Public BARX artifacts require no token. A non-path string is interpreted as
    an environment-variable name so credentials never need to appear in a
    command line or configuration file.
    """

    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser().read_text().strip()
    try:
        return os.environ[value]
    except KeyError as error:
        raise ValueError(
            f"Hugging Face token environment variable {value!r} is not set"
        ) from error
