"""Compatibility layer re-exporting the new auto utilities."""

from .imports import import_all

import_all(".auto", locals(), __package__, add_to_all="attrs")
