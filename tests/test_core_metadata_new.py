"""Polymorphic construction through ``Metadata.__new__``.

A subclass registered with a regex discriminator is chosen when a string
value matches. A non-string value simply does not match that subclass -- it
must fall through to the base rather than raise, whether the discriminator is
passed by keyword or positionally.
"""

import re

import typing_extensions as tx

from abczarr._core.auto import autofrozen
from abczarr._core.metadata import Metadata, register_subclass


@autofrozen
class _Codec(Metadata):
    name: tx.Any = None


@register_subclass(name=re.compile(r"^blosc"))
@autofrozen
class _Blosc(_Codec):
    name: tx.Any = None
    level: int = 5


def test_regex_discriminator_selects_on_string_match() -> None:
    assert isinstance(_Codec(name="blosc-lz4"), _Blosc)


def test_regex_discriminator_skips_on_string_mismatch() -> None:
    assert type(_Codec(name="gzip")) is _Codec


def test_non_string_discriminator_falls_through_keyword() -> None:
    # a non-string value cannot match a regex -> the base, not a TypeError
    assert type(_Codec(name=123)) is _Codec


def test_non_string_discriminator_falls_through_positional() -> None:
    # the same, constructed positionally (the path that lacked the str guard)
    assert type(_Codec(123)) is _Codec
