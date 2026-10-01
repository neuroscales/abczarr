"""Convert 0.6 coordinate transformations between the 0.6 previews.

The 0.6 previews (``0.6.dev1`` .. ``0.6rc0`` and the ``0.6`` release) agree
on most transformations but disagree on how a transformation addresses
individual axes:

* ``mapAxis`` maps output axis names to input axis names in ``0.6.dev1``.
  From ``0.6.dev2`` on it is a list of input axis indices, one per output
  axis, that must be a permutation. A ``0.6.dev1`` mapping may drop or repeat
  input axes, which no later ``mapAxis`` can express.
* A ``byDimension`` child names its axes in ``input``/``output`` in
  ``0.6.dev1`` and in ``input_axes``/``output_axes`` in ``0.6.dev2``. From
  ``0.6.dev3`` on the child is wrapped in an object that addresses the axes
  by index, spelled ``input_axes``/``output_axes`` until ``0.6.dev4`` and
  ``inputAxes``/``outputAxes`` from ``0.6rc0`` on.
* ``projectAxis`` appears in ``0.6rc0``. ``inverseOf`` disappears in
  ``0.6.dev3``.

Translating between axis names and axis indices needs the axes of the
coordinate systems a transformation refers to. The caller supplies those as
a mapping from coordinate-system name to axis names. A ``byDimension`` hands
each child the subset of axes it addresses.

A ``sequence`` works out the axes between its steps. The axes are carried
forward from the sequence's input and backward from its output, one step at
a time. A step that names its own coordinate system fixes the axes on that
side. A step's input axes are read from the forward pass first, and its
output axes from the backward pass first, so that neither is inferred
through the step itself. The other pass fills in what the first one leaves
unknown. Axis names may legitimately differ between the two passes, for
example in a sequence from array axes to physical axes. The number of axes
may not. When the two passes disagree on the number of axes, the inferred
axes are discarded, and a conversion that needs them fails.

Where a construct has no counterpart in the target version, it is replaced by
an exactly equivalent ``affine`` whose matrix only selects input coordinates
(or writes zeros). That covers a ``mapAxis`` that drops or repeats axes and a
``projectAxis`` converted below ``0.6rc0``. The ``affine`` is not turned back
into the original construct on the way back, so the round trip preserves the
meaning but not the spelling.

Everything here works on the JSON form of a transformation; ``base`` parses
the result into the target version's classes.
"""

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

#: The 0.6 versions, oldest to newest.
CHAIN = ("0.6.dev1", "0.6.dev2", "0.6.dev3", "0.6.dev4", "0.6rc0", "0.6")

#: How each version lays out one child of a ``byDimension``: the key that
#: wraps the child transformation (``None`` when the child is the
#: transformation itself, addressing axes by name) and the keys that list its
#: input and output axes.
_CHILD_LAYOUT = {
    "0.6.dev1": (None, "input", "output"),
    "0.6.dev2": (None, "input_axes", "output_axes"),
    "0.6.dev3": ("transformation", "input_axes", "output_axes"),
    "0.6.dev4": ("transformation", "input_axes", "output_axes"),
    "0.6rc0": ("transformation", "inputAxes", "outputAxes"),
    "0.6": ("transformation", "inputAxes", "outputAxes"),
}

# The axes of one side (input or output) of a transformation, as far as they
# are known: ``None`` when nothing is known, else one entry per axis holding
# the axis name, or ``None`` where only the axis's position is known.
Axes = tx.Optional[tx.List[tx.Optional[str]]]
# For each side (input, output), a sentence saying why an enclosing sequence
# could not work out the axes there, or ``None``.
Notes = tx.Tuple[tx.Optional[str], tx.Optional[str]]
Systems = tx.Mapping[str, tx.List[str]]
Report = tx.Callable[[str], None]
Json = tx.Dict[str, tx.Any]


def _rank(version: str) -> int:
    return CHAIN.index(version)


def _map_axis_by_name(version: str) -> bool:
    return version == "0.6.dev1"


def _has_project_axis(version: str) -> bool:
    return _rank(version) >= _rank("0.6rc0")


def _has_inverse_of(version: str) -> bool:
    return _rank(version) <= _rank("0.6.dev2")


def convert(
    doc: Json,
    from_v: str,
    to_v: str,
    systems: Systems,
    report: Report,
    axes: "tx.Tuple[Axes, Axes]" = (None, None),
) -> Json:
    """Convert one transformation's JSON from `from_v` to `to_v`.

    `systems` maps each known coordinate-system name to its axis names.
    `report` is called with the name of anything the target version cannot
    hold. `axes` gives the input and output axes inherited from an enclosing
    transformation, for a transformation that names no coordinate system of
    its own.

    Raises ``ValueError`` when the conversion needs axis names or counts that
    are not known, and when a ``mapAxis`` or ``byDimension`` refers to an
    axis its coordinate system does not have. A ``ValueError`` is also raised
    when a ``byDimension`` does not write every output axis exactly once, and
    when one of its entries does not have the layout that `from_v` requires.
    """
    return _Step(from_v, to_v, systems, report).transform(doc, axes)


def is_permutation(index: tx.Sequence[tx.Any]) -> bool:
    """Whether `index` lists each of ``0 .. len(index) - 1`` exactly once."""
    return sorted(index) == list(range(len(index)))


def _selection(rows: tx.Sequence[tx.Optional[int]], n_in: int) -> tx.List:
    """An affine matrix whose output row ``k`` copies input ``rows[k]``.

    A ``None`` row writes zero. The matrix has one row per output axis and one
    column per input axis, plus the translation column, which is all zero.
    """
    matrix = []
    for source in rows:
        row = [0.0] * (n_in + 1)
        if source is not None:
            row[source] = 1.0
        matrix.append(row)
    return matrix


class _Step:
    def __init__(
        self, from_v: str, to_v: str, systems: Systems, report: Report
    ) -> None:
        self.from_v, self.to_v = from_v, to_v
        self.systems, self.report = systems, report

    # ------------------------------------------------------------------
    #   dispatch
    # ------------------------------------------------------------------

    def transform(
        self,
        doc: tx.Any,
        axes: "tx.Tuple[Axes, Axes]",
        notes: Notes = (None, None),
    ) -> tx.Any:
        if not isinstance(doc, abc.Mapping):
            return doc
        doc = dict(doc)
        axes = (
            self._own_axes(doc, "input", axes[0]),
            self._own_axes(doc, "output", axes[1]),
        )
        # A side the transformation names itself owes nothing to its context.
        notes = (
            None if "input" in doc else notes[0],
            None if "output" in doc else notes[1],
        )
        kind = doc.get("type")
        if kind == "mapAxis":
            return self._map_axis(doc, axes, notes)
        if kind == "projectAxis":
            return self._project_axis(doc, axes, notes)
        if kind == "byDimension":
            return self._by_dimension(doc, axes, notes)
        if kind == "sequence":
            return self._sequence(doc, axes, notes)
        if kind == "bijection":
            if "forward" in doc:
                doc["forward"] = self.transform(doc["forward"], axes, notes)
            if "inverse" in doc:
                doc["inverse"] = self.transform(
                    doc["inverse"], (axes[1], axes[0]), (notes[1], notes[0])
                )
            return doc
        if kind == "inverseOf":
            if not _has_inverse_of(self.to_v):
                raise ValueError(
                    f"InverseOf does not exist in OME {self.to_v}"
                )
            if "transformation" in doc:
                doc["transformation"] = self.transform(
                    doc["transformation"],
                    (axes[1], axes[0]),
                    (notes[1], notes[0]),
                )
            return doc
        return doc

    def _own_axes(self, doc: Json, key: str, inherited: Axes) -> Axes:
        """The axes `doc` names for its `key` side, else the inherited ones."""
        if key not in doc:
            return inherited
        ref = doc[key]
        if isinstance(ref, list) and all(isinstance(r, str) for r in ref):
            # a 0.6.dev1 byDimension child lists its axis names directly
            return list(ref)
        name = ref
        if isinstance(ref, abc.Mapping):
            # a reference that also carries a path points into another group
            name = None if ref.get("path") is not None else ref.get("name")
        if isinstance(name, str) and name in self.systems:
            return list(self.systems[name])
        return None

    def _fail(self, kind: str, message: str) -> tx.NoReturn:
        raise ValueError(
            f"cannot convert {kind} from OME {self.from_v} to {self.to_v}: "
            f"{message}"
        )

    def _unknown(
        self, kind: str, what: str, side: str, note: tx.Optional[str]
    ) -> tx.NoReturn:
        """Fail because the axes of one side are not known well enough."""
        if note is not None:
            cause = note
        else:
            cause = (
                "Axes are known only for a coordinate system that the "
                "converted object declares, and for the steps of a sequence "
                "whose axes follow from such a coordinate system."
            )
        self._fail(
            kind,
            f"the {what} of its {side} coordinate system are not known, and "
            f"they are needed to translate between axis names and axis "
            f"indices. {cause}",
        )

    def _names(
        self, kind: str, axes: Axes, side: str, note: tx.Optional[str]
    ) -> tx.List[str]:
        """The axis names of one side, or a clear error when unknown."""
        if axes is None or any(name is None for name in axes):
            self._unknown(kind, "axis names", side, note)
        return tx.cast(tx.List[str], axes)

    # ------------------------------------------------------------------
    #   mapAxis
    # ------------------------------------------------------------------

    def _map_axis(
        self, doc: Json, axes: "tx.Tuple[Axes, Axes]", notes: Notes
    ) -> Json:
        value = doc.get("mapAxis")
        in_axes, out_axes = axes
        if isinstance(value, abc.Mapping):
            if _map_axis_by_name(self.to_v):
                return doc
            index = self._indices_from_names(
                value,
                self._names("mapAxis", in_axes, "input", notes[0]),
                self._names("mapAxis", out_axes, "output", notes[1]),
            )
        elif _is_index(value):
            index = list(value)
            if _map_axis_by_name(self.to_v):
                doc["mapAxis"] = self._names_from_indices(
                    index,
                    self._names("mapAxis", in_axes, "input", notes[0]),
                    self._names("mapAxis", out_axes, "output", notes[1]),
                )
                return doc
        else:
            return doc

        if out_axes is not None and len(out_axes) != len(index):
            self._fail(
                "mapAxis",
                f"mapAxis {index} has {len(index)} entries but its output "
                f"coordinate system has {len(out_axes)} axes",
            )
        n_in = None if in_axes is None else len(in_axes)
        if is_permutation(index) and n_in in (None, len(index)):
            doc["mapAxis"] = index
            return doc
        # The target requires a permutation. A mapping that drops or repeats
        # input axes is a pure selection, so an affine expresses it exactly.
        if n_in is None:
            self._fail(
                "mapAxis",
                f"mapAxis {index} is not a permutation, so it must become an "
                f"affine, and the number of input axes that affine needs is "
                f"not known.{_because(notes[0])}",
            )
        bad = [i for i in index if not 0 <= i < n_in]
        if bad:
            self._fail(
                "mapAxis",
                f"mapAxis {index} refers to input axes {bad}, but the input "
                f"coordinate system has {n_in} axes",
            )
        return _as_affine(doc, ("mapAxis",), _selection(index, n_in))

    def _indices_from_names(
        self,
        mapping: tx.Mapping[str, tx.Any],
        in_names: tx.List[str],
        out_names: tx.List[str],
    ) -> tx.List[int]:
        unknown = [k for k in mapping if k not in out_names]
        if unknown:
            self._fail(
                "mapAxis",
                f"mapAxis names output axes {unknown} that the output "
                f"coordinate system {out_names} does not have",
            )
        missing = [k for k in out_names if k not in mapping]
        if missing:
            self._fail(
                "mapAxis",
                f"mapAxis does not say which input axis output axes "
                f"{missing} take their values from",
            )
        sources = [mapping[k] for k in out_names]
        unknown = [s for s in sources if s not in in_names]
        if unknown:
            self._fail(
                "mapAxis",
                f"mapAxis names input axes {unknown} that the input "
                f"coordinate system {in_names} does not have",
            )
        return [in_names.index(s) for s in sources]

    def _names_from_indices(
        self,
        index: tx.List[int],
        in_names: tx.List[str],
        out_names: tx.List[str],
    ) -> tx.Dict[str, str]:
        if len(index) != len(out_names):
            self._fail(
                "mapAxis",
                f"mapAxis {index} has {len(index)} entries but its output "
                f"coordinate system {out_names} has {len(out_names)} axes",
            )
        bad = [i for i in index if not 0 <= i < len(in_names)]
        if bad:
            self._fail(
                "mapAxis",
                f"mapAxis {index} refers to input axes {bad}, but the input "
                f"coordinate system {in_names} has {len(in_names)} axes",
            )
        return {out: in_names[i] for out, i in zip(out_names, index)}

    # ------------------------------------------------------------------
    #   projectAxis
    # ------------------------------------------------------------------

    def _project_axis(
        self, doc: Json, axes: "tx.Tuple[Axes, Axes]", notes: Notes
    ) -> Json:
        if _has_project_axis(self.to_v):
            return doc
        dropped = list(doc.get("droppedInputs") or [])
        created = list(doc.get("createdOutputs") or [])
        in_axes, out_axes = axes
        if in_axes is not None:
            n_in = len(in_axes)
        elif out_axes is not None:
            n_in = len(out_axes) - len(created) + len(dropped)
        else:
            self._fail(
                "projectAxis",
                "it becomes an affine, and the number of input or output "
                "axes that affine needs is not known."
                + _because(notes[0] or notes[1]),
            )
        kept = [i for i in range(n_in) if i not in dropped]
        n_out = len(kept) + len(created)
        bad = [i for i in dropped if not 0 <= i < n_in]
        if bad:
            self._fail(
                "projectAxis",
                f"droppedInputs {bad} lie outside the {n_in} input axes",
            )
        bad = [i for i in created if not 0 <= i < n_out]
        if bad or (out_axes is not None and len(out_axes) != n_out):
            self._fail(
                "projectAxis",
                f"createdOutputs {created} and droppedInputs {dropped} do "
                f"not fit an input coordinate system of {n_in} axes and an "
                f"output coordinate system of "
                f"{n_out if out_axes is None else len(out_axes)} axes",
            )
        sources = iter(kept)
        rows = [None if o in created else next(sources) for o in range(n_out)]
        return _as_affine(
            doc, ("droppedInputs", "createdOutputs"), _selection(rows, n_in)
        )

    # ------------------------------------------------------------------
    #   sequence
    # ------------------------------------------------------------------

    def _sequence(
        self, doc: Json, axes: "tx.Tuple[Axes, Axes]", notes: Notes
    ) -> Json:
        items = doc.get("transformations")
        if not isinstance(items, list):
            return doc
        n = len(items)
        # Boundary k lies before step k, so boundary 0 is the input of the
        # sequence and boundary n is its output. A boundary is fixed when a
        # step next to it names its own coordinate system.
        fixed: tx.List[Axes] = [None] * (n + 1)
        for k, item in enumerate(items):
            fixed[k + 1] = self._declared(item, "output")
        for k, item in enumerate(items):
            own = self._declared(item, "input")
            if own is not None:
                fixed[k] = own
        ahead, ahead_notes = _propagate(items, fixed, axes[0], notes[0], True)
        behind, behind_notes = _propagate(
            items, fixed, axes[1], notes[1], False
        )

        conflict = _count_conflict(ahead, behind)
        if conflict is not None:
            # The steps disagree about the number of axes, so nothing that
            # was inferred is trusted. A conversion that needs the inferred
            # axes fails with the disagreement as the cause.
            k = conflict
            note = (
                f"The enclosing sequence has {len(ahead[k] or [])} axes "
                f"{_boundary(k, n)} when read forward from its input, but "
                f"{len(behind[k] or [])} when read backward from its output."
            )
            ahead = behind = [ahead[0], *fixed[1:n], behind[n]]
            ahead_notes = behind_notes = [
                ahead_notes[0],
                *[None if f is not None else note for f in fixed[1:n]],
                behind_notes[n],
            ]

        # A step's input is read from the forward pass first and its output
        # from the backward pass first, which never passes through the step.
        doc["transformations"] = [
            self.transform(
                item,
                (
                    _pick(ahead[k], behind[k]),
                    _pick(behind[k + 1], ahead[k + 1]),
                ),
                (ahead_notes[k], behind_notes[k + 1]),
            )
            for k, item in enumerate(items)
        ]
        return doc

    def _declared(self, item: tx.Any, key: str) -> Axes:
        """The axes a sequence step names for its `key` side, if any."""
        if not isinstance(item, abc.Mapping) or key not in item:
            return None
        return self._own_axes(item, key, None)

    # ------------------------------------------------------------------
    #   byDimension
    # ------------------------------------------------------------------

    def _by_dimension(
        self, doc: Json, axes: "tx.Tuple[Axes, Axes]", notes: Notes
    ) -> Json:
        items = doc.get("transformations")
        if not isinstance(items, list):
            return doc
        in_axes, out_axes = axes
        src_wrap, src_in, src_out = _CHILD_LAYOUT[self.from_v]
        dst_wrap, dst_in, dst_out = _CHILD_LAYOUT[self.to_v]
        # Axes addressed by name can be checked for coverage only when every
        # output axis has a known name.
        by_index = src_wrap is not None or _named(out_axes)

        children = []
        written: tx.Dict[tx.Any, int] = {}
        for k, item in enumerate(items):
            if not isinstance(item, abc.Mapping):
                self._fail(
                    "byDimension",
                    f"entry {k} of its transformations is not an object",
                )
            if src_wrap is None:
                inner = dict(item)
                in_refs = inner.pop(src_in, None)
                out_refs = inner.pop(src_out, None)
                extras: Json = {}
            else:
                inner = item.get(src_wrap)
                if not isinstance(inner, abc.Mapping):
                    self._fail(
                        "byDimension",
                        f"entry {k} of its transformations has no "
                        f"{src_wrap!r} object. OME {self.from_v} wraps each "
                        f"transformation of a byDimension in an object that "
                        f"also lists the axes the transformation applies to.",
                    )
                in_refs, out_refs = item.get(src_in), item.get(src_out)
                extras = {
                    key: value
                    for key, value in item.items()
                    if key not in (src_wrap, src_in, src_out)
                }
            in_names, in_index = self._child_axes(k, in_refs, in_axes, "input")
            out_names, out_index = self._child_axes(
                k, out_refs, out_axes, "output"
            )

            # every output axis is written by exactly one child
            keys = out_index if by_index else out_refs
            for key, name in zip(keys, out_names):
                label = name if name is not None else key
                if key in written:
                    self._fail(
                        "byDimension",
                        f"output axis {label!r} is written by transformations "
                        f"{written[key]} and {k}, but each output axis must "
                        f"be written exactly once",
                    )
                written[key] = k

            inner = self.transform(inner, (in_names, out_names))
            if dst_wrap is None:
                in_new = self._names(
                    "byDimension", in_names, "input", notes[0]
                )
                out_new: tx.List[tx.Any] = self._names(
                    "byDimension", out_names, "output", notes[1]
                )
            else:
                in_new = self._known(in_index, "input", notes[0])
                out_new = self._known(out_index, "output", notes[1])
            if dst_wrap is None:
                child = dict(inner)
                for key in extras:
                    self.report(key)
                for key in (dst_in, dst_out):
                    if key in child:
                        self.report(key)
                child[dst_in], child[dst_out] = in_new, out_new
            else:
                child = {dst_wrap: inner, dst_in: in_new, dst_out: out_new}
                child.update(extras)
            children.append(child)

        if out_axes is not None and by_index:
            for position, name in enumerate(out_axes):
                if position not in written:
                    label = name if name is not None else position
                    self._fail(
                        "byDimension",
                        f"no transformation writes output axis {label!r}, "
                        f"but each output axis must be written exactly once",
                    )
        doc["transformations"] = children
        return doc

    def _child_axes(
        self, k: int, refs: tx.Any, axes: Axes, side: str
    ) -> "tx.Tuple[tx.List[tx.Optional[str]], tx.List[tx.Optional[int]]]":
        """Read the axes child `k` addresses on one side.

        Returns the axis names and the axis indices, each entry ``None``
        where it cannot be worked out from what is known.
        """
        if not isinstance(refs, list):
            self._fail(
                "byDimension",
                f"transformation {k} does not list its {side} axes",
            )
        if _CHILD_LAYOUT[self.from_v][0] is None:
            names: tx.List[tx.Optional[str]] = list(refs)
            if not _named(axes):
                return names, [None] * len(refs)
            unknown = [r for r in refs if r not in axes]
            if unknown:
                self._fail(
                    "byDimension",
                    f"transformation {k} names {side} axes {unknown} that "
                    f"the {side} coordinate system {axes} does not have",
                )
            return names, [tx.cast(list, axes).index(r) for r in refs]
        index: tx.List[tx.Optional[int]] = list(refs)
        if axes is None:
            return [None] * len(refs), index
        bad = [
            i for i in refs if not (isinstance(i, int) and 0 <= i < len(axes))
        ]
        if bad:
            self._fail(
                "byDimension",
                f"transformation {k} refers to {side} axes {bad}, but the "
                f"{side} coordinate system has {len(axes)} axes",
            )
        return [axes[i] for i in refs], index

    def _known(
        self,
        index: tx.List[tx.Optional[int]],
        side: str,
        note: tx.Optional[str],
    ) -> tx.List[int]:
        if any(i is None for i in index):
            self._unknown("byDimension", "axes", side, note)
        return tx.cast(tx.List[int], index)


# ----------------------------------------------------------------------
#   sequence: inferring the axes between steps
# ----------------------------------------------------------------------


def _is_index(value: tx.Any) -> bool:
    return isinstance(value, list) and all(isinstance(i, int) for i in value)


def _named(axes: Axes) -> bool:
    """Whether `axes` is known and names every axis."""
    return axes is not None and None not in axes


def _pick(first: Axes, second: Axes) -> Axes:
    """The better known of two views of the same axes, `first` on a tie."""
    for axes in (first, second):
        if _named(axes):
            return list(tx.cast(list, axes))
    return first if first is not None else second


def _hidden_by(item: tx.Any, k: int) -> str:
    """Say that step `k` of a sequence hides the axes beyond it."""
    kind = item.get("type") if isinstance(item, abc.Mapping) else None
    label = kind if isinstance(kind, str) else "transformation"
    return (
        f"The axes cannot be inferred through the {label} at index {k} of "
        f"the enclosing sequence."
    )


def _because(note: tx.Optional[str]) -> str:
    return "" if note is None else " " + note


def _propagate(
    items: tx.List[tx.Any],
    fixed: tx.List[Axes],
    start: Axes,
    note: tx.Optional[str],
    forward: bool,
) -> "tx.Tuple[tx.List[Axes], tx.List[tx.Optional[str]]]":
    """Carry axes through the steps of a sequence in one direction.

    Returns the axes at every boundary and, where they are not fully known,
    the reason. A fixed boundary keeps its axes. `start` and `note` seed the
    first boundary when it is not fixed.
    """
    n = len(items)
    axes: tx.List[Axes] = [None] * (n + 1)
    notes: tx.List[tx.Optional[str]] = [None] * (n + 1)
    first = 0 if forward else n
    if fixed[first] is not None:
        axes[first] = fixed[first]
    else:
        axes[first], notes[first] = start, note
    for k in range(n) if forward else reversed(range(n)):
        src, dst = (k, k + 1) if forward else (k + 1, k)
        if fixed[dst] is not None:
            axes[dst] = fixed[dst]
            continue
        result, hides = _infer(items[k], axes[src], forward)
        axes[dst] = result
        if not _named(result):
            notes[dst] = _hidden_by(items[k], k) if hides else notes[src]
    return axes, notes


def _count_conflict(
    ahead: tx.List[Axes], behind: tx.List[Axes]
) -> tx.Optional[int]:
    """The first boundary where the two passes disagree on the axis count."""
    for k, (one, other) in enumerate(zip(ahead, behind)):
        if one is not None and other is not None and len(one) != len(other):
            return k
    return None


def _boundary(k: int, n: int) -> str:
    if k == 0:
        return "before transformation 0"
    if k == n:
        return f"after transformation {n - 1}"
    return f"between transformations {k - 1} and {k}"


def _infer(item: tx.Any, axes: Axes, forward: bool) -> "tx.Tuple[Axes, bool]":
    """Infer the axes on one side of a sequence step from the other side.

    With `forward`, `axes` are the step's input axes and the output axes are
    returned. Otherwise the input axes are inferred from the output axes.
    The second value tells whether the step itself hides axis names or the
    number of axes, as opposed to passing on what was already unknown.
    """
    if not isinstance(item, abc.Mapping):
        return None, True
    kind = item.get("type")
    if kind in ("identity", "scale", "translation"):
        if axes is not None:
            return list(axes), False
        vector = item.get(kind)
        if isinstance(vector, list):
            return [None] * len(vector), False
        return None, False
    if kind == "mapAxis":
        return _infer_map_axis(item.get("mapAxis"), axes, forward)
    if kind == "projectAxis":
        return _infer_project_axis(item, axes, forward)
    if kind in ("affine", "rotation"):
        shape = _matrix_shape(item.get(kind), square=kind == "rotation")
        if shape is None:
            return None, True
        n_in, n_out = shape
        return [None] * (n_out if forward else n_in), True
    # coordinates, displacements, and nested byDimension, sequence,
    # bijection or inverseOf: nothing is known without declared axes.
    return None, True


def _infer_map_axis(
    value: tx.Any, axes: Axes, forward: bool
) -> "tx.Tuple[Axes, bool]":
    if isinstance(value, abc.Mapping):
        # A name mapping says how many outputs there are, but JSON key order
        # carries no meaning, and the mapping may drop or repeat inputs.
        return ([None] * len(value), True) if forward else (None, True)
    if not _is_index(value):
        return None, True
    if forward:
        if axes is None:
            return [None] * len(value), False
        return [axes[i] if 0 <= i < len(axes) else None for i in value], False
    if not is_permutation(value):
        return None, True
    if axes is None:
        return [None] * len(value), False
    if len(axes) != len(value):
        return None, False
    inputs: tx.List[tx.Optional[str]] = [None] * len(value)
    for k, i in enumerate(value):
        inputs[i] = axes[k]
    return inputs, False


def _infer_project_axis(
    item: Json, axes: Axes, forward: bool
) -> "tx.Tuple[Axes, bool]":
    dropped = item.get("droppedInputs") or []
    created = item.get("createdOutputs") or []
    if not (_is_index(dropped) and _is_index(created)):
        return None, True
    if len(set(dropped)) != len(dropped) or len(set(created)) != len(created):
        return None, True
    if axes is None:
        return None, False
    # Seen backward, the created outputs are dropped and the dropped inputs
    # are created.
    removed, added = (dropped, created) if forward else (created, dropped)
    n_from = len(axes)
    n_to = n_from - len(removed) + len(added)
    if not all(0 <= i < n_from for i in removed):
        return None, True
    if not all(0 <= i < n_to for i in added):
        return None, True
    kept = iter(axes[i] for i in range(n_from) if i not in removed)
    result = [None if i in added else next(kept) for i in range(n_to)]
    return result, bool(added)


def _matrix_shape(
    matrix: tx.Any, square: bool
) -> "tx.Optional[tx.Tuple[int, int]]":
    """The input and output axis counts of an inline affine or rotation.

    An affine matrix has one row per output axis and one column per input
    axis plus the translation column. A last row of ``[0, ..., 0, 1]`` that
    makes the matrix square is the homogeneous row, which is not an axis. A
    rotation matrix is square, with one row and one column per axis.
    """
    if not isinstance(matrix, list) or not matrix:
        return None
    if not all(isinstance(row, list) for row in matrix):
        return None
    n_cols = len(matrix[0])
    if any(len(row) != n_cols for row in matrix):
        return None
    n_rows = len(matrix)
    if square:
        return (n_rows, n_rows) if n_rows == n_cols else None
    if n_cols < 2:
        return None
    homogeneous = [0] * (n_cols - 1) + [1]
    if n_rows == n_cols and matrix[-1] == homogeneous:
        n_rows -= 1
    return n_cols - 1, n_rows


def _as_affine(doc: Json, drop: tx.Iterable[str], matrix: tx.List) -> Json:
    """Replace `doc` with an affine that keeps its name and references."""
    result: Json = {"type": "affine"}
    for key, value in doc.items():
        if key != "type" and key not in drop:
            result[key] = value
    result["affine"] = matrix
    return result
