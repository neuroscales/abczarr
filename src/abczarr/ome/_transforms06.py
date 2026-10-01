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
each child the subset of axes it addresses, and a ``sequence`` hands its
first child the sequence's input axes and its last child the sequence's
output axes.

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
    are not known, when a ``mapAxis`` or ``byDimension`` refers to an axis
    its coordinate system does not have, and when a ``byDimension`` does not
    write every output axis exactly once.
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

    def transform(self, doc: tx.Any, axes: "tx.Tuple[Axes, Axes]") -> tx.Any:
        if not isinstance(doc, abc.Mapping):
            return doc
        doc = dict(doc)
        axes = (
            self._own_axes(doc, "input", axes[0]),
            self._own_axes(doc, "output", axes[1]),
        )
        kind = doc.get("type")
        if kind == "mapAxis":
            return self._map_axis(doc, axes)
        if kind == "projectAxis":
            return self._project_axis(doc, axes)
        if kind == "byDimension":
            return self._by_dimension(doc, axes)
        if kind == "sequence":
            return self._sequence(doc, axes)
        if kind == "bijection":
            if "forward" in doc:
                doc["forward"] = self.transform(doc["forward"], axes)
            if "inverse" in doc:
                doc["inverse"] = self.transform(
                    doc["inverse"], (axes[1], axes[0])
                )
            return doc
        if kind == "inverseOf":
            if not _has_inverse_of(self.to_v):
                raise ValueError(
                    f"InverseOf does not exist in OME {self.to_v}"
                )
            if "transformation" in doc:
                doc["transformation"] = self.transform(
                    doc["transformation"], (axes[1], axes[0])
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

    def _names(self, kind: str, axes: Axes, side: str) -> tx.List[str]:
        """The axis names of one side, or a clear error when unknown."""
        if axes is None or any(name is None for name in axes):
            self._fail(
                kind,
                f"the axis names of its {side} coordinate system are not "
                f"known, and they are needed to translate between axis names "
                f"and axis indices. Convert the object that declares the "
                f"coordinate system together with the transformation, such "
                f"as the enclosing multiscale.",
            )
        return tx.cast(tx.List[str], axes)

    # ------------------------------------------------------------------
    #   mapAxis
    # ------------------------------------------------------------------

    def _map_axis(self, doc: Json, axes: "tx.Tuple[Axes, Axes]") -> Json:
        value = doc.get("mapAxis")
        in_axes, out_axes = axes
        if isinstance(value, abc.Mapping):
            if _map_axis_by_name(self.to_v):
                return doc
            index = self._indices_from_names(
                value,
                self._names("mapAxis", in_axes, "input"),
                self._names("mapAxis", out_axes, "output"),
            )
        elif isinstance(value, list) and all(
            isinstance(i, int) for i in value
        ):
            index = list(value)
            if _map_axis_by_name(self.to_v):
                doc["mapAxis"] = self._names_from_indices(
                    index,
                    self._names("mapAxis", in_axes, "input"),
                    self._names("mapAxis", out_axes, "output"),
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
                f"not known",
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

    def _project_axis(self, doc: Json, axes: "tx.Tuple[Axes, Axes]") -> Json:
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
                "axes that affine needs is not known",
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

    def _sequence(self, doc: Json, axes: "tx.Tuple[Axes, Axes]") -> Json:
        items = doc.get("transformations")
        if isinstance(items, list):
            last = len(items) - 1
            doc["transformations"] = [
                self.transform(
                    item,
                    (
                        axes[0] if k == 0 else None,
                        axes[1] if k == last else None,
                    ),
                )
                for k, item in enumerate(items)
            ]
        return doc

    # ------------------------------------------------------------------
    #   byDimension
    # ------------------------------------------------------------------

    def _by_dimension(self, doc: Json, axes: "tx.Tuple[Axes, Axes]") -> Json:
        items = doc.get("transformations")
        if not isinstance(items, list):
            return doc
        in_axes, out_axes = axes
        src_wrap, src_in, src_out = _CHILD_LAYOUT[self.from_v]
        dst_wrap, dst_in, dst_out = _CHILD_LAYOUT[self.to_v]

        children = []
        written: tx.Dict[tx.Any, int] = {}
        for k, item in enumerate(items):
            if not isinstance(item, abc.Mapping):
                children.append(item)
                continue
            if src_wrap is None:
                inner = dict(item)
                in_refs = inner.pop(src_in, None)
                out_refs = inner.pop(src_out, None)
                extras: Json = {}
            else:
                inner = item.get(src_wrap)
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
            keys = out_index if out_axes is not None else out_refs
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
                in_new = self._names("byDimension", in_names, "input")
                out_new: tx.List[tx.Any] = self._names(
                    "byDimension", out_names, "output"
                )
            else:
                in_new = self._known(in_index, "input")
                out_new = self._known(out_index, "output")
            if dst_wrap is None:
                child = dict(inner) if isinstance(inner, abc.Mapping) else {}
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

        if out_axes is not None:
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
            if axes is None:
                return names, [None] * len(refs)
            unknown = [r for r in refs if r not in axes]
            if unknown:
                self._fail(
                    "byDimension",
                    f"transformation {k} names {side} axes {unknown} that "
                    f"the {side} coordinate system {axes} does not have",
                )
            return names, [axes.index(r) for r in refs]
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
        self, index: tx.List[tx.Optional[int]], side: str
    ) -> tx.List[int]:
        if any(i is None for i in index):
            self._fail(
                "byDimension",
                f"the axes of its {side} coordinate system are not known, "
                f"and they are needed to translate axis names into axis "
                f"indices. Convert the object that declares the coordinate "
                f"system together with the transformation, such as the "
                f"enclosing multiscale.",
            )
        return tx.cast(tx.List[int], index)


def _as_affine(doc: Json, drop: tx.Iterable[str], matrix: tx.List) -> Json:
    """Replace `doc` with an affine that keeps its name and references."""
    result: Json = {"type": "affine"}
    for key, value in doc.items():
        if key != "type" and key not in drop:
            result[key] = value
    result["affine"] = matrix
    return result
