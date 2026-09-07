"""The high-level [ImageConfig][abczarr.ome.config.ImageConfig].

An ``ImageConfig`` resolves axes, voxel geometry and a downsampling strategy
into OME-Zarr multiscales metadata. It builds the rich 0.6 (RFC-5) model --
named coordinate systems and general transforms -- internally, then converts
it to the requested version. These tests pin the field resolution, the 0.6
shape, the version conversion and its loss policy, the ``voxel_to_world``
decomposition, and the per-level strategy math.
"""

import pathlib
import warnings

import numpy as np
import pytest

from abczarr.abc.sync import PathGroup
from abczarr.errors import UnsupportedConversion
from abczarr.metadata.base import GroupMetadataV3
from abczarr.ome import ImageConfig, Transform, axis
from abczarr.ome.base import LATEST_STABLE
from abczarr.ome.v0_6rc0.transformations import Rotation


def _level0(ome: object) -> object:
    """The voxel->intrinsic transform of the finest level."""
    return ome.multiscales[0].datasets[0].coordinateTransformations[0]


def _dataset_scale(transform: object) -> list:
    """The scale of a voxel->intrinsic transform, sequence or bare."""
    if transform.type == "scale":
        return transform.scale
    return transform.transformations[0].scale


def _dataset_translation(transform: object) -> object:
    """The translation of a voxel->intrinsic transform, or `None`."""
    if transform.type == "scale":
        return None
    return transform.transformations[1].translation


# --- axis type inference ---------------------------------------------------


def test_axis_helper_infers_type_from_name() -> None:
    assert axis("x").type == "space"
    assert axis("y").type == "space"
    assert axis("z").type == "space"
    assert axis("t").type == "time"
    assert axis("c").type == "channel"
    assert axis("anything").type == "space"


def test_axis_helper_keeps_an_explicit_type_and_unit() -> None:
    a = axis("x", "time", "second")
    assert a.type == "time"
    assert a.unit == "second"


def test_config_infers_axis_types() -> None:
    cfg = ImageConfig(axes=["t", "c", "z", "y", "x"])
    types = [(a.name, a.type) for a in cfg._axes()]
    assert types == [
        ("t", "time"), ("c", "channel"), ("z", "space"),
        ("y", "space"), ("x", "space"),
    ]


def test_axes_accept_tuples_and_typed_axes() -> None:
    cfg = ImageConfig(axes=["x", ("q", "space", "micrometer"), axis("t")])
    resolved = cfg._axes()
    assert [a.name for a in resolved] == ["x", "q", "t"]
    assert resolved[1].unit == "micrometer"
    assert resolved[2].type == "time"


# --- scale / translation resolution ----------------------------------------


def _scale_of(cfg: ImageConfig) -> list:
    return _dataset_scale(_level0(cfg.to_ome(version="0.6rc0")))


def test_scale_scalar_broadcasts_to_every_axis() -> None:
    assert _scale_of(ImageConfig(axes=["y", "x"], scale=3.0)) == [3.0, 3.0]


def test_scale_sequence_is_per_axis() -> None:
    cfg = ImageConfig(axes=["y", "x"], scale=[2.0, 4.0])
    assert _scale_of(cfg) == [2.0, 4.0]


def test_scale_mapping_by_name_and_type_with_name_winning() -> None:
    cfg = ImageConfig(
        axes=["t", "z", "y", "x"],
        scale={"space": 2.0, "time": 5.0, "x": 9.0},
    )
    assert _scale_of(cfg) == [5.0, 2.0, 2.0, 9.0]


def test_scale_and_translation_default_to_one_and_zero() -> None:
    cfg = ImageConfig(axes=["y", "x"])
    transform = _level0(cfg.to_ome(version="0.6rc0"))
    assert _dataset_scale(transform) == [1.0, 1.0]
    # zero translation collapses to a bare scale, no translation emitted
    assert transform.type == "scale"


def test_translation_makes_a_scale_translation_sequence() -> None:
    cfg = ImageConfig(axes=["y", "x"], translation=[0.0, 5.0])
    transform = _level0(cfg.to_ome(version="0.6rc0"))
    assert transform.type == "sequence"
    assert _dataset_translation(transform) == [0.0, 5.0]


def test_a_per_axis_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError):
        ImageConfig(axes=["y", "x"], scale=[1.0, 2.0, 3.0]).to_ome(
            version="0.6rc0"
        )


# --- the 0.6 (RFC-5) shape -------------------------------------------------


def test_to_ome_06_builds_intrinsic_and_model_systems() -> None:
    cfg = ImageConfig(
        axes=["y", "x"],
        scale=[2.0, 2.0],
        voxel_to_world=np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 5.0], [0, 0, 1]]),
        name="img",
    )
    ome = cfg.to_ome(version="0.6rc0", level_shapes=[[8, 8]])
    assert ome.version == "0.6rc0"
    ms = ome.multiscales[0]
    assert ms.name == "img"
    assert [s.name for s in ms.coordinateSystems] == ["intrinsic", "model"]
    # the intrinsic system carries the config axes
    assert [a.name for a in ms.coordinateSystems[0].axes] == ["y", "x"]


def test_to_ome_06_maps_each_level_onto_the_intrinsic_system() -> None:
    cfg = ImageConfig(axes=["y", "x"], scale=[3.0, 3.0])
    ome = cfg.to_ome(version="0.6rc0", level_shapes=[[8, 8]])
    transform = ome.multiscales[0].datasets[0].coordinateTransformations[0]
    assert transform.input.path == "0"
    assert transform.output.name == "intrinsic"
    assert transform.scale == [3.0, 3.0]


def test_to_ome_06_puts_intrinsic_to_model_at_the_multiscale_level() -> None:
    cfg = ImageConfig(
        axes=["y", "x"],
        scale=[1.0, 1.0],
        transforms=[np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 9.0], [0, 0, 1]])],
    )
    ms = cfg.to_ome(version="0.6rc0").multiscales[0]
    (w,) = ms.coordinateTransformations
    assert w.type == "affine"
    assert w.input.name == "intrinsic"
    assert w.output.name == "model"


def test_a_custom_output_name_gets_its_own_system() -> None:
    cfg = ImageConfig(
        axes=["y", "x"],
        transforms=[Transform(np.eye(3), output="anatomical")],
    )
    ms = cfg.to_ome(version="0.6rc0").multiscales[0]
    names = [s.name for s in ms.coordinateSystems]
    assert names == ["intrinsic", "anatomical"]
    assert ms.coordinateTransformations[0].output.name == "anatomical"


def test_a_typed_transform_is_used_as_is_with_references_filled() -> None:
    rot = Rotation(type="rotation", rotation=[[0.0, -1.0], [1.0, 0.0]])
    cfg = ImageConfig(axes=["y", "x"], transforms=[rot])
    ms = cfg.to_ome(version="0.6rc0").multiscales[0]
    r = ms.coordinateTransformations[0]
    assert r.type == "rotation"
    assert r.rotation == [[0.0, -1.0], [1.0, 0.0]]
    assert r.input.name == "intrinsic"
    assert r.output.name == "model"


# --- version resolution and the 0.5 loss policy ----------------------------


def test_stable_resolves_to_the_latest_released_version() -> None:
    assert ImageConfig(axes=["x"]).resolved_version() == LATEST_STABLE


def test_latest_resolves_to_the_newest_preview() -> None:
    assert ImageConfig(axes=["x"]).resolved_version("latest") == "0.6rc0"


def test_default_lowering_is_the_stable_05_shape() -> None:
    cfg = ImageConfig(axes=["y", "x"], scale=[2.0, 2.0])
    ome = cfg.to_ome(level_shapes=[[8, 8]])
    assert ome.version == "0.5"
    # 0.5 keeps the per-level scale on the dataset and the axes on the
    # multiscale
    ms = ome.multiscales[0]
    assert [a.name for a in ms.axes] == ["y", "x"]
    assert ms.datasets[0].coordinateTransformations[0].scale == [2.0, 2.0]


def _rich_config() -> ImageConfig:
    return ImageConfig(
        axes=["y", "x"],
        scale=[2.0, 2.0],
        voxel_to_world=np.array(
            [[2.0, 0.0, 3.0], [0.0, 2.0, 5.0], [0, 0, 1]]
        ),
    )


def test_rich_affine_is_dropped_with_one_warning_going_to_05() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _rich_config().to_ome(level_shapes=[[8, 8]])
    assert len(caught) == 1
    assert "affine" in str(caught[0].message)


def test_rich_affine_is_preserved_at_06() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no loss, so no warning
        ome = _rich_config().to_ome(version="0.6rc0", level_shapes=[[8, 8]])
    assert ome.multiscales[0].coordinateTransformations[0].type == "affine"


def test_rich_affine_raises_going_to_05_under_strict() -> None:
    with pytest.raises(UnsupportedConversion):
        _rich_config().to_ome(policy="strict", level_shapes=[[8, 8]])


# --- voxel_to_world decomposition ------------------------------------------


def test_voxel_to_world_without_translation() -> None:
    vox2world = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cfg = ImageConfig(axes=["z", "y", "x"], voxel_to_world=vox2world)
    ome = cfg.to_ome(version="0.6rc0", level_shapes=[[4, 4, 4]])
    ms = ome.multiscales[0]

    voxel_to_intrinsic = ms.datasets[0].coordinateTransformations[0]
    # scale is derived from the matrix's column norms
    assert voxel_to_intrinsic.type == "scale"
    assert voxel_to_intrinsic.scale == [2.0, 3.0, 4.0]

    intrinsic_to_world = ms.coordinateTransformations[0]
    assert intrinsic_to_world.output.name == "model"
    # with no explicit translation, the offset lands in intrinsic->world
    assert [row[-1] for row in intrinsic_to_world.affine] == [10.0, 20.0, 30.0]

    # composed voxel->world equals the input matrix, numerically
    n = 3
    v = np.eye(n + 1)
    for i in range(n):
        v[i, i] = voxel_to_intrinsic.scale[i]
    w = np.eye(n + 1)
    w[:n, :] = np.array(intrinsic_to_world.affine)
    assert np.allclose(w @ v, vox2world)


def test_voxel_to_world_with_explicit_translation_stays_exact() -> None:
    vox2world = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0, 0, 1]])
    cfg = ImageConfig(
        axes=["y", "x"], scale=[2.0, 2.0], translation=[1.0, 1.0],
        voxel_to_world=vox2world,
    )
    ome = cfg.to_ome(version="0.6rc0", level_shapes=[[4, 4]])
    ms = ome.multiscales[0]
    voxel_to_intrinsic = ms.datasets[0].coordinateTransformations[0]
    # the explicit translation lands in voxel->intrinsic here
    assert voxel_to_intrinsic.type == "sequence"
    assert _dataset_translation(voxel_to_intrinsic) == [1.0, 1.0]

    n = 2
    v = np.eye(n + 1)
    for i in range(n):
        v[i, i] = _dataset_scale(voxel_to_intrinsic)[i]
        v[i, n] = _dataset_translation(voxel_to_intrinsic)[i]
    w = np.eye(n + 1)
    w[:n, :] = np.array(ms.coordinateTransformations[0].affine)
    assert np.allclose(w @ v, vox2world)


# --- per-level strategy math -----------------------------------------------


def _level_scales_translations(cfg: ImageConfig, **kwargs: object) -> tuple:
    ome = cfg.to_ome(version="0.6rc0", **kwargs)
    transforms = [
        d.coordinateTransformations[0] for d in ome.multiscales[0].datasets
    ]
    return (
        [_dataset_scale(t) for t in transforms],
        [_dataset_translation(t) for t in transforms],
    )


def test_edge_strategy_matches_the_formula() -> None:
    cfg = ImageConfig(axes=["x"], scale=1.0, factor=2, strategy="edge")
    scales, trans = _level_scales_translations(
        cfg, level_shapes=[[8], [4], [2]]
    )
    # scale = s0 * (N/N'); translation = t0 + s0 * (N/N' - 1) / 2
    assert scales == [[1.0], [2.0], [4.0]]
    assert trans == [None, [0.5], [1.5]]


def test_center_strategy_matches_the_formula() -> None:
    cfg = ImageConfig(axes=["x"], scale=1.0, factor=2, strategy="center")
    scales, trans = _level_scales_translations(
        cfg, level_shapes=[[9], [5], [3]]
    )
    # scale = s0 * (N-1)/(N'-1); translation = t0
    assert scales == [[1.0], [2.0], [4.0]]
    assert trans == [None, None, None]


def test_window_strategy_matches_the_formula() -> None:
    cfg = ImageConfig(axes=["x"], scale=1.0, factor=2, strategy="window")
    scales, trans = _level_scales_translations(
        cfg, level_shapes=[[8], [4], [2]]
    )
    # scale = s0 * F; translation = t0 + s0 * (F - 1) / 2, F = factor ** level
    assert scales == [[1.0], [2.0], [4.0]]
    assert trans == [None, [0.5], [1.5]]


def test_window_needs_no_shapes_and_uses_the_paths() -> None:
    cfg = ImageConfig(axes=["x"], scale=1.0, factor=3, strategy="window")
    scales, trans = _level_scales_translations(
        cfg, level_paths=["0", "1", "2"]
    )
    assert scales == [[1.0], [3.0], [9.0]]
    assert trans == [None, [1.0], [4.0]]


def test_an_int_strategy_is_a_window_size() -> None:
    cfg = ImageConfig(axes=["x"], scale=1.0, strategy=2)
    scales, _ = _level_scales_translations(cfg, level_paths=["0", "1", "2"])
    assert scales == [[1.0], [2.0], [4.0]]


def test_an_unshaped_edge_pyramid_is_level_zero_only() -> None:
    cfg = ImageConfig(axes=["x"], scale=2.0, strategy="edge")
    ome = cfg.to_ome(version="0.6rc0")
    assert len(ome.multiscales[0].datasets) == 1


def test_a_factor_one_axis_is_not_downsampled() -> None:
    cfg = ImageConfig(axes=["c", "x"], scale=1.0, factor={"c": 1, "x": 2},
                      strategy="window")
    scales, _ = _level_scales_translations(cfg, level_paths=["0", "1"])
    assert scales == [[1.0, 1.0], [1.0, 2.0]]


# --- round trip through a group --------------------------------------------


def test_apply_and_read_back(tmp_path: pathlib.Path) -> None:
    root = pathlib.Path(tmp_path) / "image.zarr"
    GroupMetadataV3(attributes={}).to_file(root)
    group = PathGroup(str(root))

    cfg = ImageConfig(axes=["y", "x"], scale=[2.0, 2.0], name="img")
    written = cfg.apply(group, version="0.6rc0", level_shapes=[[8, 8]])

    reopened = PathGroup(group.store_path)
    assert reopened.ome == written
    assert reopened.ome.version == "0.6rc0"
