"""Crash-safe writes, run-directory layout, and configurable sample storage (M8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import dense_polytope

from gsmm_compiler.affine_geometry import build_geometry
from gsmm_compiler.flux_polytope import ReducedPolytope
from gsmm_compiler.maxent_sampler import ChainDiagnostics, ChainResult, ObjectiveTrace
from gsmm_compiler.output import (
    COMPLETE_MARKER,
    LoadedChain,
    OutputError,
    RunLayout,
    SampleStorage,
    atomic_write_bytes,
    is_complete,
    load_array,
    load_chain,
    read_json,
    save_array,
    staged_directory,
    store_chain,
    write_json,
)
from gsmm_compiler.rounding import build_transform

# ---- factories for the sampler's result objects (built by hand, small) --------------------------


def _chain(
    coordinates: np.ndarray, fluxes: np.ndarray, *, beta: float = 0.0, chain_index: int = 0
) -> ChainResult:
    diagnostics = ChainDiagnostics(
        chain_index=chain_index,
        beta=beta,
        dimension=int(coordinates.shape[1]),
        n_sweeps=10,
        n_samples=int(coordinates.shape[0]),
        n_degenerate_steps=0,
        n_refreshes=1,
        max_refresh_drift=0.0,
        max_bound_violation=0.0,
        max_mass_balance_residual=0.0,
        mean_chord_length=1.0,
        start_shrink=1.0,
        spawn_key=(1, 2, 0, chain_index),
    )
    return ChainResult(
        coordinates=np.ascontiguousarray(coordinates, dtype=np.float64),
        fluxes=np.ascontiguousarray(fluxes, dtype=np.float64),
        diagnostics=diagnostics,
    )


def _trace(n_samples: int, n_free: int) -> ObjectiveTrace:
    thresholds = (1e-6, 1e-3)
    return ObjectiveTrace(
        mu=np.linspace(0.0, 1.0, n_samples),
        cost=np.linspace(1.0, 2.0, n_samples),
        j=np.linspace(-1.0, 0.0, n_samples),
        normalized_log_energy=np.linspace(-2.0, -1.0, n_samples),
        near_zero_thresholds=thresholds,
        near_zero_counts=np.zeros((n_samples, len(thresholds)), dtype=np.int64),
        near_zero_counts_all_free=np.ones((n_samples, len(thresholds)), dtype=np.int64),
        n_free=n_free,
        n_movable=n_free,
        n_blocked=0,
    )


# ---- atomic file writes ------------------------------------------------------------------------


class TestAtomicWrites:
    def test_bytes_roundtrip_and_leave_no_temp_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "data.bin"
        atomic_write_bytes(target, b"hello")

        assert target.read_bytes() == b"hello"
        assert list(target.parent.glob("*.tmp")) == []

    def test_overwrite_replaces_the_old_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        atomic_write_bytes(target, b"first")
        atomic_write_bytes(target, b"second")
        assert target.read_bytes() == b"second"

    def test_json_roundtrip_coerces_numpy_scalars(self, tmp_path: Path) -> None:
        target = tmp_path / "manifest.json"
        write_json(target, {"n": np.int64(7), "x": np.float64(1.5), "path": tmp_path})

        reloaded = read_json(target)
        assert reloaded["n"] == 7
        assert reloaded["x"] == 1.5
        assert reloaded["path"] == str(tmp_path)

    def test_read_json_on_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OutputError, match="found nothing"):
            read_json(tmp_path / "absent.json")


# ---- validated array persistence ---------------------------------------------------------------


class TestArrayPersistence:
    def test_roundtrip_and_hash_is_stable(self, tmp_path: Path) -> None:
        array = np.arange(12, dtype=np.float64).reshape(3, 4)
        digest = save_array(tmp_path / "a.npy", array)

        loaded = load_array(tmp_path / "a.npy", sha256=digest, dtype="<f8", shape=(3, 4))
        assert np.array_equal(loaded, array)

    def test_load_rejects_a_dtype_mismatch(self, tmp_path: Path) -> None:
        save_array(tmp_path / "a.npy", np.arange(4, dtype=np.float64))
        with pytest.raises(OutputError, match="dtype"):
            load_array(tmp_path / "a.npy", dtype="<f4")

    def test_load_rejects_a_shape_mismatch(self, tmp_path: Path) -> None:
        save_array(tmp_path / "a.npy", np.arange(4, dtype=np.float64))
        with pytest.raises(OutputError, match="shape"):
            load_array(tmp_path / "a.npy", shape=(2, 2))

    def test_load_rejects_non_finite_floats(self, tmp_path: Path) -> None:
        save_array(tmp_path / "a.npy", np.array([1.0, np.inf], dtype=np.float64))
        with pytest.raises(OutputError, match="NaN or infinity"):
            load_array(tmp_path / "a.npy")

    def test_load_rejects_a_corrupted_file_by_its_hash(self, tmp_path: Path) -> None:
        digest = save_array(tmp_path / "a.npy", np.zeros(4, dtype=np.float64))
        # A bit-flip that keeps the shape and dtype — only the content hash can catch it.
        np.save(tmp_path / "a.npy", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64))

        with pytest.raises(OutputError, match="content hash mismatch"):
            load_array(tmp_path / "a.npy", sha256=digest)

    def test_integer_arrays_skip_the_finite_check(self, tmp_path: Path) -> None:
        counts = np.array([[0, 1], [2, 3]], dtype=np.int64)
        digest = save_array(tmp_path / "c.npy", counts)
        loaded = load_array(tmp_path / "c.npy", sha256=digest, dtype="<i8")
        assert np.array_equal(loaded, counts)


# ---- atomic directory publication --------------------------------------------------------------


class TestStagedDirectory:
    def test_publishes_with_a_complete_marker_last(self, tmp_path: Path) -> None:
        final = tmp_path / "unit"
        with staged_directory(final) as staging:
            atomic_write_bytes(staging / "payload.bin", b"x")

        assert is_complete(final)
        assert (final / "payload.bin").read_bytes() == b"x"
        assert (final / COMPLETE_MARKER).is_file()
        assert not (tmp_path / "unit.staging").exists()

    def test_a_failure_leaves_no_final_directory(self, tmp_path: Path) -> None:
        final = tmp_path / "unit"
        with pytest.raises(RuntimeError, match="boom"), staged_directory(final) as staging:
            atomic_write_bytes(staging / "payload.bin", b"x")
            raise RuntimeError("boom")

        assert not final.exists()
        assert not is_complete(final)

    def test_retry_overwrites_a_stale_partial(self, tmp_path: Path) -> None:
        final = tmp_path / "unit"
        # A crash could leave a directory with no COMPLETE marker. A retry must replace it cleanly.
        final.mkdir()
        (final / "old.bin").write_text("stale")

        with staged_directory(final) as staging:
            atomic_write_bytes(staging / "new.bin", b"fresh")

        assert is_complete(final)
        assert (final / "new.bin").read_bytes() == b"fresh"
        assert not (final / "old.bin").exists()  # the stale partial was cleared

    def test_sibling_units_in_one_parent_do_not_clobber_each_other(self, tmp_path: Path) -> None:
        parent = tmp_path / "beta_000"
        with staged_directory(parent / "chain_000") as a:
            atomic_write_bytes(a / "v.bin", b"a")
        with staged_directory(parent / "chain_001") as b:
            atomic_write_bytes(b / "v.bin", b"b")

        assert (parent / "chain_000" / "v.bin").read_bytes() == b"a"
        assert (parent / "chain_001" / "v.bin").read_bytes() == b"b"


# ---- run-directory layout ----------------------------------------------------------------------


class TestRunLayout:
    def test_paths_are_where_the_spec_puts_them(self, tmp_path: Path) -> None:
        layout = RunLayout(root=tmp_path, batch="run42")

        assert layout.batch_dir == tmp_path / "run42"
        assert layout.model_dir("strainA") == tmp_path / "run42" / "strainA"
        assert layout.cross_model_dir() == tmp_path / "run42" / "cross_model"
        assert layout.chain_dir("strainA", 2, 3) == (
            tmp_path / "run42" / "strainA" / "samples" / "beta_002" / "chain_003"
        )

    @pytest.mark.parametrize("bad", ["..", "a/b", "", "."])
    def test_unsafe_model_ids_are_refused(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(OutputError, match="safe path component"):
            RunLayout(root=tmp_path, batch="run").model_dir(bad)

    def test_an_unsafe_batch_name_is_refused_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(OutputError, match="safe path component"):
            RunLayout(root=tmp_path, batch="../escape")

    def test_is_chain_complete_tracks_the_marker(self, tmp_path: Path) -> None:
        layout = RunLayout(root=tmp_path, batch="run")
        assert not layout.is_chain_complete("m", 0, 0)

        with staged_directory(layout.chain_dir("m", 0, 0)) as staging:
            atomic_write_bytes(staging / "x.bin", b"x")
        assert layout.is_chain_complete("m", 0, 0)


# ---- storage policy ----------------------------------------------------------------------------


class TestSampleStorage:
    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(OutputError, match="store_mode"):
            SampleStorage(mode="dense")

    def test_rejects_an_unknown_dtype(self) -> None:
        with pytest.raises(OutputError, match="store_flux_dtype"):
            SampleStorage(mode="full_flux", flux_dtype="float16")


# ---- store / load one chain --------------------------------------------------------------------


@pytest.fixture(scope="module")
def box() -> ReducedPolytope:
    """``{v0 = v1 ∈ [0, 2], v2 ∈ [−1, 3]}`` — small, fully free, ``d = 2``, ``n_free = 3``."""
    return dense_polytope(
        stoichiometry=[[1.0, -1.0, 0.0]],
        lower=[0.0, 0.0, -1.0],
        upper=[2.0, 2.0, 3.0],
    )


@pytest.fixture(scope="module")
def box_transform(box: ReducedPolytope):  # type: ignore[no-untyped-def]
    geometry = build_geometry(box, model_id="box")
    return build_transform(geometry, box)


def _box_chain_and_trace(box: ReducedPolytope, transform):  # type: ignore[no-untyped-def]
    """A chain whose reduced fluxes are the exact lift of its coordinates, so both storage modes
    must reconstruct the *same* full-length fluxes."""
    n_samples = 4
    coordinates = np.array([[0.1, -0.2], [0.0, 0.0], [0.3, 0.1], [-0.1, 0.05]], dtype=np.float64)
    coordinates = coordinates[:, : transform.dimension]
    reduced_flux = transform.to_flux(coordinates)  # (n_samples, n_free)
    chain = _chain(coordinates, reduced_flux)
    trace = _trace(n_samples, box.n_free)
    return chain, trace


class TestStoreLoadChain:
    def test_full_flux_float64_roundtrip(self, tmp_path: Path, box, box_transform) -> None:  # type: ignore[no-untyped-def]
        chain, trace = _box_chain_and_trace(box, box_transform)
        expected_full = box.to_full(chain.fluxes)

        manifest = store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("full_flux", "float64"),
        )
        assert manifest["store_mode"] == "full_flux"
        assert manifest["n_full"] == box.n_full
        assert manifest["polytope_key"] == box.content_key()

        loaded = load_chain(tmp_path / "chain")
        assert isinstance(loaded, LoadedChain)
        assert loaded.fluxes.shape == (4, box.n_full)
        np.testing.assert_allclose(loaded.fluxes, expected_full)
        np.testing.assert_allclose(loaded.traces["j"], trace.j)
        np.testing.assert_array_equal(loaded.traces["near_zero_counts_all_free"], np.ones((4, 2)))

    def test_full_flux_float32_halves_the_width_but_stays_close(
        self, tmp_path: Path, box, box_transform  # type: ignore[no-untyped-def]
    ) -> None:
        chain, trace = _box_chain_and_trace(box, box_transform)
        expected_full = box.to_full(chain.fluxes)

        store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("full_flux", "float32"),
        )
        stored = np.load(tmp_path / "chain" / "flux.npy")
        assert stored.dtype == np.float32  # width narrowed on disk

        loaded = load_chain(tmp_path / "chain")
        assert loaded.fluxes.dtype == np.float64  # computation width restored
        np.testing.assert_allclose(loaded.fluxes, expected_full, rtol=1e-6, atol=1e-6)

    def test_reduced_mode_reconstructs_the_same_fluxes_as_full_flux(
        self, tmp_path: Path, box, box_transform  # type: ignore[no-untyped-def]
    ) -> None:
        chain, trace = _box_chain_and_trace(box, box_transform)
        expected_full = box.to_full(chain.fluxes)

        store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("reduced", "float64"),
        )
        # Reduced mode stores only the coordinates — a smaller artifact than the full flux.
        assert (tmp_path / "chain" / "coordinates.npy").is_file()
        assert not (tmp_path / "chain" / "flux.npy").exists()

        loaded = load_chain(tmp_path / "chain", reduced=box, transform=box_transform)
        np.testing.assert_allclose(loaded.fluxes, expected_full, rtol=1e-9, atol=1e-9)

    def test_reduced_mode_needs_the_geometry_to_load(
        self, tmp_path: Path, box, box_transform  # type: ignore[no-untyped-def]
    ) -> None:
        chain, trace = _box_chain_and_trace(box, box_transform)
        store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("reduced", "float64"),
        )
        with pytest.raises(OutputError, match="needs the geometry"):
            load_chain(tmp_path / "chain")

    def test_reduced_mode_refuses_the_wrong_geometry(
        self, tmp_path: Path, box, box_transform  # type: ignore[no-untyped-def]
    ) -> None:
        chain, trace = _box_chain_and_trace(box, box_transform)
        store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("reduced", "float64"),
        )
        other = dense_polytope(
            stoichiometry=[[1.0, -1.0, 0.0]],
            lower=[0.0, 0.0, -1.0],
            upper=[5.0, 5.0, 3.0],  # different bounds ⇒ different content_key
        )
        with pytest.raises(OutputError, match="polytope mismatch"):
            load_chain(tmp_path / "chain", reduced=other, transform=box_transform)

    def test_a_corrupted_flux_array_is_rejected_on_load(
        self, tmp_path: Path, box, box_transform  # type: ignore[no-untyped-def]
    ) -> None:
        chain, trace = _box_chain_and_trace(box, box_transform)
        store_chain(
            tmp_path / "chain",
            chain=chain,
            trace=trace,
            reduced=box,
            model_id="box",
            beta=0.0,
            beta_index=0,
            chain_index=0,
            storage=SampleStorage("full_flux", "float64"),
        )
        flux_path = tmp_path / "chain" / "flux.npy"
        tampered = np.load(flux_path)
        tampered[0, 0] += 1.0
        np.save(flux_path, tampered)  # same shape/dtype, different content

        with pytest.raises(OutputError, match="content hash mismatch"):
            load_chain(tmp_path / "chain")

    def test_loading_an_unfinished_unit_raises(self, tmp_path: Path) -> None:
        (tmp_path / "chain").mkdir()  # a directory that never got its COMPLETE marker
        with pytest.raises(OutputError, match="not a finished unit"):
            load_chain(tmp_path / "chain")
