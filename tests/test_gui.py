"""Offscreen smoke tests of the graphical interface."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyvistaqt")
pytest.importorskip("qdarkstyle")
pytest.importorskip("qtawesome")

from conftest import LEGACY_DATA  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from rmc_discord.gui.app import MainWindow

    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def test_parse_mapping():
    from rmc_discord.gui.models.structure import format_mapping, parse_mapping

    assert parse_mapping("C1: O7, O9; C2: O8", many=True) == {"C1": ["O7", "O9"], "C2": ["O8"]}
    assert parse_mapping("Fe3+: 5.92; Mn3+: 4.9", number=True) == {"Fe3+": 5.92, "Mn3+": 4.9}
    assert parse_mapping("") == {}
    assert format_mapping({"C1": ["O7", "O9"]}) == "C1: O7, O9"
    with pytest.raises(ValueError):
        parse_mapping("C1 O7")


def test_structure_workflow(window):
    from qtpy.QtCore import Qt

    w = window
    sv = w.structure_view
    assert not w.workflow.item(2).flags() & Qt.ItemIsEnabled
    w.structure.load_cif(str(LEGACY_DATA / "Ba3Co2O6(CO3)0.6.cif"))
    assert w.session.crystal is not None and w.session.structure is not None
    labels = [sv.site_table.item(i, 1).text() for i in range(sv.site_table.rowCount())]
    assert {"C1", "C2", "O7", "O8"} <= set(labels)
    assert w.workflow.item(1).flags() & Qt.ItemIsEnabled
    assert set(sv.get_active_labels()) >= {"C1", "C2"}  # partial sites are checked by default

    for i, label in enumerate(labels):
        sv.site_table.item(i, 0).setCheckState(Qt.Checked if label in ("C1", "C2") else Qt.Unchecked)
        if label == "C1":
            sv.site_table.item(i, 3).setText("0.9")
    sv.cell_lines[2].setText("9.6")
    assert sv.get_active_labels() == ["C1", "C2"]
    occupancies = sv.get_occupancies()
    assert occupancies["C1"] == 0.9
    sv.r_max_spin.setValue(6.0)
    channels, magnetic, r_max, cluster, axis, difference = sv.get_model_parameters()
    assert channels == dict(occupancy=True, size_effect=False, displacement=False) and magnetic is None

    result = w.structure.model.build_model(sv.get_cell(), occupancies, sv.get_active_labels(), channels, magnetic,
                                           {}, {}, r_max, cluster, axis, difference)
    w.structure.build_complete(result)
    model = w.session.model
    assert model is not None and [b.name for b in model.blocks] == ["occupancy"]
    assert np.isclose(w.session.crystal.cell[2], 9.6)
    assert np.allclose(w.session.structure.A, w.session.crystal.A)
    assert sv.model_table.rowCount() == 1 and sv.model_table.item(0, 0).text() == "occupancy"
    assert w.workflow.item(0).toolTip() == "Done"
    assert sv.model_lines[1].text() == "occupancy"

    sv.set_display("Model Sites")
    w.structure.redraw()


def test_magnetic_positions_group_mixed_sites():
    from rmc_discord.crystal.structure import Crystal
    from rmc_discord.gui.models.structure import StructureModel

    crystal = Crystal.from_cif(str(LEGACY_DATA / "Tb2Ir3Ga9.cif"))
    label = crystal.asymmetric[0].label
    labels, mixtures, by_ion = StructureModel.magnetic_positions(crystal, {label: "Fe3+"}, {"Fe3+": 5.0})
    assert labels == [label] and by_ion == {"Fe3+": 5.0}
    assert mixtures[label][0] == (crystal.asymmetric[0].occupancy, "Fe3+")
    with pytest.raises(ValueError):
        StructureModel.magnetic_positions(crystal, {"nope": "Fe3+"}, {"Fe3+": 5.0})


def test_data_slice_metric(window):
    """Slices of a hexagonal [H,H,0]/[-K,K,0] projection have aspect 1/sqrt(3)."""
    from rmc_discord.data.volume import Volume

    w = window
    model = w.data.model
    n = 21
    edges = [np.linspace(-2, 2, n + 1), np.linspace(-2, 2, n + 1), np.linspace(-1, 1, n + 1)]
    W = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    signal = np.random.default_rng(0).random((n, n, n))
    v = Volume(signal=signal, variance=signal, edges=edges, W=W)
    model.set_loaded(dict(data=v, norm=None, subtract=None, display=v, files={}))
    assert model.axis_names() == ["[H,H,0]", "[-K,K,0]", "[0,0,L]"]
    image, ea, eb, axes, (skew, aspect) = model.slice(2, 0.0, 0.1)
    assert axes == (0, 1) and image.shape == (n, n)
    assert np.isclose(skew, 0, atol=1e-12)
    assert np.isclose(aspect, 1 / np.sqrt(3))
    w.data.update_slice()

    planes = model.planes([0.0, 0.0, 0.5])
    assert [p.shape for p, _ in planes] == [(n + 1, n + 1, 3)] * 3
    assert np.allclose(planes[2][1], signal[:, :, np.argmin(np.abs(v.centers[2] - 0.5))])
    q = model.axis_vectors()
    assert np.allclose(planes[2][0][..., :] @ np.linalg.inv(q).T @ [0, 0, 1], v.centers[2][15])
    w.data_view.draw_planes(planes)


def synthetic_volume(model, rng):
    """A volume whose intensity is a known pair model plus a flat background."""
    from rmc_discord.data.volume import Volume
    from rmc_discord.pairs.constraints import stack_parameterizations

    n = 31
    edges = [np.linspace(-1.55, 1.55, n + 1), np.linspace(-1.55, 1.55, n + 1), np.linspace(-1.0, 1.0, 9)]
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
    h = np.stack(np.meshgrid(*centers, indexing="ij"), -1).reshape(-1, 3)
    par = stack_parameterizations(*[b.parameterization for b in model.blocks])
    c = par.c0 + par.N @ (0.02 * rng.normal(size=par.n_free))
    widths = np.array([e[1] - e[0] for e in edges])
    y = model.design(h, widths, np.eye(3)) @ c
    y = y + 0.2 * np.abs(y).mean()
    sigma = 1e-3 * (np.abs(y) + np.abs(y).mean())
    signal = (y + sigma * rng.normal(size=len(y))).reshape(n, n, 8)
    return Volume(signal=signal, variance=(sigma**2).reshape(n, n, 8), edges=edges, W=np.eye(3)), c


def test_refinement_workflow(window):
    from qtpy.QtCore import Qt

    w = window
    model = w.session.model
    assert model is not None, "runs after test_structure_workflow"
    volume, c = synthetic_volume(model, np.random.default_rng(0))
    dm = w.data.model
    dm.set_loaded(dict(data=volume, norm=None, subtract=None, display=volume, files={}))
    dv = w.data_view
    dv.merge_combo.setCurrentIndex(0)
    dv.held_line.setText("0.2")
    dv.f_sys_line.setText("0")  # the default 3% floor would swamp the 0.1% synthetic noise
    params = dv.get_sampling()
    assert params[5] == "mean" and params[7] == 0.2
    obs, summary = dm.sample(*params)
    w.data.sample_complete((obs, summary))
    assert w.session.sample is obs and len(obs) < summary["bins"]  # equivalent bins merged
    assert w.workflow.item(2).flags() & Qt.ItemIsEnabled

    rv = w.refinement_view
    rv.background_combo.setCurrentIndex(2)  # none: the synthetic background is a constant...
    rv.background_combo.setCurrentIndex(1)  # ...which the polynomial basis contains
    rv.degree_line.setText("0")
    st = rv.get_settings()
    assert st.background == "polynomial" and st.degree == 0 and st.ridge == 0
    rm = w.refinement.model
    out = rm.fit(st)
    w.refinement.fit_complete(out)
    fit = w.session.fit
    assert fit is not None and rv.record_table.rowCount() == 1
    assert np.allclose(fit.refinement.c, c, atol=1e-4 + 5 * fit.refinement.esd.max())
    assert 0.5 < fit.chi2_test < 2.0
    assert "occupancy" in w.session.correlations
    assert w.workflow.item(2).toolTip() == "Done"
    rv.compare_value_line.setText("0.125")
    rv.compare_thickness_line.setText("0.25")
    res = rm.compare_slice(2, 0.125, 0.25)
    assert res is not None and res["obs"].shape == res["calc"].shape
    assert np.nanmax(np.abs(res["residual"])) < 6


def test_correlations_and_recalculation(window):
    w = window
    assert w.session.fit is not None, "runs after the refinement test"
    cv, cm = w.correlations_view, w.correlations.model
    assert cm.available() == ["occupancy"] and cv.quantity_combo.count() == 1
    w.correlations.show_channel()
    assert cm.shown.normalized
    raw = w.session.correlations["occupancy"]
    assert not raw.normalized
    pairs = cm.pair_names()
    assert pairs and cv.pair_combo.count() == len(pairs) + 1
    shells = cm.shells("All pairs", 0, 10)
    assert shells is not None and np.all(shells["distance"] > 0) and cv.shell_table.rowCount() > 0
    sub = cm.filtered(pairs[0], 0, 10)
    assert all("-".join(sorted(l.split("-"))) == pairs[0] for l in sub.labels)

    plane = cm.plane("All pairs", [1, 0, 0], 0.0, 0.01, 10)  # contains the c channels
    A, B = w.session.structure.A, w.session.structure.B
    full = cm.filtered("All pairs", 0, 10)
    on = np.abs(full.r @ B @ [1, 0, 0]) <= 0.01
    assert 0 < len(plane["value"]) <= on.sum()
    assert np.allclose(np.hypot(plane["x"], plane["y"]).max(), full.distance[on].max())
    assert np.all(plane["directions"] @ [1, 0, 0] == 0)
    cv.l_line.setText("1")
    w.correlations.update_slice()
    assert cv.slice_plot.fig.axes

    rv = w.recalculation_view
    keys = [rv.quantity_combo.itemData(i) for i in range(rv.quantity_combo.count())]
    assert keys == ["model", "background", "occupancy"]
    rm = w.recalculation.model
    res = rm.recalculate("model", 2, 0.125, 0, 100, 0.12)
    # the plane has little structure compared with the noise, so r is modest; chi2 is the test
    assert res["data"].shape == (31, 31) and res["chi2"] < 1.5 and res["r"] > 0.5
    part = rm.recalculate("occupancy", 2, 0.125, 0, 100, 0.12)
    assert np.all(np.isfinite(part["model"]) == np.isfinite(res["model"]))
    w.recalculation.recalculate_complete(res)
    assert w.workflow.item(4).toolTip() == "Done"


def test_camera_rotations():
    from rmc_discord.gui.models.base_model import BaseModel

    m = BaseModel.__new__(BaseModel)
    pos, fp, up = np.array([0.0, -10.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])

    p, f, u = m.rotate_camera_roll(pos, fp, up, 90)
    assert np.allclose(p, pos) and np.allclose(f, fp)
    assert np.allclose(u, [1, 0, 0])  # up turns to the camera's right: the scene turns counter-clockwise

    p, f, u = m.rotate_camera_elevation(pos, fp, up, 90)
    assert np.allclose(np.linalg.norm(p - f), 10)
    assert np.allclose((f - p) / 10, [0, 0, 1]) and np.allclose(u, [0, -1, 0])

    p, f, u = m.rotate_camera_azimuth(pos, fp, up, 90)
    assert np.allclose(u, up) and np.isclose(np.dot(p, up), 0) and np.allclose(np.linalg.norm(p), 10)

    roll, elev, az = m.calculate_camera_parameters([0, 1, 1], 5.0)
    assert roll == 5.0 and np.isclose(elev, 45) and np.isclose(az, 0)


def test_console_and_stop(window):
    from qtpy.QtWidgets import QApplication

    view = window.structure_view

    def task(progress=None):
        print("hello from a worker")
        return 1

    worker = view.worker(task)
    done = []
    worker.connect_result(done.append)
    view.start_worker_pool(worker)
    view.threadpool.waitForDone(10_000)
    QApplication.processEvents()
    assert done == [1]
    assert "hello from a worker" in view.console.toPlainText()

    view.cons_box.setChecked(False)
    assert view.console.isHidden()
    view.cons_box.setChecked(True)

    window.refinement_view.stop_process_button.click()  # fits run to completion; stop is harmless


def test_theme_and_view_controls(window):
    import pyvista as pv
    from rmc_discord.gui import theme

    s, d = window.structure_view, window.data_view
    s.theme_combo.setCurrentText("dark")
    assert theme.current == "dark" and d.theme_combo.currentText() == "dark"
    assert pv.global_theme.background.hex_rgb == "#000000"
    s.theme_combo.setCurrentText("default")
    assert theme.current == "default" and d.theme_combo.currentText() == "default"

    s.reset_view()
    s.rotate_step_line.setText("30")
    d0 = np.array(s.plotter.camera.direction)
    s.az_right_button.click()
    d1 = np.array(s.plotter.camera.direction)
    turn = np.degrees(np.arctan2(d1[0], d1[1]) - np.arctan2(d0[0], d0[1]))
    assert np.isclose(abs((turn + 180) % 360 - 180), 30, atol=1e-6) and np.isclose(d0[2], d1[2])
    assert s.camera_pos_line.text().count(",") == 2


def test_refit_after_rebuilding_model(window):
    """Changing the structure or model after a fit must leave Refinement ready for another fit."""
    from qtpy.QtCore import Qt

    w = window
    assert w.session.fit is not None, "runs after the refinement test"
    sv, rv = w.structure_view, w.refinement_view
    sv.r_max_spin.setValue(5.0)
    channels, magnetic, r_max, cluster, axis, difference = sv.get_model_parameters()
    result = w.structure.model.build_model(sv.get_cell(), sv.get_occupancies(), sv.get_active_labels(), channels,
                                           magnetic, {}, {}, r_max, cluster, axis, difference)
    w.structure.build_complete(result)
    assert w.workflow.item(2).flags() & Qt.ItemIsEnabled
    for _ in range(2):  # through the button and worker thread, twice
        assert rv.fit_button.isEnabled()
        rv.fit_button.click()
        _wait(rv)
        assert w.session.fit is not None and w.session.model.spec.r_max == 5.0
        assert rv.fit_button.isEnabled(), "fit button left disabled after a fit"


def _wait(view, timeout=60.0):
    import time

    from qtpy.QtWidgets import QApplication

    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if view.threadpool.activeThreadCount() == 0 and view.fit_button.isEnabled():
            break
        time.sleep(0.02)
    for _ in range(20):
        QApplication.processEvents()
