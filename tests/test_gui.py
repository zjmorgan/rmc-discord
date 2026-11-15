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
    assert w.session.crystal is not None
    labels = [sv.site_table.item(i, 1).text() for i in range(sv.site_table.rowCount())]
    assert {"C1", "C2", "O7", "O8"} <= set(labels)
    assert w.workflow.item(1).flags() & Qt.ItemIsEnabled

    for i, label in enumerate(labels):
        if label in ("C1", "C2"):
            sv.site_table.item(i, 0).setCheckState(Qt.Checked)
        if label == "C1":
            sv.site_table.item(i, 3).setText("0.9")
    sv.cell_lines[2].setText("9.6")
    assert sv.get_active_labels() == ["C1", "C2"]
    occupancies = sv.get_occupancies()
    assert occupancies["C1"] == 0.9

    result = w.structure.model.build_supercell(sv.get_cell(), occupancies, sv.get_active_labels(), (2, 2, 3),
                                               {"C1": ["O7"], "C2": ["O8"]}, {}, {}, 1, True)
    w.structure.build_complete(result)
    sc = w.session.supercell
    assert sc is not None and sc.N == (2, 2, 3)
    assert np.isclose(w.session.crystal.cell[2], 9.6)
    assert sv.kind_table.rowCount() == len(sc.kinds)
    fractions = {k.label: k.fraction for k in sc.kinds}
    assert np.isclose(fractions["C1"], 0.9)
    assert all(k.group for k in sc.kinds if not k.vacancy)
    assert w.workflow.item(0).toolTip() == "Done"
    assert sv.supercell_lines[0].text() == "2"


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


def test_refinement_workflow(window, tmp_path):
    from qtpy.QtCore import Qt

    from rmc_discord.data.sampling import QSample
    from rmc_discord.gui.models.refinement import schedule_from

    w = window
    sc = w.session.supercell
    assert sc is not None, "runs after test_structure_workflow"
    N = np.array(sc.N)
    m = np.stack(np.meshgrid(*[np.arange(-2 * n, 2 * n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    m = m[~np.all(m % N == 0, axis=1)]
    hkl = m / N
    y = np.random.default_rng(0).random(len(m)) + 1
    sample = QSample(m=m, N=tuple(N), hkl=hkl, Q=w.session.crystal.Q(hkl), y=y, variance=0.01 * y,
                     bins=np.ones(len(m), dtype=int))
    w.session.set_sample(None, sample)
    assert w.workflow.item(2).flags() & Qt.ItemIsEnabled

    rv = w.refinement_view
    assert rv.displacive_list.count() == sc.n_sites
    assert float(rv.weight_lines["swap"].text()) == 1.0
    for i in range(rv.displacive_list.count()):
        rv.displacive_list.item(i).setCheckState(Qt.Checked)
    rv.weight_lines["exchange"].setText("1")
    rv.strata_combo.setCurrentText("Planes")
    rv.planes_line.setText("0.5, 1.5")
    rv.plane_width_line.setText("0.2")
    st = rv.get_settings()
    assert st.displacive == list(range(sc.n_sites)) and st.planes == (2, [0.5, 1.5], 0.2)

    model = w.refinement.model
    result = model.initialize(st)
    w.refinement.initialize_complete(result)
    assert set(w.session.held) == {"on", "off"} and w.session.refinement is not None
    assert "Pool" in rv.init_info.text()

    values = rv.get_schedule_values()
    values[:9] = [4, 1.0, 0.9, 1.0, None, 1.0, 2, None, 0.25]
    schedule = schedule_from(values)
    assert schedule.batch is None and schedule.temperature == 1.0
    out = model.run(schedule)
    assert [r["round"] for r in model.records] == [1, 3]
    w.refinement.progress("", 50)
    w.refinement.run_complete(out)
    assert w.session.result is out and rv.record_table.rowCount() == 2
    assert w.workflow.item(2).toolTip() == "Done"

    model.records = []
    model.stop_event.set()
    model.stop_event.clear()
    calls = []
    original = model.stop_event.is_set
    model.stop_event.is_set = lambda: (calls.append(1), len(calls) >= 1)[1]
    try:
        model.run(schedule_from([10, 1.0, 0.9, 1.0, None, 1.0, 5, None, 0.25]))
    finally:
        model.stop_event.is_set = original
    assert [r["round"] for r in model.records] == [0]

    path = tmp_path / "config.npz"
    kind = sc.kind.copy()
    model.save_configuration(path)
    sc.kind[:] = sc.kind[::-1]
    model.load_configuration(path)
    assert np.array_equal(sc.kind, kind)


def test_correlations_and_recalculation(window):
    from rmc_discord.data.volume import Volume

    w = window
    sc = w.session.supercell
    assert sc is not None, "runs after the refinement test"
    cm = w.correlations.model
    assert cm.available() == ["occupancy", "displacement"]
    assert w.correlations_view.quantity_combo.count() == 2

    out = cm.compute("occupancy", True, True, True)
    w.correlations.compute_complete(out)
    assert "occupancy" in w.session.correlations and not w.session.correlations["occupancy"].normalized
    assert cm.shown.normalized
    pairs = cm.pair_names()
    assert pairs and w.correlations_view.pair_combo.count() == len(pairs) + 1
    shells = cm.shells("All pairs", 0, 10)
    assert shells is not None and np.all(shells["distance"] > 0)
    assert w.correlations_view.shell_table.rowCount() > 0
    c = cm.filtered(pairs[0], 0, 10)
    assert all("-".join(sorted(l.split("-"))) == pairs[0] for l in c.labels)

    n = 24
    edges = [np.linspace(-1.5, 1.5, n + 1), np.linspace(-1.5, 1.5, n + 1), np.linspace(0, 1, 5)]
    W = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    signal = np.random.default_rng(1).random((n, n, 4)) + 1
    w.session.volume = Volume(signal=signal, variance=signal, edges=edges, W=W)
    w.session.correlations_changed.emit()
    rv = w.recalculation_view
    assert rv.quantity_combo.count() == 1 and rv.r_max_line.text()
    rm = w.recalculation.model
    r_max = 0.99 * rm.max_range()
    res = rm.recalculate("occupancy", 2, 0.4, r_max, 0, 100, 0.12, 2, "neutron")
    assert res["data"].shape == (n, n) and np.isfinite(res["r"])
    assert np.all(np.isfinite(res["model"]) == np.isfinite(res["data"]))
    assert np.isclose(res["transform"][1], 1 / np.sqrt(3))
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

    window.refinement_view.stop_process_button.click()
    assert window.refinement.model.stop_event.is_set()
    window.refinement.model.stop_event.clear()


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
