"""
Tutorial screenshots: magnetic correlations of bixbyite (Fe,Mn)2O3, CORELLI.

The settings follow ``examples/bixbyite.py``: the 6 K - 300 K difference,
spins only on the Fe/Mn sites, a paramagnetic offset, a 10 x 10 x 10
supercell, and 10% of the points held out.
"""

from qtpy.QtCore import Qt

from utilities import DATA, check, click, run_qt_scenario, screenshot, wait

ROOT = DATA / "bixbyite_CORELLI"
PREFIX = "Bixbyite"


def bixbyite(app, window, errors):
    w = window

    # 1. Structure --------------------------------------------------------
    w.workflow.setCurrentRow(0)
    sv = w.structure_view
    sv.load_cif_file_dialog = lambda: str(ROOT / "bixbyite.cif")
    click(app, w, sv.load_cif_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_structure_cif")

    for i in range(sv.site_table.rowCount()):
        if sv.site_table.item(i, 1).text() in ("Fe1", "Fe2"):
            sv.site_table.item(i, 0).setCheckState(Qt.Checked)
    sv.tab_widget.setCurrentIndex(1)
    sv.ions_line.setText("Fe1: Fe3+; Mn1: Mn3+; Fe2: Fe3+; Mn2: Mn3+")
    sv.moments_line.setText("Fe3+: 5.92; Mn3+: 4.90")
    for spin in sv.N_spins:
        spin.setValue(10)
    sv.seed_spin.setValue(0)
    sv.displace_box.setChecked(False)
    click(app, w, sv.build_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_structure_supercell")

    # 2. Data -------------------------------------------------------------
    w.workflow.setCurrentRow(1)
    dv = w.data_view
    dv.file_lines["data"].setText(str(next(ROOT.glob("bixbyite_006K_*.nxs"))))
    dv.file_lines["subtract"].setText(str(next(ROOT.glob("bixbyite_300K_*.nxs"))))
    click(app, w, dv.load_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_data_load")

    for line in dv.bragg_lines:
        line.setText("0.15")
    dv.f_sys_line.setText("0.03")
    click(app, w, dv.sample_button)
    check(errors)
    dv.tab_widget.setCurrentIndex(1)
    dv.points_box.setChecked(True)
    screenshot(app, w, f"{PREFIX}_data_sample")

    # 3. Refinement -------------------------------------------------------
    w.workflow.setCurrentRow(2)
    rv = w.refinement_view
    rv.tab_widget.setCurrentIndex(0)
    rv.nuclear_box.setChecked(False)
    rv.magnetic_box.setChecked(True)
    rv.offset_combo.setCurrentText("Paramagnetic")
    for key, line in rv.weight_lines.items():
        line.setText("1" if key == "spin" else "0")
    for i in range(rv.displacive_list.count()):
        rv.displacive_list.item(i).setCheckState(Qt.Unchecked)
    rv.cone_line.setText("60")
    rv.background_combo.setCurrentText("Constant")
    rv.symmetrize_line.setText("")
    rv.strata_combo.setCurrentText("None")
    rv.held_line.setText("0.1")
    rv.seed_line.setText("1")
    click(app, w, rv.init_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_refinement_setup")

    rv.tab_widget.setCurrentIndex(1)
    for (_, line), value in zip(rv.schedule_lines, ["40", "2", "0.9", "", "100000", "1", "5", "0.3", ""]):
        line.setText(value)
    click(app, w, rv.run_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_refinement_anneal")

    # 4. Correlations -----------------------------------------------------
    w.workflow.setCurrentRow(3)
    cv = w.correlations_view
    cv.tab_widget.setCurrentIndex(0)
    cv.quantity_combo.setCurrentIndex(cv.quantity_combo.findData("spin"))
    click(app, w, cv.compute_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_correlations_compute")
    cv.tab_widget.setCurrentIndex(1)
    screenshot(app, w, f"{PREFIX}_correlations_shells")

    # 5. Recalculation ----------------------------------------------------
    w.workflow.setCurrentRow(4)
    xv = w.recalculation_view
    xv.quantity_combo.setCurrentIndex(xv.quantity_combo.findData("spin"))
    xv.normal_combo.setCurrentText("Axis 1/2")
    xv.slice_line.setText("0")
    click(app, w, xv.recalc_button)
    check(errors)
    wait(app, w)
    screenshot(app, w, f"{PREFIX}_recalculation")


if __name__ == "__main__":
    run_qt_scenario(bixbyite)
