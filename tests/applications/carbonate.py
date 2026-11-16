"""
Tutorial screenshots: carbonate disorder in Ba3Co2O6(CO3)0.7, CORELLI, 50 K.

The settings follow ``examples/co3.py``: the data lattice, carbonate
occupancies scaled to the measured composition, rigid C1/O7 and C2/O8
groups on an 8 x 8 x 20 supercell, swap and displacement-exchange moves, a
radial background, the extra mirror of the 6/mmm-symmetrized data, and
held-out points on and off the diffuse sheets.
"""

from qtpy.QtCore import Qt

from utilities import DATA, check, click, run_qt_scenario, screenshot, wait

ROOT = DATA / "Ba3Co2O6_CO3_0p7_50K"
PREFIX = "Carbonate"
CELL = ("9.67894", "9.67894", "9.59882", "90", "90", "120")
OCCUPANCY = {"C1": "0.91", "O7": "0.91", "C2": "0.9333", "O8": "0.9333"}  # x 7/6


def carbonate(app, window, errors):
    w = window

    # 1. Structure --------------------------------------------------------
    w.workflow.setCurrentRow(0)
    sv = w.structure_view
    sv.load_cif_file_dialog = lambda: str(next(ROOT.glob("*.cif")))
    click(app, w, sv.load_cif_button)
    check(errors)
    for line, value in zip(sv.cell_lines, CELL):
        line.setText(value)
    for i in range(sv.site_table.rowCount()):
        label = sv.site_table.item(i, 1).text()
        if label in ("C1", "C2"):
            sv.site_table.item(i, 0).setCheckState(Qt.Checked)
        if label in OCCUPANCY:
            sv.site_table.item(i, 3).setText(OCCUPANCY[label])
    screenshot(app, w, f"{PREFIX}_structure_cif")

    sv.tab_widget.setCurrentIndex(1)
    sv.groups_line.setText("C1: O7; C2: O8")
    for spin, n in zip(sv.N_spins, (8, 8, 20)):
        spin.setValue(n)
    sv.seed_spin.setValue(1)
    sv.displace_box.setChecked(True)
    click(app, w, sv.build_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_structure_supercell")

    # 2. Data -------------------------------------------------------------
    w.workflow.setCurrentRow(1)
    dv = w.data_view
    dv.file_lines["data"].setText(str(next(ROOT.glob("*_data.nxs"))))
    dv.file_lines["norm"].setText(str(next(ROOT.glob("*_norm.nxs"))))
    click(app, w, dv.load_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_data_load")

    dv.q_min_line.setText("2.5")
    dv.q_max_line.setText("10")
    for line, value in zip(dv.bragg_lines, ("0.15", "0.15", "0.3")):
        line.setText(value)
    dv.allowed_box.setChecked(False)
    dv.rings_box.setChecked(True)
    dv.f_sys_line.setText("0.03")
    click(app, w, dv.sample_button)
    check(errors)
    dv.tab_widget.setCurrentIndex(1)
    dv.normal_combo.setCurrentText("Axis 1/3")
    dv.points_box.setChecked(True)
    screenshot(app, w, f"{PREFIX}_data_sample")

    # 3. Refinement -------------------------------------------------------
    w.workflow.setCurrentRow(2)
    rv = w.refinement_view
    rv.tab_widget.setCurrentIndex(0)
    rv.nuclear_box.setChecked(True)
    rv.magnetic_box.setChecked(False)
    rv.offset_combo.setCurrentText("None")
    for key, line in rv.weight_lines.items():
        line.setText("1" if key in ("swap", "exchange") else "0")
    for i in range(rv.displacive_list.count()):
        rv.displacive_list.item(i).setCheckState(Qt.Checked)
    rv.penalty_line.setText("10")
    rv.background_combo.setCurrentText("Radial")
    rv.knot_line.setText("0.5")
    rv.symmetrize_line.setText("k,h,l")
    rv.seed_line.setText("2")
    rv.strata_combo.setCurrentText("Planes")
    rv.plane_axis_combo.setCurrentIndex(2)
    rv.planes_line.setText("0.8, 2.85, 4.9")
    rv.plane_width_line.setText("0.1")
    rv.n_held_line.setText("50000")
    rv.ratio_line.setText("3")
    click(app, w, rv.init_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_refinement_setup")

    rv.tab_widget.setCurrentIndex(1)
    for (_, line), value in zip(rv.schedule_lines, ["60", "4", "0.92", "0.01", "100000", "2", "5", "", "0.25"]):
        line.setText(value)
    click(app, w, rv.run_button)
    check(errors)
    screenshot(app, w, f"{PREFIX}_refinement_anneal")

    # 4. Correlations -----------------------------------------------------
    w.workflow.setCurrentRow(3)
    cv = w.correlations_view
    for quantity in ("occupancy", "displacement"):
        cv.tab_widget.setCurrentIndex(0)
        cv.quantity_combo.setCurrentIndex(cv.quantity_combo.findData(quantity))
        click(app, w, cv.compute_button)
        check(errors)
        if quantity == "occupancy":
            screenshot(app, w, f"{PREFIX}_correlations_compute")
        cv.tab_widget.setCurrentIndex(1)
        screenshot(app, w, f"{PREFIX}_correlations_{quantity}")

    # 5. Recalculation ----------------------------------------------------
    w.workflow.setCurrentRow(4)
    xv = w.recalculation_view
    xv.quantity_combo.setCurrentIndex(xv.quantity_combo.findData("occupancy"))
    xv.normal_combo.setCurrentText("Axis 1/3")
    xv.slice_line.setText("0")
    xv.q_min_line.setText("2.5")
    xv.q_max_line.setText("10")
    click(app, w, xv.recalc_button)
    check(errors)
    wait(app, w)
    screenshot(app, w, f"{PREFIX}_recalculation")


if __name__ == "__main__":
    run_qt_scenario(carbonate)
