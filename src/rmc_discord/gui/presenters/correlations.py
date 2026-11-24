"""Correlations application presenter."""

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class CorrelationsPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_channel(self.show_channel)
        self.view.connect_filter(self.update_display)
        self.view.connect_slice(self.update_slice)
        self.view.connect_save(self.save)
        self.view.connect_csv(self.write_csv)
        model.session.correlations_changed.connect(self.correlations_changed)
        self.quantity = None
        self.normalized = False

    def correlations_changed(self):
        self.view.set_quantities(self.model.available())
        if self.model.available():
            self.view.camera_position = None
            self.show_channel()
            self.view.reset_view()
        else:
            self.model.shown = None
            self.view.set_compute_info("Refine the correlations first (Refinement).")

    def show_channel(self, *args):
        channel, normalize = self.view.get_channel()
        if channel is None:
            return
        self.model.scale = self.view.get_scale()
        self.model.channel = channel
        self.view.set_difference(self.model.difference())
        self.view.set_occupancy(channel == "occupancy")
        table = self.model.select(channel, normalize)
        self.quantity = table.quantity if table is not None else None
        self.normalized = self.model.normalized
        self.view.set_pairs(self.model.pair_names())
        self.view.set_compute_info(self.model.info())
        self.update_display()

    def update_display(self, *args):
        params = self.view.get_filter()
        if params is None or self.model.shown is None:
            return
        pair, r_min, r_max, size = params
        c = self.model.filtered(pair, r_min, r_max)
        self.view.draw_shells(self.model.shells(pair, r_min, r_max), self.quantity, self.normalized)
        self.view.draw_map(c.r if c is not None else None, c.value if c is not None else None, size)
        self.update_slice()

    def label(self):
        if self.quantity == "spin":
            if self.model.difference():
                return "⟨S·S⟩ (scaled)" if self.model.scale else "s ⟨S·S⟩ (difference data, arb. units)"
            return "⟨S·S⟩"
        if self.normalized:
            return "correlation coefficient"
        return {"occupancy": "⟨δn δn⟩", "displacement": "tr V (Å²)", "bond": "size effect along the bond (Å)"}.get(
            self.quantity, "")

    def update_slice(self, *args):
        params, sl = self.view.get_filter(), self.view.get_slice()
        if params is None or sl is None or self.model.shown is None:
            return
        pair, _, r_max, _ = params
        hkl, d, tol, size = sl
        try:
            plane = self.model.plane(pair, hkl, d, tol, r_max)
        except ValueError as error:
            self.update_status(str(error))
            return
        self.view.draw_slice(plane, self.label(), size)

    def save(self):
        if self.model.shown is None:
            return
        filename = self.view.save_file_dialog("Save correlations", "HDF5 files (*.h5)")
        if filename:
            self.model.save(filename)
            self.update_complete(f"Saved {filename}.")

    def write_csv(self):
        params = self.view.get_filter()
        if self.model.shown is None or params is None:
            return
        shells = self.model.shells(*params[:3])
        if shells is None:
            return
        filename = self.view.save_file_dialog("Write shells", "CSV files (*.csv)")
        if filename:
            self.model.write_shells(shells, filename)
            self.update_complete(f"Wrote {filename}.")
