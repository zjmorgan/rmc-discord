"""Correlations application presenter."""

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class CorrelationsPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_compute(self.compute)
        self.view.connect_add(self.add_states)
        self.view.connect_remove(self.remove_state)
        self.view.connect_filter(self.update_display)
        self.view.connect_save(self.save)
        self.view.connect_csv(self.write_csv)
        model.session.supercell_changed.connect(self.supercell_changed)
        model.session.configuration_changed.connect(self.supercell_changed)
        self.quantity = None
        self.normalized = False

    def supercell_changed(self):
        self.view.set_quantities(self.model.available())

    def add_states(self):
        for filename in self.view.states_dialog():
            try:
                self.model.add_state(filename)
            except (ValueError, KeyError) as e:
                self.update_status(str(e))
        self.view.set_states([n for n, _ in self.model.states])

    def remove_state(self):
        self.model.remove_state(self.view.selected_state())
        self.view.set_states([n for n, _ in self.model.states])

    def compute(self):
        if self.model.session.supercell is None:
            return self.update_status("Build a supercell first.")
        quantity, symmetrize, normalize, current = self.view.get_compute()
        if quantity is None:
            return self.update_status("No correlations for this supercell (no mixed, displaced, or magnetic sites).")
        self.normalized = normalize
        self.update_processing("Computing correlations.", 5)
        self.run_task(self.model.compute, self.compute_complete, quantity, symmetrize, normalize, current)

    def compute_complete(self, result):
        quantity, raw, shown, n_states = result
        self.quantity = quantity
        self.model.set_computed(quantity, raw, shown)
        self.view.set_pairs(self.model.pair_names())
        self.view.set_compute_info(f"{len(shown):,} site pairs × separations from {n_states} configuration(s).")
        self.view.camera_position = None
        self.update_display()
        self.view.reset_view()
        self.update_complete("Correlations computed!")

    def update_display(self, *args):
        params = self.view.get_filter()
        if params is None or self.model.shown is None:
            return
        pair, r_min, r_max, size = params
        c = self.model.filtered(pair, r_min, r_max)
        self.view.draw_shells(self.model.shells(pair, r_min, r_max), self.quantity, self.normalized)
        self.view.draw_map(c.r if c is not None else None, c.value if c is not None else None, size)

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
