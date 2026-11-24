"""Structure application presenter."""

from rmc_discord.gui.models.structure import parse_mapping
from rmc_discord.gui.presenters.base_presenter import BasePresenter


class StructurePresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_load_cif(self.load_cif)
        self.view.connect_build(self.build)
        self.view.connect_display(self.redraw)

    def load_cif(self, filename=None):
        filename = filename or self.view.load_cif_file_dialog()
        if not filename:
            return
        self.update_processing("Loading CIF.", 20)
        try:
            crystal = self.model.load_cif(filename)
        except Exception as e:
            self.update_status(f"Could not read CIF: {e}")
            self.update_progress(0)
            return
        self.view.set_cif_file(filename)
        self.view.set_cell(crystal.cell, crystal.space_group.getHMSymbol())
        partial = [s.label for s in crystal.asymmetric if s.occupancy < 1 - 1e-6]
        self.view.set_sites(self.model.site_rows(), partial)
        self.view.set_model_rows([])
        self.view.set_display("Unit Cell")
        self.view.camera_position = None
        self.redraw()
        self.view.reset_view()
        self.update_complete("CIF loaded!")

    def build(self):
        if self.model.cif is None:
            return self.update_status("Load a CIF first.")
        cell = self.view.get_cell()
        labels = self.view.get_active_labels()
        if cell is None:
            return self.update_status("Check the lattice.")
        ions_text, moments_text = self.view.get_magnetic_text()
        try:
            ions = parse_mapping(ions_text)
            moments = parse_mapping(moments_text, number=True)
            channels, magnetic, r_max, cluster, axis, difference = self.view.get_model_parameters()
        except ValueError as e:
            return self.update_status(str(e))
        if not magnetic and not (labels and any(channels.values())):
            return self.update_status("Check at least one disordered site and channel, or choose a magnetic model.")
        if not labels:
            channels = dict.fromkeys(channels, False)
        occupancies = self.view.get_occupancies()
        self.update_processing("Building the pair model.", 10)
        self.run_task(self.model.build_model, self.build_complete, cell, occupancies, labels, channels, magnetic,
                      ions, moments, r_max, cluster, axis, difference)

    def build_complete(self, result):
        crystal, structure, model = result
        active = self.view.get_active_labels()
        session = self.model.session
        session.set_crystal(crystal, structure)
        session.set_model(model)
        self.view.set_sites(self.model.site_rows(), active)
        self.view.set_model_rows(self.model.model_rows())
        self.redraw()
        self.update_complete(f"Model built: {model.n_coefficients} coefficients, {model.n_free} free.")

    def model_labels(self):
        model = self.model.session.model
        labels = set(self.view.get_active_labels())
        if model is not None:
            for b in model.blocks:
                labels |= set(b.sites.label)
        return labels

    def redraw(self, *args):
        if self.model.crystal is None:
            return
        mode, scale, show_inactive = self.view.get_display()
        active = self.model_labels() if mode == "Model Sites" else self.view.get_active_labels()
        style = self.model.unit_cell_atoms(active)
        self.view.draw_atoms(style, self.model.cell_edges(), scale, show_inactive and mode != "Model Sites")
