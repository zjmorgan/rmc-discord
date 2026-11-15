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
        self.view.set_sites(self.model.site_rows())
        self.view.set_kinds([])
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
        if cell is None or not labels:
            return self.update_status("Check the lattice and at least one disordered site.")
        groups_text, ions_text, moments_text = self.view.get_disorder_text()
        try:
            groups = parse_mapping(groups_text, many=True)
            ions = parse_mapping(ions_text)
            moments = parse_mapping(moments_text, number=True)
        except ValueError as e:
            return self.update_status(str(e))
        N, seed, displacements = self.view.get_supercell_parameters()
        occupancies = self.view.get_occupancies()
        self.update_processing("Building supercell.", 10)
        self.run_task(self.model.build_supercell, self.build_complete, cell, occupancies, labels, N,
                      groups, ions, moments, seed, displacements)

    def build_complete(self, result):
        crystal, sc = result
        active = self.view.get_active_labels()
        self.model.session.set_crystal(crystal)
        self.model.session.set_supercell(sc)
        self.view.set_sites(self.model.site_rows(), active)
        self.view.set_kinds(self.model.kind_rows())
        self.view.camera_position = None
        self.view.set_display("Supercell")
        self.redraw()
        self.view.reset_view()
        self.update_complete(f"Supercell built: {len(sc)} atoms on {sc.n_sites} sites.")

    def redraw(self, *args):
        if self.model.crystal is None:
            return
        mode, scale, show_inactive = self.view.get_display()
        if mode == "Supercell" and self.model.session.supercell is not None:
            sc = self.model.session.supercell
            self.view.draw_atoms(self.model.supercell_atoms(), self.model.cell_edges(sc.N), scale)
        else:
            style = self.model.unit_cell_atoms(self.view.get_active_labels())
            self.view.draw_atoms(style, self.model.cell_edges(), scale, show_inactive)
