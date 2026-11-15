"""Recalculation application presenter."""

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class RecalculationPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_recalculate(self.recalculate)
        self.view.connect_style(self.redraw)
        model.session.correlations_changed.connect(self.correlations_changed)
        model.session.supercell_changed.connect(self.correlations_changed)

    def correlations_changed(self):
        self.view.set_quantities(self.model.quantities())
        self.view.set_r_max(self.model.max_range())

    def recalculate(self):
        s = self.model.session
        if s.volume is None:
            return self.update_status("Load and sample data first (Data).")
        params = self.view.get_parameters()
        if params is None or params[0] is None:
            return self.update_invalid()
        self.update_processing("Recalculating.", 5)
        self.run_task(self.model.recalculate, self.recalculate_complete, *params)

    def recalculate_complete(self, res):
        self.model.result = res
        self.model.session.recalculated.emit()
        self.view.set_result_info(f"{res['n']:,} points: Pearson r {res['r']:.3f} with background, "
                                  f"{res['r_diffuse']:.3f} for the diffuse part; scale {res['scale']:.3g}.")
        self.view.camera_position = None
        self.redraw()
        self.view.reset_view()
        self.update_complete("Recalculated!")

    def redraw(self, *args):
        self.view.draw_planes(self.model.result)
        self.view.draw_surface(self.model.result)
