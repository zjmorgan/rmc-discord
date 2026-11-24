"""Recalculation application presenter."""

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class RecalculationPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_recalculate(self.recalculate)
        self.view.connect_style(self.redraw)
        model.session.correlations_changed.connect(self.correlations_changed)

    def correlations_changed(self):
        self.view.set_quantities(self.model.channels())
        sample = self.model.session.sample
        if sample is not None and "q_range" in sample.info:
            self.view.set_q_range(*sample.info["q_range"])

    def recalculate(self):
        s = self.model.session
        if s.volume is None or s.fit is None:
            return self.update_status("Refine the correlations first (Refinement).")
        params = self.view.get_parameters()
        if params is None or params[0] is None:
            return self.update_invalid()
        self.update_processing("Recalculating.", 5)
        self.run_task(self.model.recalculate, self.recalculate_complete, *params)

    def recalculate_complete(self, res):
        self.model.result = res
        self.model.session.recalculated.emit()
        self.view.set_result_info(f"{res['n']:,} points: χ²/N {res['chi2']:.3g}; Pearson r {res['r']:.3f} with "
                                  f"background, {res['r_diffuse']:.3f} without; scale {res['scale']:.3g}.")
        self.view.camera_position = None
        self.redraw()
        self.view.reset_view()
        self.update_complete("Recalculated!")

    def redraw(self, *args):
        self.view.draw_planes(self.model.result)
        self.view.draw_surface(self.model.result)
