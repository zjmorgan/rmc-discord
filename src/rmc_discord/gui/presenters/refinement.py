"""Refinement application presenter."""

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class RefinementPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.set_settings(model.defaults())
        self.view.connect_fit(self.fit)
        self.view.connect_compare(self.update_comparison)
        session = model.session
        session.model_changed.connect(self.reset)
        session.sample_changed.connect(self.reset)

    def reset(self):
        self.model.comparison = None
        s = self.model.session
        if s.model is None:
            self.view.set_fit_info("Build the pair model (Structure).")
        elif s.sample is None:
            self.view.set_fit_info("Sample the data (Data).")
        else:
            self.view.set_fit_info(f"{s.model.n_coefficients} coefficients ({s.model.n_free} free) against "
                                   f"{len(s.sample):,} observations ({int(s.sample.held.sum()):,} held out).")
        self.view.draw_comparison(None)

    def fit(self):
        s = self.model.session
        if s.model is None or s.sample is None:
            return self.update_status("Build the model and sample the data first.")
        settings = self.view.get_settings()
        if settings is None:
            return self.update_invalid()
        self.view.set_running(True)
        self.update_processing("Refining correlations.", 2)
        self.run_task(self.model.fit, self.fit_complete, settings, finished=lambda: self.view.set_running(False))

    def fit_complete(self, result):
        fit, tables, record = result
        self.model.set_fit(fit, tables, record)
        self.view.draw_records(self.model.records)
        self.view.set_fit_info(f"Train χ² {record['chi2_train']:.4g} on {record['n_train']:,}, held-out χ² "
                               f"{record['chi2_test']:.4g} on {record['n_test']:,} observations; rank {record['free']}"
                               + (f"; {record['ties']} indistinguishable orbits tied." if record["ties"] else "."))
        self.update_comparison()
        self.update_complete(f"Refined in {record['seconds']:.1f} s.")

    def update_comparison(self, *args):
        params = self.view.get_compare()
        if params is None or self.model.session.fit is None:
            return
        normal, value, thickness = params
        out = self.model.compare_slice(normal, value, thickness)
        names = out["names"] if out is not None else None
        title = ""
        if names is not None:
            title = f"{names[normal]} = {value:g} ± {thickness / 2:g} ({out['n']:,} bins)"
        self.view.draw_comparison(out, title or "slice")
