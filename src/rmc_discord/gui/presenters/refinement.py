"""Refinement application presenter."""

from rmc_discord.gui.models import display
from rmc_discord.gui.models.refinement import schedule_from
from rmc_discord.gui.models.structure import StructureModel
from rmc_discord.gui.presenters.base_presenter import BasePresenter


class RefinementPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_initialize(self.initialize)
        self.view.connect_run(self.run)
        self.view.connect_stop(self.stop)
        self.view.connect_save_configuration(self.save_configuration)
        self.view.connect_load_configuration(self.load_configuration)
        session = model.session
        session.supercell_changed.connect(self.supercell_changed)
        session.sample_changed.connect(self.sample_changed)
        session.configuration_changed.connect(self.draw)
        self._cells = StructureModel(session)

    def supercell_changed(self):
        if self.model.session.supercell is None:
            return
        self.view.set_settings(self.model.defaults(), self.model.site_labels())
        self.view.set_init_info("")
        self.view.draw_records([], {})
        self.view.camera_position = None
        self.draw()
        self.view.reset_view()

    def sample_changed(self):
        self.view.set_init_info("Sample changed: initialize the refinement.")

    def initialize(self):
        s = self.model.session
        if s.supercell is None or s.sample is None:
            return self.update_status("Build the supercell and sample the data first.")
        try:
            st = self.view.get_settings()
        except ValueError as e:
            return self.update_status(f"Invalid settings: {e}")
        if not (st.nuclear or st.magnetic):
            return self.update_status("Choose nuclear and/or magnetic scattering.")
        self.update_processing("Initializing refinement.", 5)
        self.run_task(self.model.initialize, self.initialize_complete, st)

    def initialize_complete(self, result):
        ref, held, info = result
        self.model.set_initialized(ref, held, info)
        held_text = ", ".join(f"{k} {v:,}" for k, v in info["held"].items())
        base = ", ".join(f"{k} {v:.4g}" for k, v in info["baseline"].items())
        moves = ", ".join(f"{k} {v:.2f}" for k, v in info["moves"].items())
        self.view.set_init_info(f"Pool {info['pool']:,} points; held out {held_text}. Pool χ² {info['chi']:.4g}; "
                                f"background-only held-out χ² {base}. Move mix: {moves}.")
        self.view.draw_records([], {})
        self.update_complete("Refinement initialized!")

    def run(self):
        if self.model.session.refinement is None:
            return self.update_status("Initialize the refinement first.")
        try:
            schedule = schedule_from(self.view.get_schedule_values())
        except (ValueError, TypeError) as e:
            return self.update_status(f"Invalid schedule: {e}")
        self.view.set_running(True)
        self.update_processing("Annealing.", 1)
        self.run_task(self.model.run, self.run_complete, schedule, progress=self.progress,
                      finished=lambda: self.view.set_running(False))

    def progress(self, message, percent):
        self.view.draw_records(self.model.records, self.model.baseline)
        self.draw()

    def run_complete(self, result):
        s = self.model.session
        s.result = result
        s.refinement_changed.emit()
        s.configuration_changed.emit()
        if result.best is not None:
            self.update_complete(f"Anneal done: best score {result.best['score']:.4f} at round "
                                 f"{result.best['round'] + 1} (configuration restored).")
        else:
            self.update_complete("Anneal stopped before the first check.")

    def stop(self):
        self.model.stop()
        self.update_status("Stopping after this round.")

    def stop_process(self):
        self.stop()

    def save_configuration(self):
        if self.model.session.supercell is None:
            return
        filename = self.view.config_save_dialog()
        if filename:
            self.model.save_configuration(filename)
            self.update_complete(f"Saved {filename}.")

    def load_configuration(self):
        if self.model.session.supercell is None:
            return
        filename = self.view.config_load_dialog()
        if filename:
            try:
                self.model.load_configuration(filename)
            except (ValueError, KeyError) as e:
                return self.update_status(str(e))
            self.update_complete(f"Loaded {filename}.")

    def draw(self):
        sc = self.model.session.supercell
        if sc is None:
            return
        snap = self.model.snapshot or {}
        style = display.configuration(sc, kind=snap.get("kind"), S=snap.get("S", sc.S), u=snap.get("u"))
        self.view.draw_configuration(style, self._cells.cell_edges(sc.N))
