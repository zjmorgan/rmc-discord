"""Data application presenter."""

import numpy as np

from rmc_discord.gui.presenters.base_presenter import BasePresenter


class DataPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_browse(self.browse)
        self.view.connect_load(self.load)
        self.view.connect_sample(self.sample)
        self.view.connect_slice(self.update_slice)

    def browse(self, key):
        filename = self.view.nxs_file_dialog()
        if filename:
            self.view.set_file(key, filename)

    def load(self):
        data, norm, subtract = self.view.get_files()
        if not data:
            return self.update_status("Choose a data file.")
        self.update_processing("Loading data.", 5)
        self.run_task(self.model.load, self.load_complete, data, norm, subtract)

    def load_complete(self, loaded):
        self.model.set_loaded(loaded)
        info = self.model.info()
        names = self.model.axis_names() if self.model.crystal is not None else ["u₁", "u₂", "u₃"]
        ranges = ", ".join(f"{n} [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in zip(names, info["ranges"]))
        lattice = ""
        if info["lattice"] is not None:
            lattice = " Data lattice: " + ", ".join(f"{v:.4f}" for v in info["lattice"]) + "."
        self.view.set_data_info(f"Grid {' × '.join(map(str, info['shape']))}; {ranges}; "
                                f"{100 * info['measured']:.0f}% measured.{lattice}")
        self.update_complete("Data loaded!")
        self.update_slice()

    def sample(self):
        if self.model.display is None:
            return self.update_status("Load data first.")
        if self.model.session.supercell is None:
            return self.update_status("Build the supercell first (Structure).")
        params = self.view.get_sampling()
        if params is None:
            return self.update_invalid()
        self.update_processing("Sampling.", 5)
        self.run_task(self.model.sample, self.sample_complete, *params)

    def sample_complete(self, result):
        sample, summary = result
        self.model.session.set_sample(self.model.display, sample)
        self.view.set_sample_info(
            f"{summary['points']:,} points kept of {summary['commensurate']:,} commensurate "
            f"({summary['bragg']:,} near Bragg peaks, {summary['rings']:,} on powder rings, rest outside |Q|)."
        )
        Q, values = self.model.sample_cloud()
        self.view.draw_cloud(Q, values)
        self.view.reset_view()
        self.update_slice()
        self.update_complete(f"{len(sample):,} points sampled!")

    def update_slice(self, *args):
        if self.model.display is None or self.model.crystal is None:
            return
        params = self.view.get_slice()
        if params is None:
            return
        normal, value, thickness = params
        image, ea, eb, axes, transform = self.model.slice(normal, value, thickness)
        names = self.model.axis_names()
        points = None
        if self.view.get_slice_style()[3]:
            pts = self.model.sample_points_in_slice(normal, value, thickness)
            points = pts if pts is not None else None
        title = f"{names[normal]} = {value:g} ± {thickness / 2:g}"
        self.view.draw_slice(image, ea, eb, transform, [names[a] for a in axes], title, points)
