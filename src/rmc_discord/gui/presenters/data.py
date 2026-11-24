"""Data application presenter."""


from rmc_discord.gui.presenters.base_presenter import BasePresenter


class DataPresenter(BasePresenter):
    def __init__(self, view, model):
        super().__init__(view, model)
        self.view.connect_browse(self.browse)
        self.view.connect_load(self.load)
        self.view.connect_sample(self.sample)
        self.view.connect_slice(self.update_slice)
        self.view.connect_view(self.update_view)

    def browse(self, key):
        filename = self.view.nxs_file_dialog()
        if filename:
            self.view.set_file(key, filename)

    def load(self):
        data, norm, subtract, subtract_norm = self.view.get_files()
        if not data:
            return self.update_status("Choose a data file.")
        self.update_processing("Loading data.", 5)
        self.run_task(self.model.load, self.load_complete, data, norm, subtract, subtract_norm,
                      self.view.get_background_mode())

    def load_complete(self, loaded):
        self.model.set_loaded(loaded)
        self.model.masked = None
        info = self.model.info()
        names = self.model.axis_names() if self.model.crystal is not None else ["u₁", "u₂", "u₃"]
        ranges = ", ".join(f"{n} [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in zip(names, info["ranges"]))
        lattice = ""
        if info["lattice"] is not None:
            lattice = " Data lattice: " + ", ".join(f"{v:.4f}" for v in info["lattice"]) + "."
        background = f" Background: {loaded['background']}." if loaded.get("background") else ""
        self.view.set_data_info(f"Grid {' × '.join(map(str, info['shape']))}; {ranges}; "
                                f"{100 * info['measured']:.0f}% measured.{lattice}{background}")
        self.update_complete(loaded.get("warning") or "Data loaded!")
        self.view.set_plane_names(names)
        self.update_slice()
        self.view.reset_view()

    def sample(self):
        if self.model.display is None:
            return self.update_status("Load data first.")
        if self.model.session.structure is None:
            return self.update_status("Load a CIF first (Structure).")
        params = self.view.get_sampling()
        if params is None:
            return self.update_invalid()
        self.update_processing("Sampling.", 5)
        self.run_task(self.model.sample, self.sample_complete, *params)

    def sample_complete(self, result):
        sample, summary = result
        self.model.masked = summary["masked"]
        self.model.session.set_sample(self.model.display, sample)
        self.view.set_sample_info(
            f"{summary['points']:,} observations ({summary['held']:,} held out) from {summary['bins']:,} measured "
            f"bins ({summary['bragg']:,} near Bragg peaks, {summary['rings']:,} on powder rings, rest outside |Q| "
            "or merged with equivalents). Check \"Show Masked Points\" on the Slice tab to see the masks."
        )
        if self.view.get_view()[0] == "planes":  # show what was sampled (drawn by update_slice)
            self.view.view_combo.blockSignals(True)
            self.view.view_combo.setCurrentIndex(1)
            self.view.view_combo.blockSignals(False)
        self.update_slice()
        self.update_complete(f"{len(sample):,} observations!")

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
            points = self.model.sample_points_in_slice(normal, value, thickness)
        masked = self.model.masked_points_in_slice(normal, value, thickness) if self.view.show_masked() else None
        title = f"{names[normal]} = {value:g} ± {thickness / 2:g}"
        self.view.draw_slice(image, ea, eb, transform, [names[a] for a in axes], title, points, masked)
        self.update_view()

    def update_view(self, *args):
        """The 3-D view: planes of the volume, or the observations (and masks) once sampled."""
        if self.model.display is None or self.model.crystal is None:
            return
        mode, color, size, max_points = self.view.get_view()
        if mode != "planes" and self.model.session.sample is not None:
            self.view.draw_cloud(self.model.cloud(mode == "masks", color, max_points), size)
            return
        positions = self.view.get_planes()
        if positions is not None:
            self.view.draw_planes(self.model.planes(positions))
