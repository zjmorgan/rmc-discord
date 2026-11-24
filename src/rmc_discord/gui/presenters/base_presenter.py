"""Base presenter: camera, status, and lattice display (as in NeuXtalViz)."""


class BasePresenter:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.connect_manual_axis(self.view_manual)
        self.view.connect_manual_up_axis(self.view_up_manual)
        self.view.connect_reciprocal_axes(self.view_bc_star, self.view_ca_star, self.view_ab_star)
        self.view.connect_real_axes(self.view_bc, self.view_ca, self.view_ab)
        self.view.connect_save_screenshot(self.save_screenshot)
        self.view.connect_reciprocal_real_compass(self.change_lattice)
        self.view.connect_rotate(self.rotate_view_ccw, self.rotate_view_cw)
        self.view.connect_elev(self.elevate_view_up, self.elevate_view_down)
        self.view.connect_az(self.azimuth_view_left, self.azimuth_view_right)
        self.view.connect_camera_ready(self.calculate_camera_parameters)
        self.view.connect_stop_process(self.stop_process)

        session = model.session
        session.crystal_changed.connect(self.update_lattice)
        session.model_changed.connect(self.update_lattice)

    # status

    def update_status(self, status):
        self.view.set_info(status)

    def update_progress(self, progress):
        self.view.set_step(progress)

    def update_invalid(self):
        self.update_status("Invalid parameters.")
        self.update_progress(0)

    def update_complete(self, status="Complete!"):
        self.update_status(status)
        self.update_progress(0)

    def update_processing(self, status="Processing...", progress=1):
        self.update_status(status)
        self.update_progress(progress)

    def update_error(self, error):
        exctype, value, details = error
        self.update_status(f"{exctype.__name__}: {value}")
        self.update_progress(0)
        self.view.show_error(f"{exctype.__name__}: {value}", details)

    def run_task(self, task, complete, *args, progress=None, finished=None):
        """
        Run ``task(*args, progress=...)`` in a worker; ``complete(result)`` on
        success, optional extra ``progress(message, percent)`` and
        ``finished()`` handlers.
        """
        worker = self.view.worker(task)
        worker.args = args
        worker.connect_result(complete)
        worker.connect_progress(self.update_processing)
        if progress is not None:
            worker.connect_progress(progress)
        if finished is not None:
            worker.connect_finished(finished)
        worker.connect_error(self.update_error)
        self.view.start_worker_pool(worker)
        return worker

    def stop_process(self):
        """Stop the running task, if the application's model can."""
        stop = getattr(self.model, "stop", None)
        if callable(stop):
            stop()
            self.update_status("Stopping...")

    # lattice

    def update_lattice(self):
        params = self.model.get_lattice_parameters()
        if params is not None:
            self.view.set_lattice_parameters(*params)
        info = self.model.get_model_info()
        if info is not None:
            self.view.set_model_info(*info)
        self.change_lattice()

    def change_lattice(self):
        self.view.set_transform(self.model.get_transform(self.view.reciprocal_lattice()))

    def save_screenshot(self):
        filename = self.view.save_screenshot_file_dialog()
        if filename:
            self.view.save_screenshot(filename)

    # camera

    def _view(self, vecs):
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_manual(self):
        indices = self.view.get_manual_axis_indices()
        if indices is not None:
            self._view(self.model.get_vector(*indices))

    def view_up_manual(self):
        indices = self.view.get_manual_axis_up_indices()
        if indices is not None:
            vec = self.model.get_vector(*indices)
            if vec is not None:
                self.view.view_up_vector(vec)

    def view_ab_star(self):
        self._view(self.model.ab_star_axes())

    def view_bc_star(self):
        self._view(self.model.bc_star_axes())

    def view_ca_star(self):
        self._view(self.model.ca_star_axes())

    def view_ab(self):
        self._view(self.model.ab_axes())

    def view_bc(self):
        self._view(self.model.bc_axes())

    def view_ca(self):
        self._view(self.model.ca_axes())

    # rotate view

    def _apply_camera_rotation(self, rotate, angle):
        position, fp, up = self.view.get_camera_state()
        self.view.set_camera_state(*rotate(position, fp, up, angle))

    def rotate_view_ccw(self):
        self._apply_camera_rotation(self.model.rotate_camera_roll, self.view.get_rotate_step())

    def rotate_view_cw(self):
        self._apply_camera_rotation(self.model.rotate_camera_roll, -self.view.get_rotate_step())

    def elevate_view_up(self):
        self._apply_camera_rotation(self.model.rotate_camera_elevation, self.view.get_rotate_step())

    def elevate_view_down(self):
        self._apply_camera_rotation(self.model.rotate_camera_elevation, -self.view.get_rotate_step())

    def azimuth_view_left(self):
        self._apply_camera_rotation(self.model.rotate_camera_azimuth, -self.view.get_rotate_step())

    def azimuth_view_right(self):
        self._apply_camera_rotation(self.model.rotate_camera_azimuth, self.view.get_rotate_step())

    def calculate_camera_parameters(self):
        roll, direction = self.view.get_camera_roll_direction()
        self.view.update_camera_display(*self.model.calculate_camera_parameters(direction, roll))
