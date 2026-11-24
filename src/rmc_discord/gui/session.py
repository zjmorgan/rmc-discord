"""State shared by the applications of a session."""

from qtpy.QtCore import QObject, Signal


class Session(QObject):
    """
    Objects produced by each workflow step.

    Each step's model stores its result here and emits the matching
    signal; later steps listen and refresh.

    Attributes
    ----------
    crystal : Crystal or None
        Average structure (Mantid symmetry, display).
    structure : rmc_discord.pairs.Structure or None
        The same structure in the frame used by the pair model.
    model : rmc_discord.pairs.workflow.PairModel or None
        Correlation channels, orbits and constraints.
    volume : Volume or None
        Reduced data volume.
    sample : rmc_discord.pairs.sampling.Observations or None
        Observations taken from the volume (after masks and merging).
    fit : rmc_discord.pairs.workflow.FitResult or None
        Latest refinement.
    correlations : dict of str to Correlations
        Refined correlations of each channel.

    The model intensity on the whole volume is computed once per fit by
    NUFFT (:func:`rmc_discord.pairs.fast.forward`) and cached, so planes
    and slabs are slices of it.
    """

    crystal_changed = Signal()
    model_changed = Signal()
    sample_changed = Signal()
    fit_changed = Signal()
    correlations_changed = Signal()
    recalculated = Signal()

    def __init__(self):
        super().__init__()
        self.crystal = None
        self.structure = None
        self.model = None
        self.volume = None
        self.sample = None
        self.fit = None
        self.correlations = {}
        self._volumes = {}

    def model_volume(self, component="model"):
        """
        Refined intensity on every bin of the volume (cached per fit).

        ``component`` is ``"model"`` (with background), ``"background"``
        or a channel name (scaled, without background).
        """
        from rmc_discord.pairs.fast import forward

        if self.model is None or self.fit is None or self.volume is None:
            raise ValueError("refine the correlations first")
        key = (id(self.fit), component)
        if key not in self._volumes:
            if component == "model":
                kw = {}
            elif component == "background":
                kw = dict(channels=[], background=True)
            else:
                kw = dict(channels=[component], background=False)
            self._volumes[key] = forward(self.model, self.fit, volume=self.volume, **kw)
        return self._volumes[key]

    def set_crystal(self, crystal, structure=None):
        self._volumes = {}
        self.crystal = crystal
        self.structure = structure
        self.model = None
        self.fit = None
        self.correlations = {}
        self.crystal_changed.emit()
        self.model_changed.emit()
        self.correlations_changed.emit()

    def set_model(self, model):
        self._volumes = {}
        self.model = model
        self.fit = None
        self.correlations = {}
        self.model_changed.emit()
        self.fit_changed.emit()
        self.correlations_changed.emit()

    def set_sample(self, volume, sample):
        self._volumes = {}
        self.volume = volume
        self.sample = sample
        self.fit = None
        self.sample_changed.emit()
        self.fit_changed.emit()

    def set_fit(self, fit, correlations):
        self._volumes = {}
        self.fit = fit
        self.correlations = dict(correlations)
        self.fit_changed.emit()
        self.correlations_changed.emit()
