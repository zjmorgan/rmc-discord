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
    supercell : Supercell or None
    sample : QSample or None
        Sampled data at commensurate points (after masks and cuts).
    volume : Volume or None
        Reduced data volume the sample was taken from.
    refinement : Refinement or None
    held : dict of str to QSample
        Held-out point sets.
    result : AnnealResult or None
    correlations : dict of str to Correlations
    """

    crystal_changed = Signal()
    supercell_changed = Signal()
    sample_changed = Signal()
    refinement_changed = Signal()
    configuration_changed = Signal()
    correlations_changed = Signal()
    recalculated = Signal()

    def __init__(self):
        super().__init__()
        self.crystal = None
        self.supercell = None
        self.volume = None
        self.sample = None
        self.refinement = None
        self.held = {}
        self.result = None
        self.correlations = {}

    def set_crystal(self, crystal):
        self.crystal = crystal
        self.supercell = None
        self.crystal_changed.emit()
        self.supercell_changed.emit()

    def set_supercell(self, supercell):
        self.supercell = supercell
        self.refinement = None
        self.result = None
        self.correlations = {}
        self.supercell_changed.emit()
        self.correlations_changed.emit()

    def set_sample(self, volume, sample):
        self.volume = volume
        self.sample = sample
        self.refinement = None
        self.sample_changed.emit()

    def set_refinement(self, refinement, held):
        self.refinement = refinement
        self.held = held
        self.refinement_changed.emit()

    def set_correlations(self, quantity, corr):
        self.correlations[quantity] = corr
        self.correlations_changed.emit()
