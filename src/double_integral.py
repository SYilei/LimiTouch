"""
Computer sample-based double integral
"""
import abc
from scipy.integrate import trapz, simps


class Smoother(metaclass=abc.ABCMeta):
    """
    Smoothes a time series.
    """
    @abc.abstractmethod
    def update(self, value):
        pass


class WeightedUpdater(Smoother):
    def __init__(self, sensitivity=0.5):
        if sensitivity < 0 or sensitivity > 1:
            raise Exception(
                    "Sensitivty for weighted updater must be between 0 and 1."
                    "  Received " + repr(sensitivity) + " instead!")
        self.sensitivity = sensitivity
        self.innertia = 1 - sensitivity
        self.residue = None

    def update(self, value):
        if self.residue is None:
            self.residue = value
        else:
            self.residue = self.sensitivity * value + self.innertia * self.residue
        return self.residue


class Integrator1D():
    """
    An integrator in 1D.
    """
    def __init__(self, spacing, initial_value=0, smoother=None):
        self.spacing = spacing
        self.initial_value = initial_value
        self.smoother = smoother
        self.data = tuple()

    def update(self, value):
        if self.smoother is not None:
            value = self.smoother.update(value)
        self.data += tuple([value])
        if len(self.data) == 1:
            return 0
        return self.initial_value + simps(self.data, dx=self.spacing)


class DoubleIntegrator1D():
    def __init__(self, spacing, initial_values=None, smoothers=None):
        initial_values = initial_values or (0, 0)
        smoothers = smoothers or [None]*2
        self.inner_integrator = Integrator1D(spacing, initial_values[0], smoothers[0])
        self.outer_integrator = Integrator1D(spacing, initial_values[1], smoothers[1])

    def update(self, value):
        return self.outer_integrator.update(self.inner_integrator.update(value))


class DoubleIntegrator3D():
    def __init__(self, spacing, initial_values=None, smoothers=None):
        """
        params:
            initial_values: two 3D vectors indicating initial values for the
                            inner and outer integrations respectively
        """
        initial_values = [None]*3 if initial_values is None else zip(*initial_values)
        smoothers = smoothers or [None]*3
        self.x = DoubleIntegrator1D(spacing, initial_values[0], smoothers[0])
        self.y = DoubleIntegrator1D(spacing, initial_values[1], smoothers[1])
        self.z = DoubleIntegrator1D(spacing, initial_values[2], smoothers[2])

    def update(self, value):
        return self.x.update(value[0]), self.y.update(value[1]), self.z.update(value[2])
