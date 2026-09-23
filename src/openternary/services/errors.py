"""Version-one command error contract."""


class CapabilityError(ValueError):
    """The requested operation is not implemented for this combination."""


class DependencyError(ImportError):
    """An explicitly selected optional integration is not installed."""


class HardwareError(RuntimeError):
    """The requested runtime cannot be used."""


class AcceptanceError(RuntimeError):
    """An explicitly required gate did not pass."""


class BudgetError(RuntimeError):
    """Execution stopped at a resource limit."""
