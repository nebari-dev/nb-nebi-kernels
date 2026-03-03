try:
    from nb_nebi_kernels.manager import NebiKernelSpecManager
except ModuleNotFoundError:  # pragma: no cover — manager not yet implemented
    pass

__version__ = "0.1.0"
__all__ = ["NebiKernelSpecManager"]
