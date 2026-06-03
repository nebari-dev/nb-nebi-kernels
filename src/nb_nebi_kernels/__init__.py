from nb_nebi_kernels._version import __version__
from nb_nebi_kernels.manager import NebiKernelSpecManager

__all__ = ["NebiKernelSpecManager", "__version__"]


def _jupyter_labextension_paths() -> list[dict[str, str]]:
    """Tell JupyterLab where this package's prebuilt federated extension lives.

    ``src`` is the built-asset directory relative to this package; ``dest`` is
    the labextension name (matches ``package.json`` and the wheel's
    ``share/jupyter/labextensions`` shared-data target).
    """
    return [{"src": "labextension", "dest": "nb-nebi-kernels"}]
