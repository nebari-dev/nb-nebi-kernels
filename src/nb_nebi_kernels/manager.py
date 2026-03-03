"""Custom KernelSpecManager that discovers kernels from nebi workspaces."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

from jupyter_client.kernelspec import KernelSpec, KernelSpecManager, NoSuchKernel

from nb_nebi_kernels.discovery import (
    NebiWorkspace,
    discover_environments,
    discover_workspaces,
)

logger = logging.getLogger(__name__)


class NebiKernelSpecManager(KernelSpecManager):  # type: ignore[misc]
    """KernelSpecManager that discovers kernels from nebi-tracked pixi workspaces.

    Each (workspace, environment) pair becomes a launchable Jupyter kernel.
    Workspaces are discovered via ``nebi workspace list`` and environments
    via ``pixi workspace environment list``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._kernel_registry: dict[str, tuple[NebiWorkspace, str]] = {}
        self._single_env_workspaces: set[str] = set()

        logger.info("NebiKernelSpecManager initialized")

    @staticmethod
    def clean_kernel_name(name: str) -> str:
        """Clean a name for use as a Jupyter kernel name.

        Jupyter kernel names must be ASCII alphanumerics, underscores,
        and hyphens only.
        """
        try:
            name.encode("ascii")
        except UnicodeEncodeError:
            import unicodedata

            nfkd_form = unicodedata.normalize("NFKD", name)
            name = "".join(c for c in nfkd_form if not unicodedata.combining(c))

        name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
        return name

    def _make_kernel_name(self, workspace: NebiWorkspace, env: str) -> str:
        """Generate a kernel name from a workspace and environment."""
        clean_ws = self.clean_kernel_name(workspace.name)
        clean_env = self.clean_kernel_name(env)
        return f"nebi-{clean_ws}-{clean_env}"

    def _make_display_name(self, workspace: NebiWorkspace, env: str) -> str:
        """Generate a display name for the Jupyter kernel picker.

        Returns 'workspace (env)' for multi-env workspaces,
        or just 'workspace' if only the default environment exists.
        """
        if env == "default" and workspace.name in self._single_env_workspaces:
            return workspace.name
        return f"{workspace.name} ({env})"

    def _discover(self) -> None:
        """Run discovery and populate the kernel registry."""
        self._kernel_registry.clear()
        self._single_env_workspaces.clear()

        workspaces = discover_workspaces()

        for ws in workspaces:
            envs = discover_environments(ws.path)

            if envs == ["default"]:
                self._single_env_workspaces.add(ws.name)

            for env in envs:
                kernel_name = self._make_kernel_name(ws, env)
                self._kernel_registry[kernel_name] = (ws, env)

        logger.info("Discovered %d nebi kernels", len(self._kernel_registry))

    def find_kernel_specs(self) -> dict[str, str]:
        """Return a dict mapping kernel names to resource directories."""
        specs = super().find_kernel_specs()

        self._discover()

        for kernel_name, (ws, env) in self._kernel_registry.items():
            specs[kernel_name] = ws.path

        return specs

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        """Get a KernelSpec by name."""
        if kernel_name in self._kernel_registry:
            ws, env = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(ws, env)

        # Refresh in case a new workspace was added
        self._discover()
        if kernel_name in self._kernel_registry:
            ws, env = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(ws, env)

        return super().get_kernel_spec(kernel_name)

    def _create_kernel_spec(self, ws: NebiWorkspace, env: str) -> KernelSpec:
        """Create a KernelSpec for a workspace environment."""
        argv = [
            sys.executable,
            "-m",
            "nb_nebi_kernels.launcher",
            ws.path,
            env,
            "{connection_file}",
        ]

        display_name = self._make_display_name(ws, env)

        metadata = {
            "nebi_workspace": ws.name,
            "nebi_workspace_path": ws.path,
            "pixi_environment": env,
        }

        return KernelSpec(
            argv=argv,
            display_name=display_name,
            language="python",
            resource_dir=ws.path,
            metadata=metadata,
        )

    def get_all_specs(self) -> dict[str, dict[str, Any]]:
        """Return all kernel specs with metadata."""
        specs: dict[str, dict[str, Any]] = {}

        for kernel_name, resource_dir in self.find_kernel_specs().items():
            try:
                spec = self.get_kernel_spec(kernel_name)
                specs[kernel_name] = {
                    "resource_dir": resource_dir,
                    "spec": spec.to_dict(),
                }
            except NoSuchKernel:
                logger.warning("Could not get spec for kernel '%s'", kernel_name)
            except Exception:
                logger.exception("Error getting spec for kernel '%s'", kernel_name)

        return specs
