"""Custom KernelSpecManager that discovers kernels from nebi workspaces."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jupyter_client.kernelspec import KernelSpec, KernelSpecManager, NoSuchKernel
from traitlets import List, Unicode

from nb_nebi_kernels.discovery import (
    NebiWorkspace,
    discover_environments,
    discover_remote_workspaces,
    discover_workspaces,
    probe_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class KernelEntry:
    """Resolved kernel state for a single (workspace, environment) pair."""

    workspace: NebiWorkspace
    environment: str
    state: str
    missing_dependencies: list[str]
    not_ready_reason: str | None


class NebiKernelSpecManager(KernelSpecManager):  # type: ignore[misc]
    """KernelSpecManager that discovers kernels from nebi-tracked pixi workspaces.

    Each (workspace, environment) pair becomes a launchable Jupyter kernel.
    Workspaces are discovered via ``nebi workspace list`` and environments
    via ``pixi workspace environment list``.
    """

    workspace_discovery_roots = List(
        Unicode(),
        default_value=[],
        config=True,
        help=(
            "Extra local roots where pulled Nebi workspaces are discovered. "
            "Use this to discover workspaces from shared volumes (e.g. RWX NFS)."
        ),
    )
    required_launch_dependencies = List(
        Unicode(),
        default_value=["ipykernel"],
        config=True,
        help="Packages required in each pixi environment for a kernel to be launchable.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._kernel_registry: dict[str, KernelEntry] = {}
        self._single_env_workspaces: set[str] = set()
        self._discovery_hash: str = ""
        self._discovered_at: str = ""
        self._fallback_resource_dir = os.path.dirname(__file__)

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

    @staticmethod
    def _merge_environment_names(
        primary: list[str] | None, secondary: list[str] | None
    ) -> list[str]:
        """Merge environment name lists while preserving order and uniqueness."""
        merged: list[str] = []
        for source in (primary or [], secondary or []):
            for env_name in source:
                if env_name not in merged:
                    merged.append(env_name)
        return merged

    def _classify_kernel_state(self, workspace: NebiWorkspace, env: str) -> KernelEntry:
        """Classify kernel state for the given workspace/environment."""
        if workspace.source == "remote" or not workspace.path:
            return KernelEntry(
                workspace=workspace,
                environment=env,
                state="remote-not-pulled",
                missing_dependencies=[],
                not_ready_reason="workspace-not-pulled",
            )

        probe = probe_environment(
            workspace.path,
            env,
            tuple(self.required_launch_dependencies) or ("ipykernel",),
        )
        if not probe.installed:
            return KernelEntry(
                workspace=workspace,
                environment=env,
                state="local-not-installed",
                missing_dependencies=[],
                not_ready_reason=probe.reason or "environment-not-installed",
            )

        if probe.missing_dependencies:
            return KernelEntry(
                workspace=workspace,
                environment=env,
                state="local-missing-deps",
                missing_dependencies=probe.missing_dependencies,
                not_ready_reason="missing-dependencies",
            )

        if (
            workspace.local_version
            and workspace.remote_version
            and workspace.local_version != workspace.remote_version
        ):
            return KernelEntry(
                workspace=workspace,
                environment=env,
                state="outdated",
                missing_dependencies=[],
                not_ready_reason="local-version-behind-remote",
            )

        return KernelEntry(
            workspace=workspace,
            environment=env,
            state="ready",
            missing_dependencies=[],
            not_ready_reason=None,
        )

    def _merge_workspaces(
        self, local_workspaces: list[NebiWorkspace], remote_workspaces: list[NebiWorkspace]
    ) -> list[NebiWorkspace]:
        """Merge local and remote workspace views by workspace name."""
        merged: dict[str, NebiWorkspace] = {}

        for ws in local_workspaces:
            merged[ws.name] = ws

        for remote_ws in remote_workspaces:
            existing = merged.get(remote_ws.name)
            if existing:
                existing.remote_version = remote_ws.remote_version or existing.remote_version
                existing.environments = self._merge_environment_names(
                    existing.environments, remote_ws.environments
                )
                continue

            merged[remote_ws.name] = NebiWorkspace(
                name=remote_ws.name,
                path=remote_ws.path,
                local_version=remote_ws.local_version,
                remote_version=remote_ws.remote_version,
                environments=remote_ws.environments,
                source="remote",
            )

        return list(merged.values())

    def _update_discovery_metadata(self) -> None:
        """Compute deterministic discovery metadata for freshness checks."""
        summary: list[dict[str, Any]] = []
        for kernel_name in sorted(self._kernel_registry):
            entry = self._kernel_registry[kernel_name]
            workspace = entry.workspace
            summary.append(
                {
                    "kernel_name": kernel_name,
                    "workspace": workspace.name,
                    "workspace_path": workspace.path,
                    "source": workspace.source,
                    "environment": entry.environment,
                    "state": entry.state,
                    "missing_dependencies": entry.missing_dependencies,
                    "local_version": workspace.local_version,
                    "remote_version": workspace.remote_version,
                    "not_ready_reason": entry.not_ready_reason,
                }
            )

        payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        self._discovery_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._discovered_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _discover(self) -> None:
        """Run discovery and populate the kernel registry."""
        self._kernel_registry.clear()
        self._single_env_workspaces.clear()

        local_workspaces = discover_workspaces(list(self.workspace_discovery_roots))
        remote_workspaces = discover_remote_workspaces()
        workspaces = self._merge_workspaces(local_workspaces, remote_workspaces)

        for ws in workspaces:
            if ws.source != "remote" and ws.path:
                envs = self._merge_environment_names(
                    discover_environments(ws.path), ws.environments
                )
            else:
                envs = ws.environments or ["default"]

            if envs == ["default"]:
                self._single_env_workspaces.add(ws.name)

            for env in envs:
                kernel_name = self._make_kernel_name(ws, env)
                self._kernel_registry[kernel_name] = self._classify_kernel_state(ws, env)

        self._update_discovery_metadata()
        logger.info("Discovered %d nebi kernels", len(self._kernel_registry))

    def find_kernel_specs(self) -> dict[str, str]:
        """Return a dict mapping kernel names to resource directories."""
        specs = super().find_kernel_specs()

        self._discover()

        for kernel_name, entry in self._kernel_registry.items():
            workspace_path = entry.workspace.path
            specs[kernel_name] = workspace_path if workspace_path else self._fallback_resource_dir

        return specs

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        """Get a KernelSpec by name."""
        if kernel_name in self._kernel_registry:
            entry = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(entry)

        # Refresh in case a new workspace was added
        self._discover()
        if kernel_name in self._kernel_registry:
            entry = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(entry)

        return super().get_kernel_spec(kernel_name)

    def _create_kernel_spec(self, entry: KernelEntry) -> KernelSpec:
        """Create a KernelSpec for a workspace environment."""
        ws = entry.workspace
        env = entry.environment
        local_version = ws.local_version
        remote_version = ws.remote_version
        is_outdated = bool(local_version and remote_version and local_version != remote_version)

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
            "nebi_state": entry.state,
            "nebi_missing_dependencies": entry.missing_dependencies,
            "nebi_local_version": local_version,
            "nebi_remote_version": remote_version,
            "nebi_outdated": is_outdated,
            "nebi_source": ws.source,
            "nebi_not_ready_reason": entry.not_ready_reason,
            "nebi_logo_reason": (
                None if entry.state in {"ready", "outdated"} else entry.not_ready_reason
            ),
            "nebi_discovery_hash": self._discovery_hash,
            "nebi_discovered_at": self._discovered_at,
            "nebi": {
                "workspace": ws.name,
                "workspace_path": ws.path,
                "environment": env,
                "state": entry.state,
                "missing_dependencies": entry.missing_dependencies,
                "local_version": local_version,
                "remote_version": remote_version,
                "outdated": is_outdated,
                "not_ready_reason": entry.not_ready_reason,
                "logo_reason": None
                if entry.state in {"ready", "outdated"}
                else entry.not_ready_reason,
                "discovery_hash": self._discovery_hash,
                "discovered_at": self._discovered_at,
            },
        }

        resource_dir = (
            ws.path if ws.path and os.path.isdir(ws.path) else self._fallback_resource_dir
        )

        kernel_env = {
            "NB_NEBI_KERNEL_STATE": entry.state,
            "NB_NEBI_KERNEL_NAME": ws.name,
        }

        return KernelSpec(
            argv=argv,
            display_name=display_name,
            language="python",
            resource_dir=resource_dir,
            env=kernel_env,
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
