"""Custom KernelSpecManager that discovers kernels from nebi workspaces."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jupyter_client.kernelspec import KernelSpec, KernelSpecManager, NoSuchKernel
from traitlets import Float, List, Unicode

from nb_nebi_kernels.discovery import (
    EnvironmentKernelSpec,
    NebiWorkspace,
    discover_environments,
    discover_kernel_specs,
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
    kernel_spec: EnvironmentKernelSpec | None = None
    show_kernel_display_name: bool = False


class NebiKernelSpecManager(KernelSpecManager):
    """KernelSpecManager that discovers kernels from nebi-tracked pixi workspaces.

    Each local workspace environment exposes one kernel per installed kernelspec.
    Non-ready environments expose a stub kernel with an actionable error.
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
        default_value=[],
        config=True,
        help=(
            "Optional packages required in addition to an installed Jupyter kernelspec. "
            "By default, any valid kernelspec is launchable."
        ),
    )
    discovery_cache_ttl_seconds = Float(
        default_value=30.0,
        config=True,
        help=(
            "Seconds to cache workspace discovery results before recomputing. "
            "Set to 0 to disable caching."
        ),
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._kernel_registry: dict[str, KernelEntry] = {}
        self._single_env_workspaces: set[str] = set()
        self._discovery_hash: str = ""
        self._discovered_at: str = ""
        self._last_discovery_monotonic: float | None = None
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

    def _make_kernel_name(
        self, workspace: NebiWorkspace, env: str, kernel_spec_name: str | None = None
    ) -> str:
        """Generate a kernel name from a workspace and environment."""
        clean_ws = self.clean_kernel_name(workspace.name)
        clean_env = self.clean_kernel_name(env)
        name = f"nebi-{clean_ws}-{clean_env}"
        if kernel_spec_name:
            name = f"{name}-{self.clean_kernel_name(kernel_spec_name)}"
        return name

    @staticmethod
    def _entry_identity(entry: KernelEntry) -> tuple[str, str, str, str, str]:
        """Return stable fields that identify the source of a kernel entry."""
        kernel_spec_name = entry.kernel_spec.name if entry.kernel_spec else ""
        kernel_spec_dir = entry.kernel_spec.spec.resource_dir if entry.kernel_spec else ""
        return (
            entry.workspace.name,
            entry.workspace.path,
            entry.environment,
            kernel_spec_name,
            kernel_spec_dir,
        )

    @staticmethod
    def _collision_suffix(entry: KernelEntry) -> str:
        """Return a stable short suffix for a colliding kernel entry."""
        payload = "\0".join(NebiKernelSpecManager._entry_identity(entry))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    def _register_kernel_entry(self, kernel_name: str, entry: KernelEntry) -> None:
        """Register a kernel entry without silently overwriting name collisions."""
        if kernel_name not in self._kernel_registry:
            self._kernel_registry[kernel_name] = entry
            return

        entry_identity = self._entry_identity(entry)
        if self._entry_identity(self._kernel_registry[kernel_name]) == entry_identity:
            logger.warning("Skipping duplicate Nebi kernel entry for %s", kernel_name)
            return

        suffix = self._collision_suffix(entry)
        candidate = f"{kernel_name}-{suffix}"
        existing = self._kernel_registry.get(candidate)
        if existing:
            if self._entry_identity(existing) == entry_identity:
                logger.warning("Skipping duplicate Nebi kernel entry for %s", candidate)
            else:
                logger.warning(
                    "Skipping Nebi kernel entry for %s; collision name %s is already registered",
                    kernel_name,
                    candidate,
                )
            return

        logger.warning(
            "Nebi kernel name collision for %s; registered colliding entry as %s",
            kernel_name,
            candidate,
        )
        self._kernel_registry[candidate] = entry

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

    def _classify_environment(self, workspace: NebiWorkspace, env: str) -> list[KernelEntry]:
        """Classify and expand one workspace environment into kernel entries."""
        if workspace.source == "remote" or not workspace.path:
            return [
                KernelEntry(
                    workspace=workspace,
                    environment=env,
                    state="remote-not-pulled",
                    missing_dependencies=[],
                    not_ready_reason="workspace-not-pulled",
                )
            ]

        probe = probe_environment(
            workspace.path,
            env,
            tuple(self.required_launch_dependencies),
        )
        if not probe.installed:
            return [
                KernelEntry(
                    workspace=workspace,
                    environment=env,
                    state="local-not-installed",
                    missing_dependencies=[],
                    not_ready_reason=probe.reason or "environment-not-installed",
                )
            ]

        if probe.missing_dependencies:
            return [
                KernelEntry(
                    workspace=workspace,
                    environment=env,
                    state="local-missing-deps",
                    missing_dependencies=probe.missing_dependencies,
                    not_ready_reason="missing-dependencies",
                )
            ]

        kernel_specs = discover_kernel_specs(workspace.path, env)
        if not kernel_specs:
            return [
                KernelEntry(
                    workspace=workspace,
                    environment=env,
                    state="local-missing-deps",
                    missing_dependencies=[],
                    not_ready_reason="kernel-not-installed",
                )
            ]

        state = "ready"
        not_ready_reason = None
        if (
            workspace.local_version
            and workspace.remote_version
            and workspace.local_version != workspace.remote_version
        ):
            state = "outdated"
            not_ready_reason = "local-version-behind-remote"

        show_kernel_display_name = len(kernel_specs) > 1
        return [
            KernelEntry(
                workspace=workspace,
                environment=env,
                state=state,
                missing_dependencies=[],
                not_ready_reason=not_ready_reason,
                kernel_spec=kernel_spec,
                show_kernel_display_name=show_kernel_display_name,
            )
            for kernel_spec in kernel_specs
        ]

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
                install_status=remote_ws.install_status,
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
                    "kernel_spec": entry.kernel_spec.name if entry.kernel_spec else None,
                    "state": entry.state,
                    "missing_dependencies": entry.missing_dependencies,
                    "local_version": workspace.local_version,
                    "remote_version": workspace.remote_version,
                    "install_status": workspace.install_status,
                    "not_ready_reason": entry.not_ready_reason,
                }
            )

        payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        self._discovery_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._discovered_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def invalidate_discovery_cache(self) -> None:
        """Force the next discovery call to recompute immediately."""
        self._last_discovery_monotonic = None

    def _discover(self, *, force: bool = False) -> None:
        """Run discovery and populate the kernel registry."""
        ttl_seconds = float(self.discovery_cache_ttl_seconds)
        if not force and ttl_seconds > 0 and self._last_discovery_monotonic is not None:
            age_seconds = time.monotonic() - self._last_discovery_monotonic
            if age_seconds < ttl_seconds:
                logger.debug(
                    "Using cached nebi discovery (age=%.2fs, ttl=%.2fs)",
                    age_seconds,
                    ttl_seconds,
                )
                return

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
                entries = self._classify_environment(ws, env)
                for index, entry in enumerate(entries):
                    kernel_spec_name = (
                        entry.kernel_spec.name
                        if len(entries) > 1 and index > 0 and entry.kernel_spec
                        else None
                    )
                    kernel_name = self._make_kernel_name(ws, env, kernel_spec_name)
                    self._register_kernel_entry(kernel_name, entry)

        self._update_discovery_metadata()
        self._last_discovery_monotonic = time.monotonic()
        logger.info("Discovered %d nebi kernels", len(self._kernel_registry))

    def find_kernel_specs(self) -> dict[str, str]:
        """Return a dict mapping kernel names to resource directories."""
        specs = super().find_kernel_specs()

        self._discover()

        for kernel_name, entry in self._kernel_registry.items():
            if entry.kernel_spec:
                specs[kernel_name] = entry.kernel_spec.spec.resource_dir
            else:
                workspace_path = entry.workspace.path
                specs[kernel_name] = (
                    workspace_path if workspace_path else self._fallback_resource_dir
                )

        return specs

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        """Get a KernelSpec by name."""
        if kernel_name in self._kernel_registry:
            entry = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(entry)

        if not kernel_name.startswith("nebi-"):
            return super().get_kernel_spec(kernel_name)

        self._discover()
        if kernel_name in self._kernel_registry:
            entry = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(entry)

        # A Nebi miss can mean a workspace was added since the last cached
        # refresh.
        self._discover(force=True)
        if kernel_name in self._kernel_registry:
            entry = self._kernel_registry[kernel_name]
            return self._create_kernel_spec(entry)

        return super().get_kernel_spec(kernel_name)

    def _create_kernel_spec(self, entry: KernelEntry) -> KernelSpec:
        """Create a KernelSpec for a workspace environment.

        Launchable entries route through the pixi-based launcher. Local
        entries that are not installed, not pulled, or missing a Jupyter kernel route
        through ``nb_nebi_kernels.stub_kernel`` so the user gets an actionable
        cell error instead of a silent kernel failure.
        """
        if entry.state in {
            "remote-not-pulled",
            "local-not-installed",
            "local-missing-deps",
        }:
            return self._stub_kernel_spec(entry)
        return self._working_kernel_spec(entry)

    def _kernel_metadata(self, entry: KernelEntry, *, kernel_state: str) -> dict[str, Any]:
        """Build flat metadata shared by working and stub kernels."""
        ws = entry.workspace
        local_version = ws.local_version
        remote_version = ws.remote_version
        is_outdated = bool(local_version and remote_version and local_version != remote_version)

        return {
            "nebi_workspace": ws.name,
            "nebi_workspace_path": ws.path,
            "pixi_environment": entry.environment,
            "nebi_kernel_spec": entry.kernel_spec.name if entry.kernel_spec else None,
            "nebi_state": entry.state,
            "nebi_kernel_state": kernel_state,
            "nebi_missing_dependencies": entry.missing_dependencies,
            "nebi_local_version": local_version,
            "nebi_remote_version": remote_version,
            "nebi_outdated": is_outdated,
            "nebi_install_status": ws.install_status,
            "nebi_source": ws.source,
            "nebi_not_ready_reason": entry.not_ready_reason,
            "nebi_discovery_hash": self._discovery_hash,
            "nebi_discovered_at": self._discovered_at,
        }

    def _working_kernel_spec(self, entry: KernelEntry) -> KernelSpec:
        ws = entry.workspace
        env = entry.environment
        source_spec = entry.kernel_spec.spec if entry.kernel_spec else None
        kernel_argv = (
            list(source_spec.argv)
            if source_spec
            else ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        )
        argv = [
            sys.executable,
            "-m",
            "nb_nebi_kernels.launcher",
            ws.path,
            env,
            *kernel_argv,
        ]

        resource_dir = (
            source_spec.resource_dir
            if source_spec
            else ws.path
            if ws.path and os.path.isdir(ws.path)
            else self._fallback_resource_dir
        )
        display_name = self._make_display_name(ws, env)
        if entry.show_kernel_display_name and entry.kernel_spec and source_spec:
            kernel_display_name = source_spec.display_name or entry.kernel_spec.name
            display_name = f"{display_name} — {kernel_display_name}"

        metadata = dict(source_spec.metadata) if source_spec else {}
        metadata.update(
            self._kernel_metadata(
                entry,
                kernel_state=("ready" if entry.state in {"ready", "outdated"} else entry.state),
            )
        )

        return KernelSpec(
            argv=argv,
            display_name=display_name,
            language=source_spec.language if source_spec else "python",
            resource_dir=resource_dir,
            env={
                **(dict(source_spec.env) if source_spec else {}),
                "NB_NEBI_KERNEL_STATE": entry.state,
                "NB_NEBI_KERNEL_NAME": ws.name,
            },
            interrupt_mode=source_spec.interrupt_mode if source_spec else "signal",
            metadata=metadata,
        )

    def _stub_kernel_spec(self, entry: KernelEntry) -> KernelSpec:
        ws = entry.workspace
        env = entry.environment
        argv = [
            sys.executable,
            "-m",
            "nb_nebi_kernels.stub_kernel",
            "--workspace",
            ws.name,
            "--env",
            env,
            "--reason",
            entry.not_ready_reason or entry.state,
        ]
        for dependency in entry.missing_dependencies:
            argv.extend(["--missing-dependency", dependency])
        argv.extend([
            "-f",
            "{connection_file}",
        ])

        resource_dir = (
            ws.path if ws.path and os.path.isdir(ws.path) else self._fallback_resource_dir
        )

        return KernelSpec(
            argv=argv,
            display_name=self._make_display_name(ws, env),
            language="no-op",
            resource_dir=resource_dir,
            metadata=self._kernel_metadata(entry, kernel_state=entry.state),
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
