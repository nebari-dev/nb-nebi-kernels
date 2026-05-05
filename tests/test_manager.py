"""Tests for NebiKernelSpecManager."""

import sys
from unittest.mock import patch

import pytest

from nb_nebi_kernels.discovery import NebiWorkspace
from nb_nebi_kernels.manager import NebiKernelSpecManager


@pytest.fixture
def sample_workspaces() -> list[NebiWorkspace]:
    return [
        NebiWorkspace(name="data-science", path="/home/user/data-science"),
        NebiWorkspace(name="web-app", path="/home/user/web-app"),
    ]


@pytest.fixture
def sample_envs_map() -> dict[str, list[str]]:
    """Map workspace path -> environment list."""
    return {
        "/home/user/data-science": ["default", "gpu"],
        "/home/user/web-app": ["default"],
    }


class TestNebiKernelSpecManager:
    """Tests for the kernel spec manager."""

    def test_find_kernel_specs_returns_nebi_kernels(
        self, sample_workspaces: list[NebiWorkspace], sample_envs_map: dict[str, list[str]]
    ) -> None:
        """find_kernel_specs returns one entry per (workspace, env) pair."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: sample_envs_map[p],
            ),
        ):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        # 3 nebi kernels + whatever parent returns
        assert "nebi-data-science-default" in specs
        assert "nebi-data-science-gpu" in specs
        assert "nebi-web-app-default" in specs

    def test_find_kernel_specs_includes_parent_kernels(
        self, sample_workspaces: list[NebiWorkspace], sample_envs_map: dict[str, list[str]]
    ) -> None:
        """find_kernel_specs also includes standard kernels from parent."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: sample_envs_map[p],
            ),
            patch.object(
                NebiKernelSpecManager.__bases__[0],
                "find_kernel_specs",
                return_value={"python3": "/usr/share/jupyter/kernels/python3"},
            ),
        ):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        assert "python3" in specs

    def test_get_kernel_spec_returns_correct_argv(
        self, sample_workspaces: list[NebiWorkspace], sample_envs_map: dict[str, list[str]]
    ) -> None:
        """get_kernel_spec returns a KernelSpec with correct argv for pixi launch."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: sample_envs_map[p],
            ),
        ):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.argv == [
            sys.executable,
            "-m",
            "nb_nebi_kernels.launcher",
            "/home/user/data-science",
            "gpu",
            "{connection_file}",
        ]

    def test_display_name_format(
        self, sample_workspaces: list[NebiWorkspace], sample_envs_map: dict[str, list[str]]
    ) -> None:
        """Display name format: 'workspace (env)' or just 'workspace' for default."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: sample_envs_map[p],
            ),
        ):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            gpu_spec = manager.get_kernel_spec("nebi-data-science-gpu")
            default_spec = manager.get_kernel_spec("nebi-web-app-default")

        assert gpu_spec.display_name == "data-science (gpu)"
        # web-app has only default env, so display name is just the workspace name
        assert default_spec.display_name == "web-app"

    def test_get_kernel_spec_falls_back_to_parent(
        self, sample_workspaces: list[NebiWorkspace], sample_envs_map: dict[str, list[str]]
    ) -> None:
        """get_kernel_spec delegates to parent for non-nebi kernels."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: sample_envs_map[p],
            ),
        ):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            from jupyter_client.kernelspec import NoSuchKernel
            with pytest.raises(NoSuchKernel):
                manager.get_kernel_spec("nonexistent-kernel")

    def test_returns_empty_when_no_workspaces(self) -> None:
        """find_kernel_specs returns only parent kernels when nebi has no workspaces."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
        ):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        assert not any(k.startswith("nebi-") for k in specs)

    def test_clean_kernel_name(self) -> None:
        """Kernel names are sanitized for Jupyter compatibility."""
        assert NebiKernelSpecManager.clean_kernel_name("data-science") == "data-science"
        assert NebiKernelSpecManager.clean_kernel_name("my project!") == "my_project_"
        assert NebiKernelSpecManager.clean_kernel_name("café") == "cafe"

    def test_remote_workspace_is_marked_not_pulled(self) -> None:
        """Remote-only workspaces appear with remote-not-pulled state metadata."""
        remote = NebiWorkspace(
            name="remote-only",
            path="",
            remote_version="v5",
            environments=["default", "gpu"],
            source="remote",
        )
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[remote]),
        ):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-remote-only-default")

        assert "nebi-remote-only-gpu" in specs
        assert spec.metadata["nebi_state"] == "remote-not-pulled"
        assert spec.metadata["nebi_not_ready_reason"] == "workspace-not-pulled"
        assert spec.metadata["nebi"]["state"] == "remote-not-pulled"

    def test_local_workspace_state_not_installed(self) -> None:
        """Local workspace env is marked local-not-installed when probe fails install check."""
        local = NebiWorkspace(name="project", path="/tmp/project", local_version="v1")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
        ):
            mock_probe.return_value.installed = False
            mock_probe.return_value.missing_dependencies = []
            mock_probe.return_value.reason = "environment-not-installed"
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-project-default")

        assert spec.metadata["nebi_state"] == "local-not-installed"
        assert spec.metadata["nebi_not_ready_reason"] == "environment-not-installed"
        assert spec.metadata["nebi_logo_reason"] == "environment-not-installed"

    def test_local_workspace_state_missing_dependencies(self) -> None:
        """Local workspace env is marked local-missing-deps when required deps are absent."""
        local = NebiWorkspace(name="project", path="/tmp/project", local_version="v1")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
        ):
            mock_probe.return_value.installed = True
            mock_probe.return_value.missing_dependencies = ["ipykernel"]
            mock_probe.return_value.reason = None
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-project-default")

        assert spec.metadata["nebi_state"] == "local-missing-deps"
        assert spec.metadata["nebi_missing_dependencies"] == ["ipykernel"]
        assert spec.metadata["nebi_not_ready_reason"] == "missing-dependencies"

    def test_outdated_state_when_remote_version_differs(self) -> None:
        """Local workspace is marked outdated when local/ref and remote/ref drift."""
        local = NebiWorkspace(name="project", path="/tmp/project", local_version="v1")
        remote = NebiWorkspace(name="project", path="", remote_version="v2", source="remote")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[remote]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
        ):
            mock_probe.return_value.installed = True
            mock_probe.return_value.missing_dependencies = []
            mock_probe.return_value.reason = None
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-project-default")

        assert spec.metadata["nebi_state"] == "outdated"
        assert spec.metadata["nebi_local_version"] == "v1"
        assert spec.metadata["nebi_remote_version"] == "v2"
        assert spec.metadata["nebi_outdated"] is True

    def test_discovery_hash_and_timestamp_metadata(self) -> None:
        """Kernel metadata includes deterministic discovery hash and timestamp."""
        local = NebiWorkspace(name="project", path="/tmp/project")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
        ):
            mock_probe.return_value.installed = True
            mock_probe.return_value.missing_dependencies = []
            mock_probe.return_value.reason = None
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-project-default")

        discovery_hash = spec.metadata["nebi_discovery_hash"]
        discovered_at = spec.metadata["nebi_discovered_at"]
        assert isinstance(discovery_hash, str)
        assert len(discovery_hash) == 64
        assert all(c in "0123456789abcdef" for c in discovery_hash)
        assert isinstance(discovered_at, str)
        assert discovered_at.endswith("Z")
        assert "T" in discovered_at
