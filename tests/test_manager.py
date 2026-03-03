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
        """Display name is 'workspace (env)' for non-default, just 'workspace' for default when only env."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=sample_workspaces),
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
        with patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[]):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        assert not any(k.startswith("nebi-") for k in specs)

    def test_clean_kernel_name(self) -> None:
        """Kernel names are sanitized for Jupyter compatibility."""
        assert NebiKernelSpecManager.clean_kernel_name("data-science") == "data-science"
        assert NebiKernelSpecManager.clean_kernel_name("my project!") == "my_project_"
        assert NebiKernelSpecManager.clean_kernel_name("café") == "cafe"
