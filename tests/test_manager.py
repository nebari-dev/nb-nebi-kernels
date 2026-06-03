"""Tests for NebiKernelSpecManager."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from nb_nebi_kernels.discovery import NebiWorkspace
from nb_nebi_kernels.manager import NebiKernelSpecManager


@contextmanager
def _patched_discovery(
    workspaces: list[NebiWorkspace],
    envs_map: dict[str, list[str]],
    *,
    env_has_kernel: bool = True,
) -> Iterator[None]:
    """Patch discovery + env probe so tests can control the kernelspec branch."""
    with (
        patch("nb_nebi_kernels.manager.discover_workspaces", return_value=workspaces),
        patch(
            "nb_nebi_kernels.manager.discover_environments",
            side_effect=lambda p: envs_map[p],
        ),
        patch(
            "nb_nebi_kernels.manager.env_has_any_kernelspec",
            return_value=env_has_kernel,
        ),
    ):
        yield


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
        with _patched_discovery(sample_workspaces, sample_envs_map):
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
            _patched_discovery(sample_workspaces, sample_envs_map),
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
        with _patched_discovery(sample_workspaces, sample_envs_map):
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
        with _patched_discovery(sample_workspaces, sample_envs_map):
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
        with _patched_discovery(sample_workspaces, sample_envs_map):
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


class TestMissingKernelBranch:
    """KernelSpec generation for envs that have no Jupyter kernel installed."""

    @pytest.fixture
    def workspaces(self) -> list[NebiWorkspace]:
        return [NebiWorkspace(name="data-science", path="/home/user/data-science")]

    @pytest.fixture
    def envs_map(self) -> dict[str, list[str]]:
        return {"/home/user/data-science": ["default", "gpu"]}

    def test_argv_targets_stub_kernel(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """When env has no kernel, argv invokes nb_nebi_kernels.stub_kernel."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=False):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.argv == [
            sys.executable,
            "-m",
            "nb_nebi_kernels.stub_kernel",
            "--workspace",
            "data-science",
            "--env",
            "gpu",
            "-f",
            "{connection_file}",
        ]

    def test_display_name_undecorated(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Missing-kernel envs keep the normal display name (no marker)."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=False):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.display_name == "data-science (gpu)"
        assert "— no kernel installed" not in spec.display_name

    def test_metadata_state_flag(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Stub kernelspecs carry nebi_kernel_state=missing-kernel for tooling."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=False):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.metadata["nebi_kernel_state"] == "missing-kernel"

    def test_metadata_install_command(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Stub kernelspecs carry a copy-pasteable pixi install command."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=False):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.metadata["nebi_install_command"] == (
            "pixi add --manifest-path /home/user/data-science/pixi.toml "
            "--feature gpu ipykernel"
        )

    def test_working_envs_get_ready_state(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Working envs carry nebi_kernel_state=ready in metadata."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=True):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.metadata["nebi_kernel_state"] == "ready"
        assert "— no kernel installed" not in spec.display_name
