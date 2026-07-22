"""Tests for NebiKernelSpecManager."""

import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from jupyter_client.kernelspec import KernelSpec, NoSuchKernel

from nb_nebi_kernels.discovery import (
    EnvironmentKernelSpec,
    EnvironmentProbe,
    NebiWorkspace,
)
from nb_nebi_kernels.manager import NebiKernelSpecManager


def _installed_kernel(
    name: str = "python3",
    *,
    argv: list[str] | None = None,
    display_name: str = "Python 3",
    language: str = "python",
) -> EnvironmentKernelSpec:
    return EnvironmentKernelSpec(
        name=name,
        spec=KernelSpec(
            argv=argv or ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name=display_name,
            language=language,
            resource_dir=f"/tmp/kernels/{name}",
        ),
    )


@contextmanager
def _patched_discovery(
    workspaces: list[NebiWorkspace],
    envs_map: dict[str, list[str]],
    *,
    env_has_kernel: bool = True,
    kernel_specs: list[EnvironmentKernelSpec] | None = None,
) -> Iterator[None]:
    """Patch discovery + env probe so tests can control the kernelspec branch."""
    discovered_kernel_specs = kernel_specs if kernel_specs is not None else [_installed_kernel()]
    if not env_has_kernel:
        discovered_kernel_specs = []

    with (
        patch("nb_nebi_kernels.manager.discover_workspaces", return_value=workspaces),
        patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
        patch(
            "nb_nebi_kernels.manager.discover_environments",
            side_effect=lambda p: envs_map[p],
        ),
        patch(
            "nb_nebi_kernels.manager.probe_environment",
            return_value=EnvironmentProbe(installed=True, missing_dependencies=[]),
        ),
        patch(
            "nb_nebi_kernels.manager.discover_kernel_specs",
            return_value=discovered_kernel_specs,
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
            "python",
            "-m",
            "ipykernel_launcher",
            "-f",
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
        assert spec.metadata["nebi_kernel_state"] == "remote-not-pulled"
        assert spec.metadata["nebi_not_ready_reason"] == "workspace-not-pulled"
        assert spec.argv == [
            sys.executable,
            "-m",
            "nb_nebi_kernels.stub_kernel",
            "--workspace",
            "remote-only",
            "--env",
            "default",
            "--reason",
            "workspace-not-pulled",
            "-f",
            "{connection_file}",
        ]

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
        assert "nebi_logo_reason" not in spec.metadata

    def test_local_workspace_install_status_is_metadata_only(self) -> None:
        """Explicit nebi install status does not override local env probing."""
        local = NebiWorkspace(
            name="project",
            path="/tmp/project",
            local_version="v1",
            install_status="not_installed",
        )
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
            patch(
                "nb_nebi_kernels.manager.discover_kernel_specs",
                return_value=[_installed_kernel()],
            ),
        ):
            mock_probe.return_value.installed = True
            mock_probe.return_value.missing_dependencies = []
            mock_probe.return_value.reason = None
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-project-default")

        mock_probe.assert_called_once_with("/tmp/project", "default", ())
        assert spec.metadata["nebi_state"] == "ready"
        assert spec.metadata["nebi_install_status"] == "not_installed"

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
        assert "--reason" in spec.argv
        assert "missing-dependencies" in spec.argv
        assert "--missing-dependency" in spec.argv
        assert "ipykernel" in spec.argv
        assert "nebi_logo_reason" not in spec.metadata

    def test_outdated_state_when_remote_version_differs(self) -> None:
        """Local workspace is marked outdated when local/ref and remote/ref drift."""
        local = NebiWorkspace(name="project", path="/tmp/project", local_version="v1")
        remote = NebiWorkspace(name="project", path="", remote_version="v2", source="remote")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[remote]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
            patch(
                "nb_nebi_kernels.manager.discover_kernel_specs",
                return_value=[_installed_kernel()],
            ),
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
        assert "nebi_logo_reason" not in spec.metadata

    def test_discovery_hash_and_timestamp_metadata(self) -> None:
        """Kernel metadata includes deterministic discovery hash and timestamp."""
        local = NebiWorkspace(name="project", path="/tmp/project")
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
            patch(
                "nb_nebi_kernels.manager.discover_kernel_specs",
                return_value=[_installed_kernel()],
            ),
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

    def test_local_workspace_merges_remote_environment_variants(self) -> None:
        """Local workspace kernels include remote-only environment variants by name."""
        local = NebiWorkspace(name="project", path="/tmp/project", local_version="v1")
        remote = NebiWorkspace(
            name="project",
            path="",
            remote_version="v2",
            environments=["default", "gpu"],
            source="remote",
        )
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces", return_value=[local]),
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[remote]),
            patch("nb_nebi_kernels.manager.discover_environments", return_value=["default"]),
            patch("nb_nebi_kernels.manager.probe_environment") as mock_probe,
            patch(
                "nb_nebi_kernels.manager.discover_kernel_specs",
                return_value=[_installed_kernel()],
            ),
        ):
            mock_probe.return_value.installed = True
            mock_probe.return_value.missing_dependencies = []
            mock_probe.return_value.reason = None
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        assert "nebi-project-default" in specs
        assert "nebi-project-gpu" in specs

    def test_find_kernel_specs_uses_discovery_cache(self) -> None:
        """Repeated find calls within TTL reuse cached discovery results."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces") as mock_local,
            patch("nb_nebi_kernels.manager.discover_remote_workspaces") as mock_remote,
        ):
            mock_local.return_value = [NebiWorkspace(name="project", path="/tmp/project")]
            mock_remote.return_value = []

            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            manager.find_kernel_specs()

        assert mock_local.call_count == 1
        assert mock_remote.call_count == 1

    def test_parent_kernel_lookup_does_not_force_discovery_refresh(self) -> None:
        """Parent kernel misses use the cache instead of forcing discovery every time."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces") as mock_local,
            patch("nb_nebi_kernels.manager.discover_remote_workspaces") as mock_remote,
            patch.object(
                NebiKernelSpecManager.__bases__[0],
                "get_kernel_spec",
                side_effect=NoSuchKernel("python3"),
            ),
        ):
            mock_local.return_value = [NebiWorkspace(name="project", path="/tmp/project")]
            mock_remote.return_value = []

            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            with pytest.raises(NoSuchKernel):
                manager.get_kernel_spec("python3")

        assert mock_local.call_count == 1
        assert mock_remote.call_count == 1

    def test_parent_kernel_lookup_skips_nebi_discovery(self) -> None:
        """Parent kernel lookups do not run Nebi discovery."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces") as mock_local,
            patch("nb_nebi_kernels.manager.discover_remote_workspaces") as mock_remote,
            patch.object(
                NebiKernelSpecManager.__bases__[0],
                "get_kernel_spec",
                side_effect=NoSuchKernel("python3"),
            ),
        ):
            manager = NebiKernelSpecManager()
            with pytest.raises(NoSuchKernel):
                manager.get_kernel_spec("python3")

        mock_local.assert_not_called()
        mock_remote.assert_not_called()

    def test_nebi_cache_miss_forces_one_refresh(self) -> None:
        """Fresh cache misses for Nebi kernels still force discovery for new workspaces."""
        envs_map = {
            "/tmp/old": ["default"],
            "/tmp/new": ["default"],
        }
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces") as mock_local,
            patch("nb_nebi_kernels.manager.discover_remote_workspaces", return_value=[]),
            patch(
                "nb_nebi_kernels.manager.discover_environments",
                side_effect=lambda p: envs_map[p],
            ),
            patch(
                "nb_nebi_kernels.manager.probe_environment",
                return_value=EnvironmentProbe(installed=True, missing_dependencies=[]),
            ),
            patch(
                "nb_nebi_kernels.manager.discover_kernel_specs",
                return_value=[_installed_kernel()],
            ),
        ):
            mock_local.side_effect = [
                [NebiWorkspace(name="old", path="/tmp/old")],
                [NebiWorkspace(name="new", path="/tmp/new")],
            ]

            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-new-default")

        assert mock_local.call_count == 2
        assert spec.metadata["nebi_workspace"] == "new"

    def test_invalidate_discovery_cache_forces_refresh(self) -> None:
        """Manual cache invalidation forces discovery on the next lookup."""
        with (
            patch("nb_nebi_kernels.manager.discover_workspaces") as mock_local,
            patch("nb_nebi_kernels.manager.discover_remote_workspaces") as mock_remote,
        ):
            mock_local.return_value = [NebiWorkspace(name="project", path="/tmp/project")]
            mock_remote.return_value = []

            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            manager.invalidate_discovery_cache()
            manager.find_kernel_specs()

        assert mock_local.call_count == 2
        assert mock_remote.call_count == 2


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
            "--reason",
            "kernel-not-installed",
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
        """Stub kernelspecs carry the resolved state for tooling."""
        with _patched_discovery(workspaces, envs_map, env_has_kernel=False):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.metadata["nebi_kernel_state"] == "local-missing-deps"
        assert spec.metadata["nebi_missing_dependencies"] == []
        assert spec.metadata["nebi_not_ready_reason"] == "kernel-not-installed"

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

    def test_non_python_kernel_uses_installed_kernelspec(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """R and other kernels launch their own argv instead of ipykernel."""
        r_kernel = _installed_kernel(
            "ir",
            argv=["R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"],
            display_name="R",
            language="R",
        )
        r_kernel.spec.env = {"R_LIBS_USER": "$HOME/R"}
        r_kernel.spec.metadata = {"debugger": False}
        r_kernel.spec.interrupt_mode = "message"

        with _patched_discovery(workspaces, envs_map, kernel_specs=[r_kernel]):
            manager = NebiKernelSpecManager()
            manager.find_kernel_specs()
            spec = manager.get_kernel_spec("nebi-data-science-gpu")

        assert spec.argv == [
            sys.executable,
            "-m",
            "nb_nebi_kernels.launcher",
            "/home/user/data-science",
            "gpu",
            "R",
            "--slave",
            "-e",
            "IRkernel::main()",
            "--args",
            "{connection_file}",
        ]
        assert spec.language == "R"
        assert spec.env["R_LIBS_USER"] == "$HOME/R"
        assert spec.metadata["debugger"] is False
        assert spec.interrupt_mode == "message"
        assert spec.resource_dir == "/tmp/kernels/ir"

    def test_multiple_kernels_are_exposed_deterministically(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """The primary keeps the existing name and additional kernels get suffixes."""
        kernels = [
            _installed_kernel(),
            _installed_kernel(
                "ir",
                argv=["R", "--args", "{connection_file}"],
                display_name="R",
                language="R",
            ),
        ]
        with _patched_discovery(workspaces, envs_map, kernel_specs=kernels):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()
            python_spec = manager.get_kernel_spec("nebi-data-science-gpu")
            r_spec = manager.get_kernel_spec("nebi-data-science-gpu-ir")

        assert "nebi-data-science-gpu" in specs
        assert "nebi-data-science-gpu-ir" in specs
        assert python_spec.display_name == "data-science (gpu) — Python 3"
        assert r_spec.display_name == "data-science (gpu) — R"
        assert r_spec.metadata["nebi_kernel_spec"] == "ir"

    def test_colliding_kernel_names_are_not_dropped(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Sanitized kernelspec name collisions get deterministic fallback names."""
        kernels = [
            _installed_kernel(),
            _installed_kernel(
                "foo bar",
                argv=["foo-a", "{connection_file}"],
                display_name="Foo A",
                language="foo",
            ),
            _installed_kernel(
                "foo?bar",
                argv=["foo-b", "{connection_file}"],
                display_name="Foo B",
                language="foo",
            ),
        ]
        with _patched_discovery(workspaces, envs_map, kernel_specs=kernels):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        colliding_names = [
            name for name in specs if name.startswith("nebi-data-science-gpu-foo_bar")
        ]

        assert "nebi-data-science-gpu-foo_bar" in colliding_names
        assert len(colliding_names) == 2
        assert any(
            re.fullmatch(r"nebi-data-science-gpu-foo_bar-[0-9a-f]{8}", name)
            for name in colliding_names
        )

    def test_duplicate_kernel_entries_are_skipped(
        self, workspaces: list[NebiWorkspace], envs_map: dict[str, list[str]]
    ) -> None:
        """Duplicate source kernels are not exposed again with another suffix."""
        kernels = [
            _installed_kernel(),
            _installed_kernel(
                "foo bar",
                argv=["foo-a", "{connection_file}"],
                display_name="Foo A",
                language="foo",
            ),
            _installed_kernel(
                "foo bar",
                argv=["foo-a", "{connection_file}"],
                display_name="Foo A",
                language="foo",
            ),
        ]
        with _patched_discovery(workspaces, envs_map, kernel_specs=kernels):
            manager = NebiKernelSpecManager()
            specs = manager.find_kernel_specs()

        duplicate_names = [
            name for name in specs if name.startswith("nebi-data-science-gpu-foo_bar")
        ]

        assert duplicate_names == ["nebi-data-science-gpu-foo_bar"]
