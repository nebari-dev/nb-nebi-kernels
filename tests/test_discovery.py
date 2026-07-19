"""Tests for workspace and environment discovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nb_nebi_kernels.discovery import (
    EnvironmentProbe,
    _parse_pixi_toml_environments,
    discover_environments,
    discover_kernel_specs,
    discover_remote_workspaces,
    discover_workspaces,
    env_has_any_kernelspec,
    probe_environment,
)


class TestDiscoverWorkspaces:
    """Tests for discover_workspaces()."""

    def test_parses_nebi_json_output(self) -> None:
        """Parses JSON from nebi workspace list --json."""
        mock_json = json.dumps([
            {
                "name": "data-science",
                "path": "/home/user/data-science",
                "origin_id": "ws-123",
                "origin_tag": "v2",
                "install_status": "installed",
                "missing": False,
            },
            {"name": "web-app", "path": "/home/user/web-app", "missing": False},
        ])
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            workspaces = discover_workspaces()

        assert len(workspaces) == 2
        assert workspaces[0].name == "data-science"
        assert workspaces[0].path == "/home/user/data-science"
        assert workspaces[0].local_version == "v2"
        assert workspaces[0].install_status == "installed"
        assert workspaces[1].name == "web-app"
        assert workspaces[1].path == "/home/user/web-app"

    def test_filters_missing_workspaces(self) -> None:
        """Workspaces with missing=true are excluded."""
        mock_json = json.dumps([
            {"name": "data-science", "path": "/home/user/data-science", "missing": False},
            {"name": "old-project", "path": "/home/user/old-project", "missing": True},
        ])
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            workspaces = discover_workspaces()

        assert len(workspaces) == 1
        assert workspaces[0].name == "data-science"

    def test_returns_empty_when_nebi_not_found(self) -> None:
        """Returns empty list if nebi CLI is not installed."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nebi not found")
            workspaces = discover_workspaces()

        assert workspaces == []

    def test_returns_empty_on_nebi_error(self) -> None:
        """Returns empty list if nebi exits with error."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="some error"
            )
            workspaces = discover_workspaces()

        assert workspaces == []

    def test_returns_empty_when_no_workspaces(self) -> None:
        """Returns empty list when nebi returns empty JSON array."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="[]", stderr=""
            )
            workspaces = discover_workspaces()

        assert workspaces == []


class TestDiscoverEnvironments:
    """Tests for discover_environments()."""

    def test_lists_environments_for_workspace(self) -> None:
        """Parses pixi info --json to extract environment names."""
        mock_json = json.dumps({
            "environments_info": [
                {"name": "default", "features": ["default"]},
                {"name": "gpu", "features": ["gpu", "default"]},
            ]
        })
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default", "gpu"]

    def test_returns_default_on_error(self) -> None:
        """Falls back to ['default'] if pixi command fails."""
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]

    def test_returns_default_when_pixi_not_found(self) -> None:
        """Falls back to ['default'] if pixi is not installed."""
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.side_effect = FileNotFoundError("pixi not found")
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]


class TestDiscoverRemoteWorkspaces:
    """Tests for discover_remote_workspaces()."""

    def test_discovers_remote_workspaces_from_api(self) -> None:
        """Uses NEBI_REMOTE_URL + NEBI_AUTH_TOKEN to discover remote workspaces."""
        workspaces_payload = json.dumps(
            [
                {
                    "id": "ws-1",
                    "name": "remote-a",
                    "status": "ready",
                    "install_status": "not_installed",
                },
                {"id": "ws-2", "name": "remote-b", "status": "ready"},
            ]
        ).encode("utf-8")
        pixi_payload_by_workspace = {
            "ws-1": json.dumps(
                {
                    "content": """
[project]
name = "remote-a"

[environments]
default = {features = ["default"]}
gpu = {features = ["gpu", "default"]}
""".strip()
                }
            ).encode("utf-8"),
            "ws-2": json.dumps({"content": ""}).encode("utf-8"),
        }
        tags_payload_by_workspace = {
            "ws-1": json.dumps(
                [
                    {
                        "tag": "v1",
                        "version_number": 1,
                        "updated_at": "2026-01-01T00:00:00Z",
                    },
                    {
                        "tag": "v2",
                        "version_number": 2,
                        "updated_at": "2026-01-02T00:00:00Z",
                    },
                ]
            ).encode("utf-8"),
            "ws-2": json.dumps(
                [
                    {
                        "tag": "release-a",
                        "version_number": 1,
                        "updated_at": "2026-01-03T00:00:00Z",
                    }
                ]
            ).encode("utf-8"),
        }

        class _Response:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return self._payload

        def _fake_urlopen(req: object, timeout: int = 0) -> _Response:
            full_url = getattr(req, "full_url", "")
            if full_url.endswith("/api/v1/workspaces"):
                return _Response(workspaces_payload)
            if full_url.endswith("/api/v1/workspaces/ws-1/tags"):
                return _Response(tags_payload_by_workspace["ws-1"])
            if full_url.endswith("/api/v1/workspaces/ws-2/tags"):
                return _Response(tags_payload_by_workspace["ws-2"])
            if full_url.endswith("/api/v1/workspaces/ws-1/pixi-toml"):
                return _Response(pixi_payload_by_workspace["ws-1"])
            if full_url.endswith("/api/v1/workspaces/ws-2/pixi-toml"):
                return _Response(pixi_payload_by_workspace["ws-2"])
            raise AssertionError(f"Unexpected URL requested: {full_url}")

        with (
            patch.dict(
                "os.environ",
                {
                    "NEBI_REMOTE_URL": "https://nebi.example.com",
                    "NEBI_AUTH_TOKEN": "token",
                },
                clear=False,
            ),
            patch("nb_nebi_kernels.discovery.urlopen", side_effect=_fake_urlopen),
        ):
            workspaces = discover_remote_workspaces()

        assert [ws.name for ws in workspaces] == ["remote-a", "remote-b"]
        assert workspaces[0].source == "remote"
        assert workspaces[0].remote_version == "v2"
        assert workspaces[0].environments == ["default", "gpu"]
        assert workspaces[0].install_status == "not_installed"
        assert workspaces[1].remote_version == "release-a"
        assert workspaces[1].environments == []

    def test_returns_empty_when_api_env_not_configured(self) -> None:
        """Remote discovery is API-only and returns empty without required env."""
        with patch.dict("os.environ", {}, clear=True):
            workspaces = discover_remote_workspaces()

        assert workspaces == []

    def test_pixi_toml_environment_parser_handles_structured_toml(self) -> None:
        """TOML parsing handles quoted tables and pyproject pixi environment tables."""
        environments = _parse_pixi_toml_environments("""
[environments]
default = { features = ["default"] }
"gpu env" = { features = [
    "gpu",
    "default",
] }

[environments."quoted table"]
features = ["docs"]

[tool.pixi.environments]
docs = { features = ["docs"] }
""")

        assert environments == ["default", "gpu env", "quoted table", "docs"]


class TestProbeEnvironment:
    """Tests for probe_environment()."""

    def test_returns_not_installed_when_workspace_missing(self) -> None:
        """Probe reports not installed for missing workspace paths."""
        probe = probe_environment("/does/not/exist", "default")
        assert probe == EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason="workspace-missing",
        )

    def test_returns_not_installed_when_env_prefix_missing(self, tmp_path: Path) -> None:
        """Probe uses the env prefix on disk to detect explicit pixi installs."""
        (tmp_path / "pixi.toml").write_text("[project]\nname = 'demo'\n")
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            probe = probe_environment(str(tmp_path), "default")

        assert probe == EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason="environment-not-installed",
        )
        mock_run.assert_not_called()

    def test_detects_missing_ipykernel(self) -> None:
        """Probe reports explicitly configured missing launch dependencies."""
        mock_json = json.dumps([{"name": "python"}, {"name": "numpy"}])
        with (
            patch("nb_nebi_kernels.discovery.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.discovery.os.path.exists", return_value=True),
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/tmp/pixi.toml"),
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_json, stderr="")
            probe = probe_environment("/tmp/ws", "default", ("ipykernel",))

        assert probe.installed is True
        assert probe.missing_dependencies == ["ipykernel"]

    def test_does_not_require_ipykernel_by_default(self) -> None:
        """An installed non-Python kernel environment is not blocked by package name."""
        with (
            patch("nb_nebi_kernels.discovery.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.discovery.os.path.exists", return_value=True),
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/tmp/pixi.toml"),
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps([{"name": "r-base"}, {"name": "r-irkernel"}]),
                stderr="",
            )
            probe = probe_environment("/tmp/ws", "default")

        assert probe.installed is True
        assert probe.missing_dependencies == []

    def test_nonzero_pixi_list_returns_stable_reason(self) -> None:
        """Probe uses stable enum reason for generic pixi list failures."""
        with (
            patch("nb_nebi_kernels.discovery.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.discovery.os.path.exists", return_value=True),
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/tmp/pixi.toml"),
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="random dynamic pixi failure text",
            )
            probe = probe_environment("/tmp/ws", "default")

        assert probe.installed is False
        assert probe.reason == "pixi-list-failed"


class TestEnvHasAnyKernelspec:
    """Tests for env_has_any_kernelspec()."""

    def test_true_when_kernelspec_present(self, tmp_path: Path) -> None:
        """Detects a kernel.json under share/jupyter/kernels/."""
        kernels = tmp_path / ".pixi" / "envs" / "default" / "share" / "jupyter" / "kernels"
        (kernels / "python3").mkdir(parents=True)
        (kernels / "python3" / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Python 3",
                    "language": "python",
                }
            )
        )

        assert env_has_any_kernelspec(str(tmp_path), "default") is True

    def test_false_when_env_prefix_missing(self, tmp_path: Path) -> None:
        """Returns False when the env was never installed."""
        assert env_has_any_kernelspec(str(tmp_path), "default") is False

    def test_false_when_kernels_dir_empty(self, tmp_path: Path) -> None:
        """Returns False when share/jupyter/kernels/ has no kernelspecs."""
        kernels = tmp_path / ".pixi" / "envs" / "default" / "share" / "jupyter" / "kernels"
        kernels.mkdir(parents=True)
        assert env_has_any_kernelspec(str(tmp_path), "default") is False

    def test_finds_non_python_kernel(self, tmp_path: Path) -> None:
        """Returns True for any kernelspec, not just python3."""
        kernels = tmp_path / ".pixi" / "envs" / "gpu" / "share" / "jupyter" / "kernels"
        (kernels / "ir").mkdir(parents=True)
        (kernels / "ir" / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        "R",
                        "--slave",
                        "-e",
                        "IRkernel::main()",
                        "--args",
                        "{connection_file}",
                    ],
                    "display_name": "R",
                    "language": "R",
                }
            )
        )

        assert env_has_any_kernelspec(str(tmp_path), "gpu") is True

    def test_discovers_all_kernelspecs_in_deterministic_order(self, tmp_path: Path) -> None:
        """Python remains the backwards-compatible primary, then names are sorted."""
        kernels = tmp_path / ".pixi" / "envs" / "default" / "share" / "jupyter" / "kernels"
        specs = {
            "zsh": {
                "argv": ["zsh", "-c", "kernel", "{connection_file}"],
                "display_name": "Z shell",
                "language": "zsh",
            },
            "python3": {
                "argv": ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Python 3",
                "language": "python",
            },
            "ir": {
                "argv": ["R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"],
                "display_name": "R",
                "language": "R",
            },
        }
        for name, spec in specs.items():
            resource_dir = kernels / name
            resource_dir.mkdir(parents=True)
            (resource_dir / "kernel.json").write_text(json.dumps(spec))

        discovered = discover_kernel_specs(str(tmp_path), "default")

        assert [item.name for item in discovered] == ["python3", "ir", "zsh"]
        assert discovered[1].spec.language == "R"
