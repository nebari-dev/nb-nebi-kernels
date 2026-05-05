"""Tests for workspace and environment discovery."""

import json
from unittest.mock import MagicMock, patch

from nb_nebi_kernels.discovery import (
    EnvironmentProbe,
    discover_environments,
    discover_remote_workspaces,
    discover_workspaces,
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


class TestDiscoverRemoteWorkspaces:
    """Tests for discover_remote_workspaces()."""

    def test_discovers_remote_workspaces_from_api(self) -> None:
        """Uses NEBI_REMOTE_URL + NEBI_AUTH_TOKEN to discover remote workspaces."""
        workspaces_payload = json.dumps(
            [
                {"id": "ws-1", "name": "remote-a", "status": "ready"},
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
        assert workspaces[1].remote_version == "release-a"
        assert workspaces[1].environments == []

    def test_returns_empty_when_api_env_not_configured(self) -> None:
        """Remote discovery is API-only and returns empty without required env."""
        with patch.dict("os.environ", {}, clear=True):
            workspaces = discover_remote_workspaces()

        assert workspaces == []


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

    def test_detects_missing_ipykernel(self) -> None:
        """Probe reports missing launch dependencies."""
        mock_json = json.dumps([{"name": "python"}, {"name": "numpy"}])
        with (
            patch("nb_nebi_kernels.discovery.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.discovery.os.path.exists", return_value=True),
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/tmp/pixi.toml"),
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_json, stderr="")
            probe = probe_environment("/tmp/ws", "default")

        assert probe.installed is True
        assert probe.missing_dependencies == ["ipykernel"]

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

    def test_returns_default_when_pixi_not_found(self) -> None:
        """Falls back to ['default'] if pixi is not installed."""
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.side_effect = FileNotFoundError("pixi not found")
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]
