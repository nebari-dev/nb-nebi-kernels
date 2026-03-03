"""Tests for workspace and environment discovery."""

import json
from unittest.mock import MagicMock, patch

from nb_nebi_kernels.discovery import (
    discover_environments,
    discover_workspaces,
)


class TestDiscoverWorkspaces:
    """Tests for discover_workspaces()."""

    def test_parses_nebi_json_output(self) -> None:
        """Parses JSON from nebi workspace list --json."""
        mock_json = json.dumps([
            {"name": "data-science", "path": "/home/user/data-science", "missing": False},
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
