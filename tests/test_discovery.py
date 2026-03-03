"""Tests for workspace and environment discovery."""

from unittest.mock import MagicMock, patch

from nb_nebi_kernels.discovery import (
    discover_environments,
    discover_workspaces,
)


class TestDiscoverWorkspaces:
    """Tests for discover_workspaces()."""

    def test_parses_nebi_workspace_list_output(self) -> None:
        """Parses the NAME/PATH table from nebi workspace list."""
        mock_output = (
            "NAME\tPATH\n"
            "data-science\t/home/user/data-science\n"
            "web-app\t/home/user/web-app\n"
        )
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_output, stderr=""
            )
            workspaces = discover_workspaces()

        assert len(workspaces) == 2
        assert workspaces[0].name == "data-science"
        assert workspaces[0].path == "/home/user/data-science"
        assert workspaces[1].name == "web-app"
        assert workspaces[1].path == "/home/user/web-app"

    def test_filters_missing_workspaces(self) -> None:
        """Workspaces marked (missing) are excluded."""
        mock_output = (
            "NAME\tPATH\n"
            "data-science\t/home/user/data-science\n"
            "old-project\t/home/user/old-project (missing)\n"
        )
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_output, stderr=""
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
        """Returns empty list when nebi has no tracked workspaces."""
        mock_output = "No tracked workspaces. Run 'nebi init' in a pixi workspace to get started.\n"
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=mock_output
            )
            workspaces = discover_workspaces()

        assert workspaces == []


class TestDiscoverEnvironments:
    """Tests for discover_environments()."""

    def test_lists_environments_for_workspace(self) -> None:
        """Calls pixi workspace environment list and parses output."""
        mock_output = "default\ngpu\n"
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_output, stderr=""
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default", "gpu"]

    def test_returns_default_on_error(self) -> None:
        """Falls back to ['default'] if pixi command fails."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]

    def test_returns_default_when_pixi_not_found(self) -> None:
        """Falls back to ['default'] if pixi is not installed."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("pixi not found")
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]
