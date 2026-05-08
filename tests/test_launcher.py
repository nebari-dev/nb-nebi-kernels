"""Tests for the kernel launcher."""

import os
from unittest.mock import patch

import pytest

from nb_nebi_kernels.launcher import PIXI_ENV_VARS_TO_CLEAR, main


class TestLauncher:
    """Tests for the launcher main() function."""

    def test_clears_pixi_env_vars(self) -> None:
        """Launcher clears all PIXI_* env vars before exec."""
        cleared: list[str] = []

        def fake_pop(var: str, *args: object) -> None:
            cleared.append(var)

        with (
            patch("sys.argv", ["launcher", "/tmp/ws", "default", "/tmp/conn.json"]),
            patch.dict(os.environ, {v: "val" for v in PIXI_ENV_VARS_TO_CLEAR}),
            patch("nb_nebi_kernels.launcher.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.launcher.os.chdir"),
            patch("nb_nebi_kernels.launcher.os.execvp"),
        ):
            main()

            # After main() runs, the pixi env vars should have been removed
            # (check inside patch.dict context, before it restores values)
            for var in PIXI_ENV_VARS_TO_CLEAR:
                assert var not in os.environ

    def test_execs_pixi_run_with_environment(self) -> None:
        """Launcher execs pixi run with the correct environment flag."""
        with (
            patch("sys.argv", ["launcher", "/tmp/ws", "gpu", "/tmp/conn.json"]),
            patch("nb_nebi_kernels.launcher.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.launcher.os.chdir"),
            patch("nb_nebi_kernels.launcher.os.execvp") as mock_exec,
        ):
            main()

        mock_exec.assert_called_once_with(
            "pixi",
            [
                "pixi",
                "run",
                "--frozen",
                "--manifest-path",
                "/tmp/ws/pixi.toml",
                "-e",
                "gpu",
                "python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "/tmp/conn.json",
            ],
        )

    def test_execs_pixi_run_default_env(self) -> None:
        """Launcher with 'default' environment omits -e flag."""
        with (
            patch("sys.argv", ["launcher", "/tmp/ws", "default", "/tmp/conn.json"]),
            patch("nb_nebi_kernels.launcher.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.launcher.os.chdir"),
            patch("nb_nebi_kernels.launcher.os.execvp") as mock_exec,
        ):
            main()

        mock_exec.assert_called_once_with(
            "pixi",
            [
                "pixi",
                "run",
                "--frozen",
                "--manifest-path",
                "/tmp/ws/pixi.toml",
                "python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "/tmp/conn.json",
            ],
        )

    def test_changes_to_manifest_dir(self) -> None:
        """Launcher changes to the workspace directory."""
        with (
            patch("sys.argv", ["launcher", "/tmp/ws", "default", "/tmp/conn.json"]),
            patch("nb_nebi_kernels.launcher.os.path.isdir", return_value=True),
            patch("nb_nebi_kernels.launcher.os.chdir") as mock_chdir,
            patch("nb_nebi_kernels.launcher.os.execvp"),
        ):
            main()

        mock_chdir.assert_called_once_with("/tmp/ws")

    def test_exits_on_wrong_args(self) -> None:
        """Launcher exits with code 1 if wrong number of args."""
        with (
            patch("sys.argv", ["launcher", "/tmp/ws"]),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_exits_for_non_ready_kernel_state(self) -> None:
        """Launcher exits early with a clear error for blocked states."""
        with (
            patch("sys.argv", ["launcher", "/tmp/ws", "default", "/tmp/conn.json"]),
            patch.dict(os.environ, {"NB_NEBI_KERNEL_STATE": "remote-not-pulled"}),
            pytest.raises(SystemExit, match="1"),
        ):
            main()
