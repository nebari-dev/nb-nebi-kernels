"""Tests for the install CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nb_nebi_kernels.install import install, find_config_path, get_current_status


OUR_MANAGER = "nb_nebi_kernels.NebiKernelSpecManager"


class TestInstall:
    """Tests for enable/disable/status."""

    def test_enable_writes_config(self, tmp_path: Path) -> None:
        """--enable writes our manager class to jupyter_config.json."""
        config_dir = str(tmp_path)
        with patch("nb_nebi_kernels.install.jupyter_config_path", return_value=[config_dir]):
            result = install(enable=True, path=config_dir)

        assert result == 0
        config = json.loads((tmp_path / "jupyter_config.json").read_text())
        assert config["ServerApp"]["kernel_spec_manager_class"] == OUR_MANAGER

    def test_disable_removes_config(self, tmp_path: Path) -> None:
        """--disable removes our manager from jupyter_config.json."""
        config_dir = str(tmp_path)
        with patch("nb_nebi_kernels.install.jupyter_config_path", return_value=[config_dir]):
            install(enable=True, path=config_dir)
            result = install(disable=True, path=config_dir)

        assert result == 0
        config = json.loads((tmp_path / "jupyter_config.json").read_text())
        assert "kernel_spec_manager_class" not in config.get("ServerApp", {})

    def test_status_when_enabled(self, tmp_path: Path) -> None:
        """--status returns 0 and reports enabled."""
        config_dir = str(tmp_path)
        with patch("nb_nebi_kernels.install.jupyter_config_path", return_value=[config_dir]):
            install(enable=True, path=config_dir)
            result = install(status=True, path=config_dir)

        assert result == 0

    def test_status_when_disabled(self, tmp_path: Path) -> None:
        """--status returns 0 when not enabled."""
        config_dir = str(tmp_path)
        with patch("nb_nebi_kernels.install.jupyter_config_path", return_value=[config_dir]):
            result = install(status=True, path=config_dir)

        assert result == 0
