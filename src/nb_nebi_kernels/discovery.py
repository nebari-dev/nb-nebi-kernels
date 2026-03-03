"""Discover nebi workspaces and their pixi environments."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NebiWorkspace:
    """A locally-tracked nebi workspace."""

    name: str
    path: str


def discover_workspaces() -> list[NebiWorkspace]:
    """Discover locally-tracked nebi workspaces.

    Calls ``nebi workspace list`` and parses the table output.
    Filters out workspaces marked as missing.

    Returns:
        List of discovered workspaces. Empty list if nebi is not
        installed or an error occurs.
    """
    try:
        result = subprocess.run(
            ["nebi", "workspace", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("nebi CLI not found on PATH — no nebi kernels will be available")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nebi workspace list timed out")
        return []

    if result.returncode != 0:
        logger.warning("nebi workspace list failed: %s", result.stderr.strip())
        return []

    return _parse_workspace_list(result.stdout)


def _parse_workspace_list(output: str) -> list[NebiWorkspace]:
    """Parse the table output of ``nebi workspace list``.

    Expected format::

        NAME        PATH
        workspace1  /path/to/workspace1
        workspace2  /path/to/workspace2 (missing)

    Workspaces with ``(missing)`` suffix on the path are filtered out.
    """
    workspaces: list[NebiWorkspace] = []

    lines = output.strip().splitlines()
    if len(lines) < 2:
        return workspaces

    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue

        # Split on whitespace — name is first token, path is the rest
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue

        name = parts[0]
        path = parts[1].strip()

        # Filter out missing workspaces
        if path.endswith("(missing)"):
            logger.debug("Skipping missing workspace: %s", name)
            continue

        workspaces.append(NebiWorkspace(name=name, path=path))

    return workspaces


def _find_manifest(workspace_path: str) -> str:
    """Find the pixi manifest file in a workspace directory.

    Checks for ``pixi.toml`` first, then ``pyproject.toml``.
    """
    import os

    for name in ("pixi.toml", "pyproject.toml"):
        path = os.path.join(workspace_path, name)
        if os.path.exists(path):
            return path
    return os.path.join(workspace_path, "pixi.toml")


def discover_environments(workspace_path: str) -> list[str]:
    """Discover pixi environments in a workspace.

    Calls ``pixi workspace environment list`` for the given workspace.

    Args:
        workspace_path: Absolute path to the workspace directory.

    Returns:
        List of environment names. Falls back to ``["default"]``
        if pixi is not installed or the command fails.
    """
    manifest = _find_manifest(workspace_path)

    try:
        result = subprocess.run(
            [
                "pixi",
                "workspace",
                "environment",
                "list",
                "--manifest-path",
                manifest,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("pixi CLI not found on PATH — falling back to default environment")
        return ["default"]
    except subprocess.TimeoutExpired:
        logger.warning("pixi workspace environment list timed out for %s", workspace_path)
        return ["default"]

    if result.returncode != 0:
        logger.debug(
            "pixi workspace environment list failed for %s: %s",
            workspace_path,
            result.stderr.strip(),
        )
        return ["default"]

    return _parse_environment_list(result.stdout)


def _parse_environment_list(output: str) -> list[str]:
    """Parse the output of ``pixi workspace environment list``.

    Expected format::

        Environments:
        - default:
            features: default
        - gpu:
            features: gpu, default

    Extracts environment names from lines matching ``- name:``.
    """
    import re

    envs: list[str] = []
    for line in output.splitlines():
        match = re.match(r"^- (\S+?):\s*$", line.strip())
        if match:
            envs.append(match.group(1))

    return envs if envs else ["default"]
