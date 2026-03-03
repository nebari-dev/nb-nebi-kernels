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


def discover_environments(workspace_path: str) -> list[str]:
    """Discover pixi environments in a workspace.

    Calls ``pixi workspace environment list`` for the given workspace.

    Args:
        workspace_path: Absolute path to the workspace directory.

    Returns:
        List of environment names. Falls back to ``["default"]``
        if pixi is not installed or the command fails.
    """
    try:
        result = subprocess.run(
            [
                "pixi",
                "workspace",
                "environment",
                "list",
                "--manifest-path",
                f"{workspace_path}/pixi.toml",
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

    envs = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return envs if envs else ["default"]
