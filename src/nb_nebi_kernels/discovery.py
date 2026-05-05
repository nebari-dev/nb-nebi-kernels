"""Discover nebi workspaces, remote metadata, and pixi environment health."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PROBE_REASON_WORKSPACE_MISSING = "workspace-missing"
PROBE_REASON_MANIFEST_MISSING = "manifest-missing"
PROBE_REASON_PIXI_MISSING = "pixi-missing"
PROBE_REASON_PIXI_TIMEOUT = "pixi-timeout"
PROBE_REASON_PIXI_LIST_FAILED = "pixi-list-failed"
PROBE_REASON_PIXI_JSON_PARSE_FAILED = "pixi-json-parse-failed"


@dataclass
class NebiWorkspace:
    """A discovered nebi workspace."""

    name: str
    path: str
    local_version: str | None = None
    remote_version: str | None = None
    environments: list[str] | None = None
    source: str = "local"


@dataclass
class EnvironmentProbe:
    """Resolved installation and dependency state for an environment."""

    installed: bool
    missing_dependencies: list[str]
    reason: str | None = None


def _parse_local_version(ws: dict[str, Any]) -> str | None:
    """Parse local pulled tag/version from `nebi workspace list --json` output."""
    origin_tag = ws.get("origin_tag")
    return origin_tag if isinstance(origin_tag, str) and origin_tag else None


def _workspace_roots_from_env() -> list[str]:
    """Read extra discovery roots from the environment."""
    roots_value = os.environ.get("NEBI_WORKSPACE_DISCOVERY_PATHS", "").strip()
    if not roots_value:
        return []

    roots: list[str] = []
    for root in roots_value.split(os.pathsep):
        root = root.strip()
        if root:
            roots.append(root)
    return roots


def _discover_workspaces_from_roots(roots: list[str]) -> list[NebiWorkspace]:
    """Discover workspace directories from configured roots."""
    workspaces: list[NebiWorkspace] = []
    for root in roots:
        if not os.path.isdir(root):
            logger.debug("Workspace discovery root does not exist: %s", root)
            continue

        try:
            entries = os.listdir(root)
        except OSError:
            logger.exception("Failed to list workspace discovery root: %s", root)
            continue

        for entry in entries:
            path = os.path.join(root, entry)
            if not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, "pixi.toml")) and not os.path.exists(
                os.path.join(path, "pyproject.toml")
            ):
                continue
            workspaces.append(NebiWorkspace(name=entry, path=path))

    return workspaces


def discover_workspaces(discovery_roots: list[str] | None = None) -> list[NebiWorkspace]:
    """Discover locally available nebi workspaces.

    Calls ``nebi workspace list --json`` and filters out missing workspaces.
    Optionally augments discovery from configured local workspace roots.

    Returns:
        List of discovered workspaces. Empty list if nebi is not
        installed or an error occurs.
    """
    try:
        result = subprocess.run(
            ["nebi", "workspace", "list", "--json"],
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

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse nebi workspace list JSON")
        return []

    workspaces_by_name: dict[str, NebiWorkspace] = {}
    for ws in data if isinstance(data, list) else []:
        if not isinstance(ws, dict):
            continue
        if ws.get("missing", False):
            logger.debug("Skipping missing workspace: %s", ws.get("name"))
            continue

        name = ws.get("name")
        path = ws.get("path")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(path, str) or not path:
            continue

        workspaces_by_name[name] = NebiWorkspace(
            name=name,
            path=path,
            local_version=_parse_local_version(ws),
            source="local",
        )

    roots = discovery_roots or _workspace_roots_from_env()
    for workspace in _discover_workspaces_from_roots(roots):
        existing = workspaces_by_name.get(workspace.name)
        if existing and existing.path:
            continue
        workspaces_by_name[workspace.name] = workspace

    return list(workspaces_by_name.values())


def _nebi_auth_token_from_env() -> str | None:
    """Get the Nebi auth token from the canonical singleuser pod env var.
    """
    value = os.environ.get("NEBI_AUTH_TOKEN")
    return value if value else None


def _request_json(url: str, token: str, *, timeout: int) -> Any:
    """Perform an authenticated JSON request."""
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _latest_tag(tags: list[dict[str, Any]]) -> str | None:
    """Return the latest tag using version_number then updated_at ordering."""
    best: tuple[int, str, str] | None = None
    for tag in tags:
        tag_name = tag.get("tag")
        version_number = tag.get("version_number")
        updated_at_raw = tag.get("updated_at")
        if not isinstance(tag_name, str) or not tag_name:
            continue
        if not isinstance(version_number, int):
            continue

        updated_at = updated_at_raw if isinstance(updated_at_raw, str) else ""
        candidate = (version_number, updated_at, tag_name)
        if best is None or candidate > best:
            best = candidate

    return best[2] if best else None


def _discover_remote_tag(remote_url: str, token: str, workspace_id: str) -> str | None:
    """Get latest tag for a workspace from /api/v1/workspaces/{id}/tags."""
    if not workspace_id:
        return None

    tags_url = f"{remote_url.rstrip('/')}/api/v1/workspaces/{quote(workspace_id)}/tags"
    try:
        payload = _request_json(tags_url, token, timeout=10)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    if not isinstance(payload, list):
        return None

    tag_items = [item for item in payload if isinstance(item, dict)]
    return _latest_tag(tag_items)


def _parse_pixi_toml_environments(content: str) -> list[str]:
    """Parse environment names from pixi.toml content."""
    if not content:
        return []

    names: list[str] = []
    in_environments_section = False

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_environments_section = line[1:-1].strip() == "environments"
            continue
        if not in_environments_section or "=" not in line:
            continue

        key = line.split("=", 1)[0].strip().strip("\"'")
        if key and key not in names:
            names.append(key)

    return names


def _discover_remote_environments(remote_url: str, token: str, workspace_id: str) -> list[str]:
    """Discover declared environments for a remote workspace from pixi.toml."""
    pixi_toml_url = f"{remote_url.rstrip('/')}/api/v1/workspaces/{quote(workspace_id)}/pixi-toml"
    try:
        payload = _request_json(pixi_toml_url, token, timeout=10)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict):
        return []
    content = payload.get("content")
    if not isinstance(content, str):
        return []

    return _parse_pixi_toml_environments(content)


def _discover_remote_workspaces_from_api(remote_url: str, token: str) -> list[NebiWorkspace]:
    """Discover remote workspaces from the Nebi server API."""
    workspaces_url = f"{remote_url.rstrip('/')}/api/v1/workspaces"
    try:
        data = _request_json(workspaces_url, token, timeout=15)
    except HTTPError as exc:
        logger.warning("Nebi remote workspace discovery failed with HTTP %s", exc.code)
        return []
    except URLError:
        logger.warning("Nebi remote workspace discovery failed to reach %s", remote_url)
        return []
    except TimeoutError:
        logger.warning("Nebi remote workspace discovery timed out for %s", remote_url)
        return []
    except json.JSONDecodeError:
        logger.warning("Failed to parse Nebi remote workspace JSON response")
        return []

    if not isinstance(data, list):
        logger.warning("Unexpected Nebi remote workspace response format (expected array)")
        return []

    workspaces: list[NebiWorkspace] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        workspace_id = raw.get("id")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(workspace_id, str) or not workspace_id:
            logger.debug("Skipping remote workspace %r without id", name)
            continue

        workspaces.append(
            NebiWorkspace(
                name=name,
                path="",
                remote_version=_discover_remote_tag(remote_url, token, workspace_id),
                environments=_discover_remote_environments(remote_url, token, workspace_id),
                source="remote",
            )
        )
    return workspaces


def discover_remote_workspaces() -> list[NebiWorkspace]:
    """Discover workspaces available on the configured remote Nebi server."""
    remote_url = os.environ.get("NEBI_REMOTE_URL", "").strip()
    token = _nebi_auth_token_from_env()

    if not remote_url:
        logger.debug("NEBI_REMOTE_URL is not configured; skipping remote workspace discovery")
        return []
    if not token:
        logger.warning(
            "NEBI_AUTH_TOKEN is not configured; skipping remote workspace discovery"
        )
        return []

    return _discover_remote_workspaces_from_api(remote_url, token)


def _find_manifest(workspace_path: str) -> str:
    """Find the pixi manifest file in a workspace directory.

    Checks for ``pixi.toml`` first, then ``pyproject.toml``.
    """
    for name in ("pixi.toml", "pyproject.toml"):
        path = os.path.join(workspace_path, name)
        if os.path.exists(path):
            return path
    return os.path.join(workspace_path, "pixi.toml")


def discover_environments(workspace_path: str) -> list[str]:
    """Discover pixi environments in a workspace.

    Calls ``pixi info --json`` and extracts environment names from
    the ``environments_info`` array.

    Args:
        workspace_path: Absolute path to the workspace directory.

    Returns:
        List of environment names. Falls back to ``["default"]``
        if pixi is not installed or the command fails.
    """
    manifest = _find_manifest(workspace_path)

    try:
        result = subprocess.run(
            ["pixi", "info", "--json", "--manifest-path", manifest],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.warning("pixi CLI not found on PATH — falling back to default environment")
        return ["default"]
    except subprocess.TimeoutExpired:
        logger.warning("pixi info timed out for %s", workspace_path)
        return ["default"]

    if result.returncode != 0:
        logger.debug(
            "pixi info failed for %s: %s",
            workspace_path,
            result.stderr.strip(),
        )
        return ["default"]

    try:
        data = json.loads(result.stdout)
        envs = [env["name"] for env in data.get("environments_info", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.debug("Failed to parse pixi info JSON for %s", workspace_path)
        return ["default"]

    return envs if envs else ["default"]


def _extract_package_names(data: Any) -> set[str]:
    """Extract package names from a pixi list JSON payload."""
    names: set[str] = set()
    if isinstance(data, dict):
        name = data.get("name")
        if isinstance(name, str) and name:
            names.add(name)

        for key in ("packages", "dependencies", "installed"):
            value = data.get(key)
            if isinstance(value, list):
                names.update(_extract_package_names(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                names.update(_extract_package_names(item))
    return names


def probe_environment(
    workspace_path: str,
    env: str,
    required_dependencies: tuple[str, ...] = ("ipykernel",),
) -> EnvironmentProbe:
    """Check whether an environment is installed and has required dependencies."""
    if not workspace_path or not os.path.isdir(workspace_path):
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_WORKSPACE_MISSING,
        )

    manifest = _find_manifest(workspace_path)
    if not os.path.exists(manifest):
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_MANIFEST_MISSING,
        )

    cmd = ["pixi", "list", "--json", "--no-install", "--manifest-path", manifest]
    if env != "default":
        cmd.extend(["-e", env])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_PIXI_MISSING,
        )
    except subprocess.TimeoutExpired:
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_PIXI_TIMEOUT,
        )

    if result.returncode != 0:
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_PIXI_LIST_FAILED,
        )

    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return EnvironmentProbe(
            installed=False,
            missing_dependencies=[],
            reason=PROBE_REASON_PIXI_JSON_PARSE_FAILED,
        )

    installed_packages = {name.lower() for name in _extract_package_names(payload)}
    missing = [dep for dep in required_dependencies if dep.lower() not in installed_packages]
    return EnvironmentProbe(installed=True, missing_dependencies=missing)
