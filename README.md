# nb-nebi-kernels

A custom Jupyter `KernelSpecManager` that automatically discovers local and remote [Nebi](https://github.com/nebari-dev/nebi)-tracked [Pixi](https://pixi.sh/) workspaces and exposes each environment as a Jupyter kernel.

## How it works

1. Discovers locally-tracked workspaces via `nebi workspace list --json`.
2. Optionally discovers additional local workspaces from configured roots.
3. Optionally discovers remote workspaces from the Nebi API using `NEBI_REMOTE_URL` and `NEBI_AUTH_TOKEN`.
4. Enumerates environments per workspace (`pixi info --json` for local workspaces, remote `pixi.toml` parsing for remote-only workspaces).
5. Classifies each kernel into a structured state (`ready`, `outdated`, `remote-not-pulled`, `local-not-installed`, `local-missing-deps`).
6. Launches ready/outdated kernels via `pixi run --frozen` in the workspace directory with environment isolation.

## Installation

```bash
pip install nb-nebi-kernels
```

That's it — the kernel spec manager is automatically configured when installed into your JupyterLab environment.

### Prerequisites

- [Nebi CLI](https://github.com/nebari-dev/nebi) on your PATH
- [Pixi](https://pixi.sh/) on your PATH
- At least one tracked nebi workspace (`nebi init` in a pixi project)
- For remote discovery: `NEBI_REMOTE_URL` and `NEBI_AUTH_TOKEN`

## Usage

Once installed, any nebi-tracked pixi workspace appears as a kernel in JupyterLab or Notebook:

- A workspace `data-science` with environments `default` and `gpu` shows as two kernels: **data-science (default)** and **data-science (gpu)**
- A workspace `web-app` with only the default environment shows as just **web-app**
- A remote-only workspace can still appear as a kernel (state: `remote-not-pulled`) before it is pulled locally

If discovery returns no workspaces, Jupyter falls back to its default kernels. If workspaces are discovered but local tools/dependencies are missing, kernels are surfaced as non-ready instead of crashing.

## Configuration

Optional behavior can be configured through environment variables and traitlets.

### Environment variables

- `NEBI_REMOTE_URL`: Base Nebi server URL used for remote workspace discovery.
- `NEBI_AUTH_TOKEN`: Bearer token used for Nebi API requests.
- `NEBI_WORKSPACE_DISCOVERY_PATHS`: Extra local workspace roots, separated by the OS path separator (`:` on Linux/macOS, `;` on Windows).

### Traitlets

```python
# Example in jupyter_server_config.py
c.NebiKernelSpecManager.workspace_discovery_roots = ["/mnt/shared/nebi-workspaces"]
c.NebiKernelSpecManager.required_launch_dependencies = ["ipykernel"]
```

- `workspace_discovery_roots` adds local discovery roots (in addition to `nebi workspace list` results).
- `required_launch_dependencies` controls package checks required for a kernel to be considered launchable. Default: `["ipykernel"]`.

## Kernel states and metadata

Each generated kernelspec includes state metadata for UI consumers.

| `nebi_state` | Meaning | Launch |
| --- | --- | --- |
| `ready` | Local workspace/env is installed and dependencies are satisfied. | Allowed |
| `outdated` | Local and remote versions differ. | Allowed |
| `remote-not-pulled` | Workspace exists remotely but has no local path yet. | Blocked |
| `local-not-installed` | Local environment is not installed/materialized. | Blocked |
| `local-missing-deps` | Local environment is missing required dependencies. | Blocked |

Important metadata fields:

- `nebi_state`, `nebi_not_ready_reason`, `nebi_missing_dependencies`
- `nebi_local_version`, `nebi_remote_version`, `nebi_outdated`
- `nebi_discovery_hash`, `nebi_discovered_at`
- `nebi` (nested metadata object with the same core state fields)

Stable `nebi_not_ready_reason` values include:

- `workspace-not-pulled`
- `missing-dependencies`
- `local-version-behind-remote`
- `environment-not-installed`
- `workspace-missing`
- `manifest-missing`
- `pixi-missing`
- `pixi-timeout`
- `pixi-list-failed`
- `pixi-json-parse-failed`

## Development

```bash
# Install dev dependencies
pixi install -e dev

# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Run linting
pixi run lint

# Format code
pixi run format

# Run type checking
pixi run typecheck
```

## Architecture

```
src/nb_nebi_kernels/
├── __init__.py      # Exports NebiKernelSpecManager
├── discovery.py     # Local/remote workspace discovery + environment probing
├── launcher.py      # Kernel launcher with state-aware launch blocking
└── manager.py       # KernelSpecManager subclass (core logic)
```

- **discovery.py** — Parses local `nebi workspace list --json`, optionally discovers remote workspaces through the Nebi API, resolves environment names, and probes local env install/dependency health.
- **launcher.py** — Clears `PIXI_*` environment variables, blocks non-launchable states with actionable stderr messages, then execs `pixi run` in the workspace directory.
- **manager.py** — Subclasses `KernelSpecManager`, merges local and remote workspace views, classifies per-kernel state, and emits structured kernelspec metadata.

## License

MIT
