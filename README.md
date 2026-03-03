# nb-nebi-kernels

A custom Jupyter `KernelSpecManager` that automatically discovers [Nebi](https://github.com/nebari-dev/nebi)-tracked [Pixi](https://pixi.sh/) workspaces and exposes each environment as a launchable Jupyter kernel.

## How it works

1. Discovers locally-tracked workspaces via `nebi workspace list`
2. Enumerates pixi environments per workspace via `pixi workspace environment list`
3. Each (workspace, environment) pair appears as a selectable kernel in Jupyter
4. Kernels launch via `pixi run` in the workspace directory with full environment isolation

## Installation

```bash
pip install nb-nebi-kernels
```

### Prerequisites

- [Nebi CLI](https://github.com/nebari-dev/nebi) on your PATH
- [Pixi](https://pixi.sh/) on your PATH
- At least one tracked nebi workspace (`nebi init` in a pixi project)

### Enable for Jupyter

```bash
# Enable nb-nebi-kernels
nb-nebi-kernels --enable

# Check status
nb-nebi-kernels --status

# Disable (restore default Jupyter behavior)
nb-nebi-kernels --disable
```

## Usage

Once enabled, any nebi-tracked pixi workspace appears as a kernel in JupyterLab or Notebook:

- A workspace `data-science` with environments `default` and `gpu` shows as two kernels: **data-science (default)** and **data-science (gpu)**
- A workspace `web-app` with only the default environment shows as just **web-app**

If nebi or pixi are not installed, or no workspaces are tracked, Jupyter falls back to its default kernels — it never crashes.

## Configuration

### CLI Options

```bash
nb-nebi-kernels --help

# Enable in a specific environment
nb-nebi-kernels --enable --prefix /path/to/env

# Enable with explicit config path
nb-nebi-kernels --enable --path /path/to/jupyter/config

# Verbose output
nb-nebi-kernels --status --verbose
```

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
├── discovery.py     # Subprocess calls to nebi + pixi CLIs
├── launcher.py      # Kernel launcher with pixi environment isolation
├── manager.py       # KernelSpecManager subclass (core logic)
└── install.py       # CLI: nb-nebi-kernels --enable/--disable/--status
```

- **discovery.py** — Parses `nebi workspace list` output, calls `pixi workspace environment list` per workspace. Filters out missing workspaces. Returns structured data.
- **launcher.py** — Clears PIXI_* environment variables to prevent inheriting parent context, then exec's `pixi run` in the workspace directory.
- **manager.py** — Subclasses `KernelSpecManager`, implements `find_kernel_specs()` and `get_kernel_spec()`. Merges parent kernels with nebi-discovered ones.
- **install.py** — CLI that writes `ServerApp.kernel_spec_manager_class` to `jupyter_config.json`.

## License

MIT
