"""Kernel launcher script that isolates pixi from parent manifests."""

import os
import sys

PIXI_ENV_VARS_TO_CLEAR = [
    "PIXI_ENVIRONMENT_NAME",
    "PIXI_ENVIRONMENT_PLATFORMS",
    "PIXI_IN_SHELL",
    "PIXI_PROJECT_MANIFEST",
    "PIXI_PROJECT_NAME",
    "PIXI_PROJECT_ROOT",
    "PIXI_PROJECT_VERSION",
    "PIXI_PROMPT",
]


def _print_state_error(state: str, kernel_name: str) -> None:
    """Print a user-facing launch blocker message for non-ready kernels."""
    if state == "remote-not-pulled":
        print(
            f"Nebi kernel '{kernel_name}' is remote-only and has not been pulled locally yet.",
            file=sys.stderr,
        )
        print("Run `nebi pull` for this workspace, then refresh kernels.", file=sys.stderr)
    elif state == "local-not-installed":
        print(
            f"Nebi kernel '{kernel_name}' has not been installed locally yet.",
            file=sys.stderr,
        )
        print(
            "Run `pixi install` (or equivalent) in the workspace, then refresh kernels.",
            file=sys.stderr,
        )
    elif state == "local-missing-deps":
        print(
            (
                f"Nebi kernel '{kernel_name}' is missing required dependencies "
                "(for example ipykernel)."
            ),
            file=sys.stderr,
        )
        print(
            "Install missing dependencies in the pixi environment, then refresh kernels.",
            file=sys.stderr,
        )


def main() -> None:
    """Launch ipykernel via pixi in an isolated workspace directory.

    Usage:
        python -m nb_nebi_kernels.launcher <workspace_dir> <environment> <connection_file>

    Args:
        workspace_dir: Directory containing pixi.toml
        environment: Pixi environment name (e.g. "default", "gpu")
        connection_file: Path to Jupyter connection file
    """
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <workspace_dir> <environment> <connection_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    workspace_dir = sys.argv[1]
    environment = sys.argv[2]
    connection_file = sys.argv[3]
    kernel_state = os.environ.get("NB_NEBI_KERNEL_STATE", "").strip()
    kernel_name = os.environ.get("NB_NEBI_KERNEL_NAME", "<unknown>")

    if kernel_state and kernel_state not in {"ready", "outdated"}:
        _print_state_error(kernel_state, kernel_name)
        sys.exit(1)

    for var in PIXI_ENV_VARS_TO_CLEAR:
        os.environ.pop(var, None)

    if not os.path.isdir(workspace_dir):
        print(
            f"Workspace directory does not exist: {workspace_dir!r}. "
            "Pull the Nebi workspace and refresh kernels.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(workspace_dir)

    cmd = [
        "pixi",
        "run",
        "--frozen",
        "--manifest-path",
        os.path.join(workspace_dir, "pixi.toml"),
    ]

    # Only pass -e flag for non-default environments
    if environment != "default":
        cmd.extend(["-e", environment])

    cmd.extend([
        "python",
        "-m",
        "ipykernel_launcher",
        "-f",
        connection_file,
    ])

    os.execvp("pixi", cmd)


if __name__ == "__main__":
    main()
