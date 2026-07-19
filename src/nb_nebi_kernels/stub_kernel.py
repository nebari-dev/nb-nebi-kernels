"""Stub kernel that surfaces not-ready Nebi kernel states to the user.

When ``NebiKernelSpecManager`` discovers a (workspace, env) pair that cannot
launch yet, it emits a kernelspec pointing at this module instead of the regular
launcher. The stub subclasses
``ipykernel.kernelbase.Kernel`` so it inherits the full Jupyter messaging
protocol (ZMQ binding, heartbeat, signing, kernel_info handshake). JupyterLab
treats it as a live kernel rather than misclassifying a fast-exiting launcher
as "kernel died, retry". Any ``execute_request`` returns a structured error
whose traceback is the recovery instruction — visible in cell output.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

ENAME = "MissingKernelError"


def build_message(
    workspace: str,
    env: str,
    reason: str = "kernel-not-installed",
    missing_dependencies: list[str] | None = None,
) -> list[str]:
    """Return the install message as a list of traceback lines."""
    missing_dependencies = missing_dependencies or []
    common = [
        "",
        f"  Workspace:  {workspace}",
        f"  Pixi env:   {env}",
        f"  Reason:     {reason}",
        "",
    ]

    if reason == "missing-dependencies":
        lines = [
            f"This pixi environment ('{env}') is missing required launch dependencies.",
            *common,
        ]
        if missing_dependencies:
            lines.extend([
                "Missing dependencies:",
                "",
                *[f"  - {dependency}" for dependency in missing_dependencies],
                "",
            ])
        lines.extend([
            "Install the missing dependencies in the pixi environment.",
            "",
            "Then re-open this notebook and select the kernel again.",
        ])
        return lines

    if reason == "workspace-not-pulled":
        return [
            f"This Nebi workspace ('{workspace}') has not been pulled locally yet.",
            *common,
            f"Pull the workspace with `nebi pull {workspace}`.",
            "",
            "Then refresh kernels and select this kernel again.",
        ]

    if reason != "kernel-not-installed":
        return [
            f"This Nebi kernel is not ready for pixi environment ('{env}').",
            *common,
            "Resolve the pixi environment issue, refresh kernels, then select the kernel again.",
        ]

    return [
        f"This pixi environment ('{env}') does not contain a Jupyter kernel.",
        *common,
        (
            "From the workspace directory, install a Jupyter kernel supported "
            "by this environment, for example:"
        ),
        "",
        f"  cd /path/to/{workspace}",
        f"  pixi add --feature {env} ipykernel",
        "",
        "Other kernels such as IRkernel, IJulia, and xeus-based kernels are also supported.",
        "",
        "Then re-open this notebook and select the kernel again.",
    ]


def _error_value(reason: str, env: str) -> str:
    """Return the short error value for a not-ready reason."""
    if reason == "kernel-not-installed":
        return f"No Jupyter kernel installed in pixi env '{env}'"
    if reason == "missing-dependencies":
        return f"Missing required launch dependencies in pixi env '{env}'"
    if reason == "workspace-not-pulled":
        return "Nebi workspace has not been pulled locally"
    return f"Nebi kernel is not ready in pixi env '{env}'"


class StubKernel(Kernel):
    """A Jupyter kernel that exists only to surface a not-ready error.

    Override class attributes ``nebi_workspace`` and ``nebi_env`` (typically via
    a subclass produced at launch time) so the error message names the
    specific workspace and env.
    """

    implementation = "nebi-stub"
    implementation_version = "0.1.0"
    language = "no-op"
    language_version = "0"
    language_info = {
        "name": "no-op",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }

    nebi_workspace: str = "<unknown>"
    nebi_env: str = "<unknown>"
    nebi_reason: str = "kernel-not-installed"
    nebi_missing_dependencies: list[str] = []
    banner = "Nebi stub kernel — this pixi environment is not ready."

    async def do_execute(  # type: ignore[override]
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict[str, Any] | None = None,
        allow_stdin: bool = False,
        *,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        evalue = _error_value(self.nebi_reason, self.nebi_env)
        traceback = build_message(
            self.nebi_workspace,
            self.nebi_env,
            self.nebi_reason,
            self.nebi_missing_dependencies,
        )

        if not silent:
            self.send_response(
                self.iopub_socket,
                "error",
                {"ename": ENAME, "evalue": evalue, "traceback": traceback},
            )

        return {
            "status": "error",
            "ename": ENAME,
            "evalue": evalue,
            "traceback": traceback,
            "execution_count": self.execution_count,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nb_nebi_kernels.stub_kernel",
        description="Nebi stub kernel for envs that are not ready to launch.",
    )
    parser.add_argument("--workspace", required=True, help="Nebi workspace name")
    parser.add_argument("--env", required=True, help="Pixi environment name")
    parser.add_argument(
        "--reason",
        default="kernel-not-installed",
        help="Stable not-ready reason for this stub kernel",
    )
    parser.add_argument(
        "--missing-dependency",
        action="append",
        default=[],
        dest="missing_dependencies",
        help="Missing launch dependency; may be provided multiple times",
    )
    parser.add_argument(
        "-f",
        "--connection-file",
        required=True,
        help="Path to Jupyter connection file (provided by Jupyter)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Subclass at runtime so the workspace/env are baked into the class
    # used by IPKernelApp (which instantiates the kernel itself).
    kernel_cls = type(
        "ConfiguredStubKernel",
        (StubKernel,),
        {
            "nebi_workspace": args.workspace,
            "nebi_env": args.env,
            "nebi_reason": args.reason,
            "nebi_missing_dependencies": args.missing_dependencies,
            "banner": (
                f"Nebi stub kernel — pixi env '{args.env}' in workspace "
                f"'{args.workspace}' is not ready ({args.reason}).\n"
                "Any cell you run will print the recovery instructions."
            ),
        },
    )

    # ipykernel's CLI parses sys.argv for `-f <connection_file>`. We've already
    # consumed our own flags, so reset argv to what ipykernel expects.
    sys.argv = [sys.argv[0], "-f", args.connection_file]
    IPKernelApp.launch_instance(kernel_class=kernel_cls)


if __name__ == "__main__":
    main()
