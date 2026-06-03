"""Stub kernel that surfaces 'no kernel installed in env' to the user.

When ``NebiKernelSpecManager`` discovers a (workspace, env) pair whose pixi
env has no Jupyter kernel package installed, it emits a kernelspec pointing
at this module instead of the regular launcher. The stub subclasses
``ipykernel.kernelbase.Kernel`` so it inherits the full Jupyter messaging
protocol (ZMQ binding, heartbeat, signing, kernel_info handshake). JupyterLab
treats it as a live kernel rather than misclassifying a fast-exiting launcher
as "kernel died, retry". Any ``execute_request`` returns a structured error
whose traceback is the install instructions — visible in cell output.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

ENAME = "MissingKernelError"


def build_message(workspace: str, env: str) -> list[str]:
    """Return the install message as a list of traceback lines."""
    return [
        f"This pixi environment ('{env}') does not contain a Jupyter kernel.",
        "",
        f"  Workspace:  {workspace}",
        f"  Pixi env:   {env}",
        "",
        "Install one with, e.g.:",
        "",
        f"  pixi add --manifest-path /path/to/{workspace}/pixi.toml \\",
        f"      --feature {env} ipykernel",
        "",
        "Then re-open this notebook and select the kernel again.",
    ]


class StubKernel(Kernel):
    """A Jupyter kernel that exists only to surface a missing-kernel error.

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

    @property
    def banner(self) -> str:  # type: ignore[override]
        return (
            f"Nebi stub kernel — pixi env '{self.nebi_env}' in workspace "
            f"'{self.nebi_workspace}' has no Jupyter kernel installed.\n"
            f"Any cell you run will print the install instructions."
        )

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
        evalue = f"No Jupyter kernel installed in pixi env '{self.nebi_env}'"
        traceback = build_message(self.nebi_workspace, self.nebi_env)

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
        description="Nebi stub kernel for envs missing a Jupyter kernel package.",
    )
    parser.add_argument("--workspace", required=True, help="Nebi workspace name")
    parser.add_argument("--env", required=True, help="Pixi environment name")
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
        {"nebi_workspace": args.workspace, "nebi_env": args.env},
    )

    # ipykernel's CLI parses sys.argv for `-f <connection_file>`. We've already
    # consumed our own flags, so reset argv to what ipykernel expects.
    sys.argv = [sys.argv[0], "-f", args.connection_file]
    IPKernelApp.launch_instance(kernel_class=kernel_cls)


if __name__ == "__main__":
    main()
