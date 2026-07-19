"""Protocol-level tests for ``nb_nebi_kernels.stub_kernel``.

Spawns the stub as a subprocess the same way ``jupyter_server`` does on
``POST /api/kernels``, connects with a ``BlockingKernelClient``, sends an
``execute_request``, and verifies the structured error reaches the client.
This is the proof that JupyterLab will get a real error in cell output
instead of a silent "kernel died" classification.
"""

from __future__ import annotations

import queue
import sys

import pytest
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager

from nb_nebi_kernels.stub_kernel import build_message


@pytest.fixture
def stub_km() -> KernelManager:
    """Spawn the stub kernel; tear it down after the test."""
    km = KernelManager()
    km._kernel_spec = KernelSpec(
        argv=[
            sys.executable,
            "-m",
            "nb_nebi_kernels.stub_kernel",
            "--workspace",
            "demo-ws",
            "--env",
            "broken-env",
            "-f",
            "{connection_file}",
        ],
        display_name="demo-ws (broken-env) [test]",
        language="no-op",
        resource_dir="/tmp",
    )
    km.start_kernel()
    yield km
    km.shutdown_kernel(now=True)


class TestStubKernelProtocol:
    """Verify the stub speaks Jupyter protocol well enough to be visible."""

    def test_kernel_info_handshake_completes(self, stub_km: KernelManager) -> None:
        """Kernel completes handshake — JupyterLab will treat it as alive."""
        client = stub_km.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=20)
        finally:
            client.stop_channels()

    def test_execute_surfaces_install_message(self, stub_km: KernelManager) -> None:
        """Any execute_request returns a structured error with install instructions."""
        client = stub_km.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=20)
            client.execute("anything")

            traceback_lines: list[str] = []
            saw_error_on_iopub = False
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=10)
                except queue.Empty:
                    break
                if msg["msg_type"] == "error":
                    saw_error_on_iopub = True
                    traceback_lines = msg["content"].get("traceback", [])
                if (
                    msg["msg_type"] == "status"
                    and msg["content"].get("execution_state") == "idle"
                    and saw_error_on_iopub
                ):
                    break

            reply = client.get_shell_msg(timeout=10)
        finally:
            client.stop_channels()

        assert saw_error_on_iopub, "Stub kernel did not emit iopub error"
        assert reply["content"]["status"] == "error"
        assert reply["content"]["ename"] == "MissingKernelError"
        # Workspace and env should be named in the traceback so the message is
        # actionable, and the pixi add command should be present.
        joined = "\n".join(traceback_lines)
        assert "demo-ws" in joined
        assert "broken-env" in joined
        assert "pixi add" in joined
        assert "ipykernel" in joined


class TestStubKernelMessages:
    """Direct tests for reason-specific recovery messages."""

    def test_missing_dependencies_message_names_dependencies(self) -> None:
        """Configured missing deps do not masquerade as missing Jupyter kernels."""
        joined = "\n".join(
            build_message(
                "demo-ws",
                "broken-env",
                "missing-dependencies",
                ["r-irkernel"],
            )
        )

        assert "missing required launch dependencies" in joined
        assert "r-irkernel" in joined
        assert "does not contain a Jupyter kernel" not in joined
        assert "pixi add" not in joined

    def test_generic_not_ready_message_names_reason(self) -> None:
        """Non-kernelspec blockers get a generic reason-aware recovery message."""
        joined = "\n".join(
            build_message("demo-ws", "broken-env", "environment-not-installed")
        )

        assert "not ready" in joined
        assert "environment-not-installed" in joined
        assert "does not contain a Jupyter kernel" not in joined

    def test_workspace_not_pulled_message_suggests_pull(self) -> None:
        """Remote-only workspaces tell users to pull instead of install a kernel."""
        joined = "\n".join(
            build_message("demo-ws", "default", "workspace-not-pulled")
        )

        assert "has not been pulled locally" in joined
        assert "nebi pull demo-ws" in joined
        assert "pixi add" not in joined
