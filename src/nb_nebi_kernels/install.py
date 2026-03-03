"""Installation and configuration management for nb-nebi-kernels."""

import argparse
import logging
import sys
from os.path import abspath, exists, join
from pathlib import Path

from jupyter_core.paths import jupyter_config_path

try:
    from jupyter_server.config_manager import BaseJSONConfigManager
except ImportError as e:
    raise ImportError("jupyter_server must be installed") from e

logger = logging.getLogger(__name__)

SA = "ServerApp"
KSMC = "kernel_spec_manager_class"
OUR_MANAGER = "nb_nebi_kernels.NebiKernelSpecManager"
CONFIG_FILE = "jupyter_config"
ENDIS = ["disabled", "enabled"]


def shorten(path: str, prefix: bool = True) -> str:
    """Shorten a path for display purposes."""
    import os

    if prefix and path.startswith(sys.prefix + os.sep):
        var = "%CONDA_PREFIX%" if sys.platform.startswith("win") else "$CONDA_PREFIX"
        return var + path[len(sys.prefix):]
    home = os.path.expanduser("~")
    if path.startswith(home + os.sep):
        var = "%USERPROFILE%" if sys.platform.startswith("win") else "~"
        return var + path[len(home):]
    return path


def find_config_path(prefix: str | None = None, path: str | None = None) -> str:
    """Find the appropriate Jupyter configuration path."""
    import os

    all_paths = [abspath(p) for p in jupyter_config_path()]
    default_path = join(sys.prefix, "etc", "jupyter")

    if path or prefix:
        if prefix:
            path = join(prefix, "etc", "jupyter")
        return abspath(path)

    prefix_s = sys.prefix + os.sep
    for p in reversed(all_paths):
        if p.startswith(prefix_s):
            return p

    return default_path


def get_current_status(config_path: str | None = None) -> dict[str, bool | str | None]:
    """Get the current installation status."""
    all_paths = [abspath(p) for p in jupyter_config_path()]
    paths_to_check = [config_path] if config_path else all_paths

    for path in reversed(paths_to_check):
        config_file = join(path, CONFIG_FILE + ".json")
        if exists(config_file):
            cfg = BaseJSONConfigManager(config_dir=path).get(CONFIG_FILE)
            manager = cfg.get(SA, {}).get(KSMC)
            if manager:
                return {
                    "enabled": manager == OUR_MANAGER,
                    "path": path,
                    "current_manager": manager,
                }

    return {"enabled": False, "path": None, "current_manager": None}


def install(
    enable: bool = False,
    disable: bool = False,
    status: bool = False,
    prefix: str | None = None,
    path: str | None = None,
    verbose: bool = False,
) -> int:
    """Install or manage the nb-nebi-kernels configuration."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = find_config_path(prefix, path)
    all_paths = [abspath(p) for p in jupyter_config_path()]

    if config_path not in all_paths:
        logger.warning(
            "%s is not in Jupyter's config search path. "
            "The configuration may not be found by Jupyter.",
            shorten(config_path),
        )

    if status:
        current = get_current_status()
        if current["enabled"]:
            print("Status: enabled")
            print(f"Config: {shorten(current['path'])}")
        elif current["current_manager"]:
            print("Status: disabled (another manager is active)")
            print(f"Current manager: {current['current_manager']}")
            print(f"Config: {shorten(current['path'])}")
        else:
            print("Status: disabled (no custom kernel spec manager configured)")
        return 0

    Path(config_path).mkdir(parents=True, exist_ok=True)
    cfg = BaseJSONConfigManager(config_dir=config_path).get(CONFIG_FILE)

    if enable:
        cfg.setdefault(SA, {})[KSMC] = OUR_MANAGER
        logger.info("Setting %s.%s = %s", SA, KSMC, OUR_MANAGER)
    else:
        if cfg.get(SA, {}).get(KSMC) == OUR_MANAGER:
            cfg[SA].pop(KSMC, None)
            if not cfg.get(SA):
                cfg.pop(SA, None)
            logger.info("Removed %s.%s setting", SA, KSMC)
        elif cfg.get(SA, {}).get(KSMC):
            logger.warning(
                "Another kernel spec manager is configured: %s\n"
                "Not modifying configuration.",
                cfg[SA][KSMC],
            )
            return 1
        else:
            print("nb-nebi-kernels is not currently enabled.")
            return 0

    BaseJSONConfigManager(config_dir=config_path).set(CONFIG_FILE, cfg)

    config_file = join(config_path, CONFIG_FILE + ".json")
    logger.info("Wrote configuration to: %s", shorten(config_file))

    new_status = get_current_status(config_path)
    if new_status["enabled"] != enable:
        logger.error("Configuration verification failed!")
        return 1

    print(f"Status: {ENDIS[enable]}")
    return 0


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Manage nb-nebi-kernels Jupyter configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --enable          Enable nb-nebi-kernels
  %(prog)s --disable         Disable nb-nebi-kernels
  %(prog)s --status          Check current status
  %(prog)s --enable --prefix /path/to/env
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--enable", action="store_true", help="Enable nb-nebi-kernels")
    group.add_argument("-d", "--disable", action="store_true", help="Disable nb-nebi-kernels")
    group.add_argument("-s", "--status", action="store_true", help="Print current status")

    location = parser.add_mutually_exclusive_group()
    location.add_argument("-p", "--prefix", help="Python environment prefix")
    location.add_argument("--path", help="Explicit Jupyter config directory path")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    return install(
        enable=args.enable,
        disable=args.disable,
        status=args.status,
        prefix=args.prefix,
        path=args.path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
