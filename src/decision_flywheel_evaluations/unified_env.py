"""Where the unified-flywheel harness finds Jev-Flywheel, and how it stays offline.

The harness imports ``jev_flywheel`` from a clean local clone of Jev-Flywheel pinned at
``JEV_FLYWHEEL_COMMIT``, never from the working checkout (which carries uncommitted
changes). The clone lives in this repository's gitignored ``var/`` directory:

    git clone --local /path/to/Jev-Flywheel var/jev-flywheel-cd4a4886
    git -C var/jev-flywheel-cd4a4886 checkout cd4a4886

``prepare_clone`` does exactly that, idempotently, and ``verify_clone`` refuses a clone at
any other commit or with local modifications, so every run is reproducible from the commit
recorded in its output.

``install_network_guard`` is the offline mode's hard stop: once installed, any attempt to
open a non-local socket raises before a byte is sent. Offline runs and specs install it, so
an accidental real client (or a library that phones home at import time, as LiteLLM does
for its price table) fails loudly instead of spending.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

JEV_FLYWHEEL_COMMIT = "cd4a4886028d518c4dc20c88608bb05cbc6bf370"
JEV_FLYWHEEL_SHORT = JEV_FLYWHEEL_COMMIT[:8]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLONE = REPO_ROOT / "var" / f"jev-flywheel-{JEV_FLYWHEEL_SHORT}"
CLONE_ENV = "UNIFIED_FLYWHEEL_JEV_CLONE"


class EnvironmentProblem(RuntimeError):
    """The harness cannot run in this interpreter or against this clone."""


@dataclass(frozen=True)
class CloneIdentity:
    path: str
    commit: str
    clean: bool


def clone_path() -> Path:
    return Path(os.environ.get(CLONE_ENV) or DEFAULT_CLONE)


def _git(clone: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(clone), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode().strip()


def prepare_clone(source: Path, clone: Path | None = None) -> CloneIdentity:
    """Clone ``source`` locally and check out the pinned commit in the clone only.

    ``git clone --local`` never touches the source checkout's working tree or index, and
    no worktree is registered on it.
    """
    clone = Path(clone or clone_path())
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--local", "-q", str(source), str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "-q", JEV_FLYWHEEL_COMMIT], check=True)
    return verify_clone(clone)


def verify_clone(clone: Path | None = None) -> CloneIdentity:
    clone = Path(clone or clone_path())
    if not (clone / "jev_flywheel" / "__init__.py").exists():
        raise EnvironmentProblem(
            f"no Jev-Flywheel clone at {clone}; run `scripts/unified_flywheel.py prepare-clone "
            "--source /path/to/Jev-Flywheel` first")
    try:
        commit = _git(clone, "rev-parse", "HEAD")
        dirty = _git(clone, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise EnvironmentProblem(f"{clone} is not a readable git clone") from error
    if commit != JEV_FLYWHEEL_COMMIT:
        raise EnvironmentProblem(f"clone is at {commit[:12]}, expected {JEV_FLYWHEEL_SHORT}")
    if dirty:
        raise EnvironmentProblem("the Jev-Flywheel clone has local modifications; re-clone it")
    return CloneIdentity(str(clone), commit, True)


def put_clone_first(clone: Path | None = None) -> CloneIdentity:
    """Make ``import jev_flywheel`` resolve to the pinned clone, and prove that it did."""
    identity = verify_clone(clone)
    if identity.path not in sys.path:
        sys.path.insert(0, identity.path)
    import jev_flywheel  # noqa: PLC0415 - resolved only after the path is fixed

    resolved = Path(jev_flywheel.__file__).resolve()
    if Path(identity.path).resolve() not in resolved.parents:
        raise EnvironmentProblem(
            f"jev_flywheel resolved to {resolved}, not the pinned clone; another copy was "
            "imported first")
    return identity


PINNED_DECISION_FLYWHEEL = "6137fa185a1a98afa84b5e6d5948d1780df5d56d"
SHIM_DIR = REPO_ROOT / "var" / "unified-flywheel" / "pyshim"


def ensure_decision_flywheel() -> dict:
    """Make ``decision_flywheel`` importable and say which copy it is.

    The interpreter that has scikit-learn and Tactus (Jev-Flywheel's) does not have
    Decision-Flywheel installed, while this repository's own ``.venv`` has the pinned copy.
    Decision-Flywheel is pure standard-library Python, so a one-entry shim directory holding
    a symlink to the pinned package is enough; nothing else from that environment is put on
    the path.
    """
    try:
        import decision_flywheel  # noqa: F401
    except ImportError:
        candidates = sorted(REPO_ROOT.glob(".venv/lib/python3*/site-packages/decision_flywheel"))
        if not candidates:
            raise EnvironmentProblem("decision_flywheel is not importable and no pinned copy was "
                                     "found in this repository's .venv; run `pip install -e .` there")
        SHIM_DIR.mkdir(parents=True, exist_ok=True)
        link = SHIM_DIR / "decision_flywheel"
        if not link.exists():
            link.symlink_to(candidates[-1], target_is_directory=True)
        sys.path.insert(0, str(SHIM_DIR))
        import decision_flywheel  # noqa: F401,F811
    import decision_flywheel as package

    location = Path(package.__file__).resolve().parent
    commit = None
    for info in location.parent.glob("decision_flywheel-*.dist-info/direct_url.json"):
        try:
            import json

            commit = json.loads(info.read_text()).get("vcs_info", {}).get("commit_id")
        except (OSError, ValueError):
            commit = None
    return {"commit": commit, "pinned": commit == PINNED_DECISION_FLYWHEEL,
            "source": "site-packages" if "site-packages" in str(location) else "source-tree"}


def jev_dependencies_available() -> bool:
    """Whether this interpreter can fit heads (scikit-learn) and steer (Tactus)."""
    try:
        import sklearn  # noqa: F401
        import tactus  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


_GUARD_INSTALLED = False
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


class NetworkBlocked(ConnectionRefusedError):
    """Raised for any network attempt while the offline guard is installed."""


def _blocked(*_args, **_kwargs):
    raise NetworkBlocked("network access is blocked in offline mode")


def _guarded_connect(self, address):
    if self.family == getattr(socket, "AF_UNIX", object()):
        return _ORIGINAL_CONNECT(self, address)
    raise NetworkBlocked("network access is blocked in offline mode")


def _guarded_connect_ex(self, address):
    if self.family == getattr(socket, "AF_UNIX", object()):
        return _ORIGINAL_CONNECT_EX(self, address)
    raise NetworkBlocked("network access is blocked in offline mode")


def install_network_guard() -> None:
    """Block every non-local socket in this process. Idempotent."""
    global _GUARD_INSTALLED
    # LiteLLM (pulled in by Tactus) fetches a remote price table at import unless told not to.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    _GUARD_INSTALLED = True


def remove_network_guard() -> None:
    global _GUARD_INSTALLED
    socket.socket.connect = _ORIGINAL_CONNECT  # type: ignore[method-assign]
    socket.socket.connect_ex = _ORIGINAL_CONNECT_EX  # type: ignore[method-assign]
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION  # type: ignore[assignment]
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO  # type: ignore[assignment]
    _GUARD_INSTALLED = False


def network_guard_installed() -> bool:
    return _GUARD_INSTALLED
