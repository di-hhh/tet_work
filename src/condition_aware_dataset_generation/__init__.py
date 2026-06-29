from pathlib import Path
import sys


def _bootstrap_local_deps() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    deps_dir = repo_root / ".deps"
    if deps_dir.exists():
        deps_path = str(deps_dir)
        if deps_path not in sys.path:
            sys.path.insert(0, deps_path)


_bootstrap_local_deps()
