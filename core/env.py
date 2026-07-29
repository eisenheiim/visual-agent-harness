"""
Load `.env` from the project root (if present).

Call :func:`load_env` once at process start (UI / examples).
Existing OS environment variables always win over `.env`.
"""

from __future__ import annotations

from pathlib import Path


def load_env(dotenv_path: str | Path | None = None) -> bool:
    """Load key=value pairs from a ``.env`` file into ``os.environ``.

    Returns True if a file was found and loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    if dotenv_path is None:
        # tools/ → parent is repo root when imported as package; also try CWD
        root = Path(__file__).resolve().parent.parent
        candidates = [root / ".env", Path.cwd() / ".env"]
    else:
        candidates = [Path(dotenv_path)]

    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return True
    return False
