import os
from pathlib import Path
from dotenv import load_dotenv
import logging

_logger = logging.getLogger(__name__)
_DOTENV_LOAD_CACHE: dict[tuple[str | None, str, str | None], Path | None] = {}


def find_dotenv_path() -> Path | None:
    """Find the dotenv file Tooltoad should load.

    Returns
    -------
    Path or None
        The first existing dotenv file from ``TOOLTOAD_DOTENV_PATH``, the
        current working directory, or the user's home directory.
    """
    env_var_path_str = os.getenv("TOOLTOAD_DOTENV_PATH")
    if env_var_path_str:
        path_from_env = Path(os.path.expandvars(env_var_path_str)).expanduser()
        if path_from_env.is_file():
            return path_from_env
        _logger.warning(
            "Path specified by TOOLTOAD_DOTENV_PATH does not exist or is not "
            "a file: %s",
            path_from_env,
        )

    cwd_path = Path.cwd() / ".env"
    if cwd_path.is_file():
        return cwd_path

    try:
        home_path = Path.home() / ".env"
    except Exception:
        _logger.warning("Could not determine user home directory.")
        return None
    return home_path if home_path.is_file() else None


def _dotenv_cache_key() -> tuple[str | None, str, str | None]:
    """Return the cache key for dotenv discovery in the current process."""
    try:
        home = str(Path.home())
    except Exception:
        home = None
    return (os.getenv("TOOLTOAD_DOTENV_PATH"), str(Path.cwd()), home)


def find_and_load_dotenv(*, force: bool = False) -> Path | None:
    """Find and load the first Tooltoad dotenv file.

    Dotenv files are searched in this priority order:

    1. path specified by ``TOOLTOAD_DOTENV_PATH``;
    2. ``.env`` file in the current working directory;
    3. ``.env`` file in the user's home directory.

    Loads the first file found. Repeated calls with the same search context are
    cached so row-wise calculator workflows do not emit repeated dotenv
    diagnostics.

    Parameters
    ----------
    force : bool, optional
        If ``True``, repeat dotenv discovery and loading even when the current
        search context was already checked.

    Returns
    -------
    pathlib.Path or None
        Loaded dotenv path, or ``None`` when no dotenv file was found.
    """
    cache_key = _dotenv_cache_key()
    if not force and cache_key in _DOTENV_LOAD_CACHE:
        return _DOTENV_LOAD_CACHE[cache_key]

    dotenv_path = find_dotenv_path()
    if dotenv_path:
        _logger.info("Loading tooltoad environment variables from: %s", dotenv_path)
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
        if not loaded:
            _logger.warning("dotenv loading reported failure for: %s", dotenv_path)
        _DOTENV_LOAD_CACHE[cache_key] = dotenv_path
    else:
        _logger.debug(
            "No .env file found for tooltoad in TOOLTOAD_DOTENV_PATH, the "
            "current working directory, or the user home directory. Relying "
            "on pre-existing environment variables."
        )
        _DOTENV_LOAD_CACHE[cache_key] = None
    return _DOTENV_LOAD_CACHE[cache_key]
