import os
from pathlib import Path
from dotenv import load_dotenv
import logging

_logger = logging.getLogger(__name__)


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


def find_and_load_dotenv():
    """Find and load the first Tooltoad dotenv file.

    Dotenv files are searched in this priority order:

    1. path specified by ``TOOLTOAD_DOTENV_PATH``;
    2. ``.env`` file in the current working directory;
    3. ``.env`` file in the user's home directory.

    Loads the first file found. Logs the outcome.
    """
    dotenv_path = find_dotenv_path()
    if dotenv_path:
        _logger.info("Loading tooltoad environment variables from: %s", dotenv_path)
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
        if not loaded:
            _logger.warning("dotenv loading reported failure for: %s", dotenv_path)
    else:
        _logger.warning(
            "No .env file found for tooltoad in TOOLTOAD_DOTENV_PATH, the "
            "current working directory, or the user home directory. Relying "
            "on pre-existing environment variables."
        )
