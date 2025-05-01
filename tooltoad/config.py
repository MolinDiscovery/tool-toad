import os
from pathlib import Path
from dotenv import load_dotenv
import logging

_logger = logging.getLogger(__name__)

def find_and_load_dotenv():
    """
    Finds and loads a .env file based on a priority order:
    1. Path specified by TOOLTOAD_DOTENV_PATH environment variable.
    2. .env file in the current working directory.
    3. .env file in the user's home directory.

    Loads the first file found. Logs the outcome.
    """
    dotenv_path = None
    search_locations_tried = []

    env_var_path_str = os.getenv("TOOLTOAD_DOTENV_PATH")
    if env_var_path_str:
        path_from_env = Path(env_var_path_str).resolve()
        search_locations_tried.append(f"env var TOOLTOAD_DOTENV_PATH ({path_from_env})")
        if path_from_env.is_file():
            dotenv_path = path_from_env
        else:
            _logger.warning(f"Path specified by TOOLTOAD_DOTENV_PATH does not exist or is not a file: {path_from_env}")

    if dotenv_path is None:
        cwd_path = Path.cwd() / ".env"
        search_locations_tried.append(f"current working directory ({cwd_path})")
        if cwd_path.is_file():
            dotenv_path = cwd_path

    if dotenv_path is None:
        try:
            home_path = Path.home() / ".env"
            search_locations_tried.append(f"user home directory ({home_path})")
            if home_path.is_file():
                dotenv_path = home_path
        except Exception:
            _logger.warning("Could not determine user home directory.")


    if dotenv_path:
        _logger.info(f"Loading tooltoad environment variables from: {dotenv_path}")
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
        if not loaded:
             _logger.warning(f"dotenv loading reported failure for: {dotenv_path}")
    else:
        _logger.warning(f"No .env file found for tooltoad in standard locations: {'; '.join(search_locations_tried)}. Relying on pre-existing environment variables.")
