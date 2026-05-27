import contextlib
import os
import random
import shutil
import signal
import string
import subprocess
import warnings
from pathlib import Path
from typing import Generator

import joblib

alphabet = string.ascii_lowercase + string.digits

STANDARD_PROPERTIES = {"xtb": {"total energy": "electronic_energy"}, "orca": {}}


def stream(
    cmd: str, cwd: None | Path = None, shell: bool = True
) -> Generator[str, None, None]:
    """Execute a command and stream combined stdout/stderr."""
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=shell,
        cwd=cwd,
        preexec_fn=os.setsid,
        bufsize=1,
    ) as process:
        try:
            for line in iter(process.stdout.readline, ""):
                yield line
        except KeyboardInterrupt:
            print("\nCtrl+C pressed. Terminating the process...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
            print("Process terminated.")
        finally:
            process.stdout.close()
            process.wait()


def check_executable(executable: str):
    """Check if executable is in PATH."""
    results = stream(f"which {executable}")
    result = next(results)
    if result.startswith("which: no"):
        warnings.warn(f"Executable {executable} not found in PATH")


def require_executable(
    executable: str | None,
    *,
    env_var: str | None = None,
    default: str | None = None,
    executable_name: str | None = None,
) -> str:
    """Resolve and require an executable at runtime.

    Parameters
    ----------
    executable : str or None
        Explicit executable command or path supplied by the caller.
    env_var : str or None, optional
        Environment variable to use when ``executable`` is not supplied.
    default : str or None, optional
        Fallback executable command when neither ``executable`` nor ``env_var``
        resolve to a value.
    executable_name : str or None, optional
        Human-readable executable name used in error messages.

    Returns
    -------
    str
        The resolved executable command or path.

    Raises
    ------
    RuntimeError
        If no executable is configured or if the configured executable cannot
        be found.
    """
    cmd = executable
    if cmd is None and env_var:
        cmd = os.getenv(env_var)
    if cmd is None:
        cmd = default

    label = executable_name or env_var or "Executable"
    if not cmd:
        env_hint = f" Set {env_var} or pass an executable explicitly." if env_var else ""
        raise RuntimeError(f"{label} executable is not configured.{env_hint}")

    if shutil.which(cmd) is None:
        if env_var and executable is None:
            raise RuntimeError(
                f"{label} executable from {env_var} was not found: {cmd!r}. "
                f"Set {env_var} to a valid executable path or command."
            )
        raise RuntimeError(
            f"{label} executable was not found: {cmd!r}. "
            "Pass a valid executable path or ensure it is available on PATH."
        )

    return cmd


class WorkingDir:
    def __init__(self, root: str = ".", name: str = None) -> None:
        self.root = Path(root)
        self.name = name if name else self._random_str()
        self.dir = self.root / self.name
        self.create()

    def __str__(self) -> str:
        return str(self.dir.resolve())

    def __repr__(self) -> str:
        return self.__str__()

    def __truediv__(self, name: str) -> str:
        return self.dir / name

    def _random_str(self) -> str:
        name = "_" + "".join(random.choices(alphabet, k=6))
        while (self.root / name).exists():
            name = "".join(random.choices(alphabet, k=6))
        return name

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        try:
            # print("removing ", self.dir.absolute())
            shutil.rmtree(self.dir.absolute())
        except FileNotFoundError:
            pass


class WorkingFile:
    def __init__(self, root: str = ".", filename: str = None, mode="w") -> None:
        self.root = Path(root)
        self.filename = filename if filename else self._random_str()
        self.mode = mode
        self.path = self.root / self.filename

    def _random_str(self) -> str:
        name = "".join(random.choices(alphabet, k=6)) + ".ttxt"
        while (self.root / name).exists():
            name = "".join(random.choices(alphabet, k=6)) + ".ttxt"
        return name

    def __str__(self) -> str:
        return str(self.path.resolve())

    def __repr__(self) -> str:
        return self.__str__()

    def create(self) -> None:
        with open(str(self), self.mode) as _:
            pass

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass

    @property
    def stem(self):
        return str(self.path.stem)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given
    as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
