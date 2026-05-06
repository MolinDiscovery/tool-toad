import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tooltoad.config import find_dotenv_path


class ConfigTests(unittest.TestCase):
    def test_find_dotenv_path_prefers_configured_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            configured = tmp / "configured.env"
            cwd_env = tmp / ".env"
            configured.write_text("XTB_EXE=/configured/xtb\n", encoding="utf-8")
            cwd_env.write_text("XTB_EXE=/cwd/xtb\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"TOOLTOAD_DOTENV_PATH": str(configured)},
                clear=False,
            ):
                with patch("pathlib.Path.cwd", return_value=tmp):
                    self.assertEqual(find_dotenv_path(), configured)

    def test_find_dotenv_path_falls_back_to_cwd_when_configured_missing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cwd_env = tmp / ".env"
            cwd_env.write_text("XTB_EXE=/cwd/xtb\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"TOOLTOAD_DOTENV_PATH": str(tmp / "missing.env")},
                clear=False,
            ):
                with patch("pathlib.Path.cwd", return_value=tmp):
                    self.assertEqual(find_dotenv_path(), cwd_env)

    def test_loads_home_dotenv(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            home = tmp / "home"
            home.mkdir()
            (home / ".env").write_text("GXTB_EXE=/home/gxtb\n", encoding="utf-8")

            env = dict(os.environ)
            env.pop("GXTB_EXE", None)
            env.pop("TOOLTOAD_DOTENV_PATH", None)
            env["HOME"] = str(home)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from tooltoad.config import find_and_load_dotenv; "
                        "find_and_load_dotenv(); "
                        "import os; print(os.environ['GXTB_EXE'])"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "/home/gxtb")

    def test_loads_dotenv_with_braced_home_expansion(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            env_path = tmp / "tooltoad.env"
            env_path.write_text("OPEN_MPI_DIR=${HOME}/openmpi\n", encoding="utf-8")

            env = dict(os.environ)
            env.pop("OPEN_MPI_DIR", None)
            env["HOME"] = "/example/home"
            env["TOOLTOAD_DOTENV_PATH"] = str(env_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from tooltoad.config import find_and_load_dotenv; "
                        "find_and_load_dotenv(); "
                        "import os; print(os.environ['OPEN_MPI_DIR'])"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "/example/home/openmpi")


if __name__ == "__main__":
    unittest.main()
