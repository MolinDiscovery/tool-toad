import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_ENV_VARS = ("ORCA_EXE", "OPEN_MPI_DIR", "XTB_EXE", "GXTB_EXE", "XTBPATH")


def clean_subprocess_env(tmp: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in EXECUTABLE_ENV_VARS:
        env.pop(key, None)
    env["HOME"] = str(tmp / "home")
    env["TOOLTOAD_DOTENV_PATH"] = str(tmp / "missing.env")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(ROOT), env.get("PYTHONPATH", "")] if part
    )
    Path(env["HOME"]).mkdir()
    return env


class LazyImportTests(unittest.TestCase):
    def run_clean_python(self, code: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            return subprocess.run(
                [sys.executable, "-c", textwrap.dedent(code)],
                cwd=tmp,
                env=clean_subprocess_env(tmp),
                check=True,
                capture_output=True,
                text=True,
            )

    def test_tooltoad_import_does_not_import_calculators(self):
        proc = self.run_clean_python(
            """
            import sys
            import tooltoad

            assert "tooltoad.orca" not in sys.modules
            assert "tooltoad.xtb" not in sys.modules
            assert "tooltoad.gxtb" not in sys.modules
            print("ok")
            """
        )
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_lightweight_imports_do_not_need_external_executables(self):
        proc = self.run_clean_python(
            """
            from tooltoad.chemutils import ac2xyz
            from tooltoad.vis import MolTo3DGrid
            from tooltoad import ac2xyz as top_level_ac2xyz

            assert ac2xyz is top_level_ac2xyz
            print("ok")
            """
        )
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_top_level_calculator_imports_do_not_need_external_executables(self):
        proc = self.run_clean_python(
            """
            from tooltoad import (
                gxtb,
                gxtb_calculate,
                orca,
                orca_calculate,
                run_crest,
                xtb,
                xtb_calculate,
            )

            assert orca is orca_calculate
            assert xtb is xtb_calculate
            assert gxtb is gxtb_calculate
            assert callable(run_crest)
            print("ok")
            """
        )
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_direct_calculator_module_imports_do_not_need_external_executables(self):
        proc = self.run_clean_python(
            """
            import tooltoad.gxtb
            import tooltoad.orca
            import tooltoad.xtb

            print("ok")
            """
        )
        self.assertEqual(proc.stdout.strip(), "ok")


class RuntimeValidationTests(unittest.TestCase):
    def test_orca_calculate_requires_orca_exe_when_called(self):
        from tooltoad.orca import orca_calculate

        env = dict(os.environ)
        for key in EXECUTABLE_ENV_VARS:
            env.pop(key, None)
        with patch.dict(os.environ, env, clear=True):
            with patch("tooltoad.orca.find_and_load_dotenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "ORCA.*ORCA_EXE"):
                    orca_calculate(["H"], [[0.0, 0.0, 0.0]])

    def test_orca_calculate_requires_open_mpi_dir_when_no_set_env_is_given(self):
        from tooltoad.orca import orca_calculate

        env = {
            "ORCA_EXE": "/bin/echo",
            "PATH": os.environ.get("PATH", ""),
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("tooltoad.orca.find_and_load_dotenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "OPEN_MPI_DIR"):
                    orca_calculate(["H"], [[0.0, 0.0, 0.0]])

    def test_xtb_calculate_requires_resolvable_executable_when_called(self):
        from tooltoad.xtb import xtb_calculate

        with patch.dict(os.environ, {"PATH": ""}, clear=True):
            with patch("tooltoad.xtb.find_and_load_dotenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "xTB executable.*not found"):
                    xtb_calculate(["H"], [[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
