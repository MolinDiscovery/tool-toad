import os
import unittest
from unittest.mock import patch

from tooltoad.gxtb import (
    _resolve_gxtb_cmd,
    _with_gxtb_option,
    gxtb_calculate,
    mock_gxtb_calculate,
)
from tooltoad.xtb import read_gradients, read_xtb_results


class GxtbTests(unittest.TestCase):
    def test_resolve_gxtb_cmd_requires_env_or_argument(self):
        env = dict(os.environ)
        env.pop("GXTB_EXE", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GXTB_EXE is not set"):
                _resolve_gxtb_cmd(None)

    def test_resolve_gxtb_cmd_uses_argument_or_env(self):
        with patch.dict(os.environ, {"GXTB_EXE": "/env/xtb"}, clear=False):
            self.assertEqual(_resolve_gxtb_cmd(None), "/env/xtb")
            self.assertEqual(_resolve_gxtb_cmd("/arg/xtb"), "/arg/xtb")

    def test_with_gxtb_option_forces_gxtb_and_keeps_user_options(self):
        self.assertEqual(_with_gxtb_option({"opt": None}), {"gxtb": None, "opt": None})
        self.assertEqual(_with_gxtb_option(None), {"gxtb": None})

    def test_gxtb_calculate_delegates_to_xtb_with_gxtb_binary_and_option(self):
        with patch("tooltoad.gxtb.xtb_calculate") as calc:
            calc.return_value = {"normal_termination": True}
            result = gxtb_calculate(
                atoms=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 1]],
                charge=0,
                multiplicity=1,
                options={"grad": None},
                n_cores=2,
                gxtb_cmd="/special/xtb",
            )

        self.assertTrue(result["normal_termination"])
        kwargs = calc.call_args.kwargs
        self.assertEqual(kwargs["xtb_cmd"], "/special/xtb")
        self.assertEqual(kwargs["options"], {"gxtb": None, "grad": None})
        self.assertEqual(kwargs["n_cores"], 2)

    def test_reuses_xtb_parsers_for_energy_and_gradients(self):
        lines = [
            " * xtb version 6.7.1\n",
            " program call               : xtb mol.xyz --gxtb --grad\n",
            " SUMMARY\n",
            " ....................\n",
            " total energy              -1.234500 Eh\n",
            " ::::::::::::::::::::\n",
            " normal termination of xtb\n",
            " * wall-time:     0 d,  0 h,  0 min,  1.0 sec\n",
        ]
        results = read_xtb_results(lines)
        self.assertEqual(results["electronic_energy"], -1.2345)

        grad_lines = [
            "$gradient\n",
            "#\n",
            "1.0\n",
            "2.0\n",
            "3.0\n",
            "#\n",
        ]
        self.assertEqual(read_gradients(grad_lines).tolist(), [[1.0, 2.0, 3.0]])

    def test_mock_gxtb_calculate_matches_basic_result_shape(self):
        result = mock_gxtb_calculate(
            atoms=["H"],
            coords=[[0.0, 0.0, 0.0]],
            options={"opt": None},
        )

        self.assertTrue(result["normal_termination"])
        self.assertEqual(result["options"], {"gxtb": None, "opt": None})
        self.assertEqual(result["opt_coords"], [[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
