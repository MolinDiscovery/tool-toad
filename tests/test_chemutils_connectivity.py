import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tooltoad.chemutils import gfnff_connectivity


class GfnffConnectivityTests(unittest.TestCase):
    def test_uses_configured_xtb_exe_for_connectivity(self):
        commands = []

        def fake_stream(cmd, cwd=None):
            commands.append(cmd)
            Path(cwd, "gfnff_lists.json").write_text(
                json.dumps({"blist": [[1, 2, 1.0]]})
            )
            return iter(())

        with patch.dict("os.environ", {"XTB_EXE": "/normal/xtb"}, clear=False):
            with patch("tooltoad.chemutils.stream", side_effect=fake_stream):
                adj = gfnff_connectivity(
                    ["H", "H"],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    charge=0,
                    multiplicity=1,
                    scr=".",
                )

        self.assertIn("/normal/xtb --gfnff", commands[0])
        self.assertEqual(adj.tolist(), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
