import unittest
from unittest.mock import patch

from tooltoad.scene3d import (
    AtomHighlight,
    AtomLabel,
    DistanceOverlay,
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    SceneCell,
    VibrationAnimation,
)
from tooltoad.vis import show_scene


class FakeViewer:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return method


class Scene3DTests(unittest.TestCase):
    def test_renderer_renders_molecule_animation_and_overlays(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    title="test",
                    models=[
                        MoleculeModel(
                            atoms=["H", "H"],
                            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]],
                            bonds=[(0, 1)],
                        )
                    ],
                    animations=[
                        VibrationAnimation(
                            mode=[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
                            frequency=-100.0,
                        )
                    ],
                    overlays=[
                        AtomLabel(atom=0, text="H1"),
                        AtomHighlight(atom=1),
                        DistanceOverlay(atom1=0, atom2=1, label="H-H"),
                    ],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake) as view:
            viewer = Py3DmolGridRenderer(scene).render()

        self.assertIs(viewer, fake)
        view.assert_called_once()
        call_names = [name for name, _, _ in fake.calls]
        self.assertIn("addModel", call_names)
        self.assertIn("mapAtomProperties", call_names)
        self.assertIn("vibrate", call_names)
        self.assertIn("animate", call_names)
        self.assertIn("addSphere", call_names)
        self.assertIn("addCylinder", call_names)
        self.assertIn("setClickable", call_names)

    def test_show_scene_returns_viewer_without_explicit_show(self):
        scene = GridScene(
            cells=[
                SceneCell(
                    title="test",
                    models=[
                        MoleculeModel(
                            atoms=["H"],
                            coords=[[0.0, 0.0, 0.0]],
                        )
                    ],
                )
            ]
        )

        with (
            patch("tooltoad.vis.Py3DmolGridRenderer.render", return_value="viewer") as render,
            patch("tooltoad.vis.Py3DmolGridRenderer.show") as show,
        ):
            viewer = show_scene(scene)

        render.assert_called_once()
        show.assert_not_called()
        self.assertEqual(viewer, "viewer")


if __name__ == "__main__":
    unittest.main()
