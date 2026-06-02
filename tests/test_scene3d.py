import unittest
from unittest.mock import patch

import numpy as np

from tooltoad.scene3d import (
    AngleOverlay,
    AtomHighlight,
    AtomLabel,
    ArrowOverlay,
    DistanceOverlay,
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    SceneCell,
    ScreenLabelOverlay,
    VibrationAnimation,
    normalize_bond_pairs,
)
from tooltoad.vis import RxnTo3DGrid, reaction_scene_cells, show_scene


class FakeViewer:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return method


class Scene3DTests(unittest.TestCase):
    def test_normalize_bond_pairs_accepts_numpy_arrays(self):
        bonds = np.array([[0, 1], [1, 2]], dtype=object)

        self.assertEqual(normalize_bond_pairs(bonds), [(0, 1), (1, 2)])
        self.assertEqual(MoleculeModel(bonds=bonds).bonds, [(0, 1), (1, 2)])

    def test_normalize_bond_pairs_empty_values_return_none(self):
        self.assertIsNone(normalize_bond_pairs(None))
        self.assertIsNone(normalize_bond_pairs([]))
        self.assertIsNone(normalize_bond_pairs(np.array([], dtype=object)))
        self.assertIsNone(MoleculeModel(bonds=[]).bonds)

    def test_normalize_bond_pairs_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "exactly two atom indices"):
            normalize_bond_pairs([[0, 1, 2]])

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

    def test_renderer_renders_arrow_and_screen_label_overlays(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    overlays=[
                        ArrowOverlay(),
                        ScreenLabelOverlay("status", font_color="green"),
                    ]
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            Py3DmolGridRenderer(scene).render()

        call_names = [name for name, _, _ in fake.calls]
        self.assertIn("addArrow", call_names)
        self.assertIn("addLabel", call_names)

    def test_renderer_renders_angle_overlay_as_arc(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    models=[
                        MoleculeModel(
                            atoms=["C", "C", "C"],
                            coords=[
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                            bonds=[(0, 1), (1, 2)],
                        )
                    ],
                    overlays=[AngleOverlay(atom1=0, atom2=1, atom3=2)],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            Py3DmolGridRenderer(scene).render()

        call_names = [name for name, _, _ in fake.calls]
        self.assertGreater(call_names.count("addCylinder"), 2)
        self.assertIn("addLabel", call_names)

    def test_renderer_handles_degenerate_angle_overlay(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    models=[
                        MoleculeModel(
                            atoms=["C", "C", "C"],
                            coords=[
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        )
                    ],
                    overlays=[AngleOverlay(atom1=0, atom2=1, atom3=2)],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            Py3DmolGridRenderer(scene).render()

        self.assertIn("addLabel", [name for name, _, _ in fake.calls])

    def test_renderer_handles_collinear_angle_overlay(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    models=[
                        MoleculeModel(
                            atoms=["C", "C", "C"],
                            coords=[
                                [-1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                            ],
                        )
                    ],
                    overlays=[AngleOverlay(atom1=0, atom2=1, atom3=2)],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            Py3DmolGridRenderer(scene).render()

        call_names = [name for name, _, _ in fake.calls]
        self.assertGreater(call_names.count("addCylinder"), 2)
        self.assertIn("addLabel", call_names)

    def test_renderer_accepts_numpy_array_bonds(self):
        fake = FakeViewer()
        scene = GridScene(
            cells=[
                SceneCell(
                    title="test",
                    models=[
                        MoleculeModel(
                            atoms=["H", "H"],
                            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]],
                            bonds=np.array([[0, 1]], dtype=object),
                        )
                    ],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            viewer = Py3DmolGridRenderer(scene).render()

        self.assertIs(viewer, fake)
        self.assertIn("addModel", [name for name, _, _ in fake.calls])

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

    def test_reaction_scene_cells_return_reactants_arrow_and_products(self):
        cells = reaction_scene_cells("CCO>>CC=O")

        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0].title, "Reactants")
        self.assertEqual(cells[2].title, "Products")
        self.assertEqual(len(cells[0].models), 1)
        self.assertEqual(len(cells[2].models), 1)
        self.assertTrue(
            any(isinstance(overlay, ArrowOverlay) for overlay in cells[1].overlays)
        )

    def test_reaction_scene_cells_add_bond_change_overlays(self):
        broken = reaction_scene_cells(
            "[CH3:1][OH:2]>>[CH3:1].[OH:2]",
            show_bond_changes=True,
        )
        formed = reaction_scene_cells(
            "[CH3:1].[OH:2]>>[CH3:1][OH:2]",
            show_bond_changes=True,
        )
        changed = reaction_scene_cells(
            "[C:1]=[O:2]>>[C:1][O:2]",
            show_bond_changes=True,
        )

        self.assertTrue(
            any(
                isinstance(overlay, DistanceOverlay) and overlay.color == "red"
                for overlay in broken[0].overlays
            )
        )
        self.assertTrue(
            any(
                isinstance(overlay, DistanceOverlay) and overlay.color == "green"
                for overlay in formed[2].overlays
            )
        )
        self.assertTrue(
            any(
                isinstance(overlay, DistanceOverlay) and overlay.color == "orange"
                for overlay in changed[0].overlays + changed[2].overlays
            )
        )

    def test_reaction_scene_cells_add_charge_change_overlays(self):
        cells = reaction_scene_cells(
            "[N+:1]>>[N:1]",
            show_charge_changes=True,
        )

        self.assertTrue(
            any(
                isinstance(overlay, AtomHighlight) and overlay.color == "blue"
                for overlay in cells[0].overlays + cells[2].overlays
            )
        )

    def test_reaction_scene_cells_h_mode_none_hides_hydrogen_models(self):
        cells = reaction_scene_cells("O>>O", h_mode="none")

        for cell in (cells[0], cells[2]):
            for model in cell.models:
                self.assertEqual(model.hide_elements, ("H",))

    def test_rxn_to_3d_grid_uses_scene_renderer(self):
        with (
            patch("py3Dmol.view") as direct_view,
            patch(
                "tooltoad.vis.Py3DmolGridRenderer.show",
                return_value="viewer",
            ) as show,
        ):
            result = RxnTo3DGrid("CCO>>CC=O")

        self.assertIsNone(result)
        direct_view.assert_not_called()
        show.assert_called_once()

    def test_rxn_to_3d_grid_export_uses_scene_renderer(self):
        with (
            patch("tooltoad.vis.Py3DmolGridRenderer.show", return_value="viewer"),
            patch("tooltoad.vis.Py3DmolGridRenderer.write_html") as write_html,
        ):
            RxnTo3DGrid("CCO>>CC=O", export_HTML="rxn.html")

        write_html.assert_called_once_with("rxn.html")


if __name__ == "__main__":
    unittest.main()
