import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import tooltoad.vis as tooltoad_vis
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
    _angle_guide_geometry,
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

    def test_click_handler_draws_interactive_measurement_guides(self):
        handler = Py3DmolGridRenderer._CLICK_HANDLER

        self.assertIn("viewer._distLines", handler)
        self.assertIn("viewer._angleShapes", handler)
        self.assertIn("viewer.addCylinder", handler)
        self.assertIn("viewer.removeShape", handler)
        self.assertIn("DISTANCE_COLOR = 'green'", handler)
        self.assertIn("DISTANCE_RADIUS = 0.06", handler)
        self.assertIn("ANGLE_COLOR = 'orange'", handler)
        self.assertIn("ANGLE_GUIDE_CYLINDER_RADIUS = 0.035", handler)
        self.assertIn("ANGLE_GUIDE_ARC_SEGMENTS = 24", handler)
        self.assertIn("ANGLE_GUIDE_LABEL_RADIUS_FACTOR = 1.25", handler)
        self.assertIn("Math.min(a.index, c.index)", handler)
        self.assertIn("Math.max(a.index, c.index)", handler)

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
        distance_cylinder_call = next(
            call for call in fake.calls if call[0] == "addCylinder"
        )
        self.assertEqual(distance_cylinder_call[1][0]["color"], "green")
        self.assertEqual(distance_cylinder_call[1][0]["radius"], 0.06)

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

    def test_renderer_renders_angle_overlay_arms_arc_and_offset_label(self):
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
                        )
                    ],
                    overlays=[
                        AngleOverlay(
                            atom1=0,
                            atom2=1,
                            atom3=2,
                            label="90.0 deg",
                        )
                    ],
                )
            ],
            columns=1,
        )

        with patch("py3Dmol.view", return_value=fake):
            Py3DmolGridRenderer(scene).render()

        cylinder_calls = [call for call in fake.calls if call[0] == "addCylinder"]
        self.assertGreaterEqual(len(cylinder_calls), 14)
        np.testing.assert_allclose(
            list(cylinder_calls[0][1][0]["end"].values()),
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            list(cylinder_calls[1][1][0]["end"].values()),
            [0.0, 1.0, 0.0],
        )

        angle_label_call = next(
            call
            for call in fake.calls
            if call[0] == "addLabel" and call[1][0] == "90.0 deg"
        )
        label_position = angle_label_call[1][1]["position"]
        self.assertGreater(label_position["x"], 0.0)
        self.assertGreater(label_position["y"], 0.0)
        self.assertAlmostEqual(label_position["z"], 0.0)

    def test_legacy_mol_grid_uses_shared_click_handler(self):
        fake = FakeViewer()
        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

        with patch("py3Dmol.view", return_value=fake):
            tooltoad_vis._MolTo3DGrid_legacy(
                mol,
                show_confs=False,
                show_charges=False,
            )

        set_click_calls = [call for call in fake.calls if call[0] == "setClickable"]
        self.assertTrue(set_click_calls)
        self.assertIs(set_click_calls[-1][1][2], Py3DmolGridRenderer._CLICK_HANDLER)

    def test_angle_guide_geometry_for_right_angle(self):
        geometry = _angle_guide_geometry(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )

        self.assertIsNotNone(geometry)
        np.testing.assert_allclose(geometry.arm1_end, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(geometry.arm2_end, [0.0, 1.0, 0.0])
        self.assertEqual(len(geometry.arc_segments), 12)
        self.assertGreater(geometry.label_position[0], 0.0)
        self.assertGreater(geometry.label_position[1], 0.0)

    def test_angle_guide_geometry_for_obtuse_angle(self):
        geometry = _angle_guide_geometry(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        )

        self.assertIsNotNone(geometry)
        np.testing.assert_allclose(geometry.arm1_end, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            geometry.arm2_end,
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        )
        self.assertEqual(len(geometry.arc_segments), 12)
        self.assertGreater(geometry.label_position[1], 0.0)

    def test_angle_guide_geometry_returns_none_for_zero_length_vector(self):
        geometry = _angle_guide_geometry(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )

        self.assertIsNone(geometry)

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
