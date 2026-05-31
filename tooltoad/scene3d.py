"""Scene-based 3D visualization primitives.

This module contains Tooltoad's low-level, dataframe-agnostic 3D scene model
and py3Dmol renderer. Higher-level packages should translate their own data
structures into :class:`GridScene` objects before rendering.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdchem import RWMol
from rdkit.Geometry import Point3D

from tooltoad.chemutils import ac2xyz


@dataclass(slots=True)
class MoleculeModel:
    """A single molecular model to render in a scene cell.

    Parameters
    ----------
    atoms
        Atomic symbols. Required when ``coords`` are used.
    coords
        Cartesian coordinates matching ``atoms``.
    bonds
        Optional zero-based atom-index pairs used as explicit display
        connectivity.
    mol
        Optional RDKit molecule. When supplied, ``atoms`` and ``coords`` are
        ignored for rendering.
    mol_block
        Optional pre-built mol block.
    conf_id
        RDKit conformer id used when rendering ``mol``.
    kekulize
        Whether to kekulize RDKit mol blocks.
    style
        py3Dmol style dictionary.
    show_atom_labels
        Draw atom index/``atomNote`` labels before rendering.
    show_charges
        Draw non-zero formal charges for RDKit models.
    bonds_to_remove
        Optional display-only bonds removed before rendering RDKit models.
    """

    atoms: Sequence[str] | None = None
    coords: Sequence[Sequence[float]] | None = None
    bonds: Sequence[tuple[int, int]] | None = None
    mol: Chem.Mol | None = None
    mol_block: str | None = None
    conf_id: int = 0
    kekulize: bool = True
    style: dict[str, Any] | None = None
    show_atom_labels: bool = False
    show_charges: bool = True
    bonds_to_remove: Sequence[tuple[int, int]] | None = None


@dataclass(slots=True)
class VibrationAnimation:
    """A normal-mode animation attached to a scene cell."""

    mode: Sequence[Sequence[float]]
    frequency: float | None = None
    num_frames: int = 20
    amplitude: float = 1.0
    fps: float | None = None
    reps: int = 100


@dataclass(slots=True)
class AtomLabel:
    """A text label anchored to an atom index."""

    atom: int
    text: str
    color: str = "black"
    background_color: str = "white"
    font_size: int = 12


@dataclass(slots=True)
class AtomHighlight:
    """A translucent sphere overlay centered on an atom."""

    atom: int
    color: str = "cyan"
    radius: float = 0.62
    alpha: float = 0.45


@dataclass(slots=True)
class DistanceOverlay:
    """A distance cylinder and optional label between two atoms."""

    atom1: int
    atom2: int
    label: str | None = None
    color: str = "green"
    radius: float = 0.06


@dataclass(slots=True)
class AngleOverlay:
    """An angle label for three atoms."""

    atom1: int
    atom2: int
    atom3: int
    label: str | None = None
    color: str = "orange"


SceneOverlay = AtomLabel | AtomHighlight | DistanceOverlay | AngleOverlay


@dataclass(slots=True)
class SceneCell:
    """One rendered cell in a 3D grid."""

    title: str | None = None
    models: list[MoleculeModel] = field(default_factory=list)
    animations: list[VibrationAnimation] = field(default_factory=list)
    overlays: list[SceneOverlay] = field(default_factory=list)


@dataclass(slots=True)
class GridScene:
    """A complete 3D grid scene."""

    cells: list[SceneCell]
    columns: int = 3
    cell_size: tuple[int, int] = (400, 400)
    linked: bool = False
    background_color: str | tuple[str, float] = ("blue", 0.1)
    transparent: bool = False


def coerce_to_mol(item: Chem.Mol | str | os.PathLike) -> Chem.Mol:
    """Coerce an RDKit molecule, SMILES string, or XYZ path into a molecule."""

    if isinstance(item, Chem.Mol):
        return item

    if isinstance(item, (str, os.PathLike)):
        s = str(item)
        p = Path(s)

        if p.suffix.lower() == ".xyz" and p.is_file():
            try:
                from tooltoad.utils import chemutils as _chemutils

                return _chemutils.read_xyz_file(
                    str(p), return_mol=True, useHueckel=True
                )
            except Exception:
                block = p.read_text()
                mol = Chem.MolFromXYZBlock(block)
                if mol is None:
                    raise ValueError(f"Could not parse XYZ file: {p}")
                try:
                    rdDetermineBonds.DetermineConnectivity(
                        mol, useHueckel=True
                    )
                except Exception:
                    pass
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                return mol

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(
                f"Could not parse SMILES string (and not an .xyz file): {s}"
            )
        return mol

    raise TypeError(
        "Expected an RDKit Mol, SMILES string, '.xyz' path, or os.PathLike."
    )


def ensure_3d_mol(mol: Chem.Mol, *, verbose: bool = False) -> Chem.Mol:
    """Return a molecule with at least one conformer."""

    if mol.GetNumConformers() > 0:
        return mol

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if verbose:
        res = AllChem.EmbedMolecule(mol, params)
    else:
        with rdBase.BlockLogs():
            res = AllChem.EmbedMolecule(mol, params)

    if res == 0:
        return mol

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useRandomCoords = True
    res = AllChem.EmbedMolecule(mol_h, params)
    if res == 0:
        return Chem.RemoveHs(mol_h)

    raise ValueError("Could not generate a 3D conformer for molecule.")


def molecule_model_from_atoms(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    *,
    bonds: Sequence[tuple[int, int]] | None = None,
    **kwargs: Any,
) -> MoleculeModel:
    """Build a molecule model from atoms, coordinates, and optional bonds."""

    return MoleculeModel(atoms=atoms, coords=coords, bonds=bonds, **kwargs)


def molecule_model_from_mol(
    mol: Chem.Mol,
    *,
    conf_id: int = 0,
    **kwargs: Any,
) -> MoleculeModel:
    """Build a molecule model from an RDKit molecule."""

    return MoleculeModel(mol=mol, conf_id=conf_id, **kwargs)


class Py3DmolGridRenderer:
    """Render :class:`GridScene` objects with py3Dmol."""

    _CLICK_HANDLER = r'''function(atom, viewer, event, container) {
        if(!viewer._picks)       viewer._picks = [];
        if(!viewer._distLabels)  viewer._distLabels = {};
        if(!viewer._anglePicks)  viewer._anglePicks = [];
        if(!viewer._angleLabels) viewer._angleLabels = {};

        if(event.shiftKey) {
            viewer._anglePicks.push(atom);
            if(viewer._anglePicks.length === 3) {
                var A = viewer._anglePicks[0],
                    B = viewer._anglePicks[1],
                    C = viewer._anglePicks[2];
                var key = [A.index,B.index,C.index].join('-');
                if(key in viewer._angleLabels) {
                    viewer.removeLabel(viewer._angleLabels[key]);
                    delete viewer._angleLabels[key];
                } else {
                    function vec(u,v){
                        return {x:u.x-v.x, y:u.y-v.y, z:u.z-v.z};
                    };
                    var vBA=vec(A,B), vBC=vec(C,B);
                    var dot=vBA.x*vBC.x+vBA.y*vBC.y+vBA.z*vBC.z;
                    var magBA=Math.sqrt(vBA.x*vBA.x+vBA.y*vBA.y+vBA.z*vBA.z);
                    var magBC=Math.sqrt(vBC.x*vBC.x+vBC.y*vBC.y+vBC.z*vBC.z);
                    var angle=(Math.acos(dot/(magBA*magBC))*(180/Math.PI))
                              .toFixed(2)+'°';
                    var lbl = viewer.addLabel(angle, {
                        position:{x:B.x,y:B.y,z:B.z},
                        backgroundColor:'blue',
                        fontColor:'white',
                        fontSize:12
                    });
                    viewer._angleLabels[key] = lbl;
                }
                viewer._anglePicks = [];
            }
        } else if(event.ctrlKey) {
            viewer._picks.push(atom);
            if(viewer._picks.length === 2) {
                var a=viewer._picks[0], b=viewer._picks[1];
                var key=[Math.min(a.index,b.index), Math.max(a.index,b.index)]
                        .join('-');
                if(key in viewer._distLabels) {
                    viewer.removeLabel(viewer._distLabels[key]);
                    delete viewer._distLabels[key];
                } else {
                    var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
                    var dist=Math.sqrt(dx*dx+dy*dy+dz*dz).toFixed(3)+' Å';
                    var mid={x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:(a.z+b.z)/2};
                    var lbl=viewer.addLabel(dist, {
                        position:mid,
                        backgroundColor:'grey',
                        fontColor:'white',
                        fontSize:12
                    });
                    viewer._distLabels[key]=lbl;
                }
                viewer._picks = [];
            }
        } else {
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            } else {
                atom.label = viewer.addLabel(atom.index, {
                    position:atom,
                    backgroundColor:'white',
                    fontColor:'black',
                    fontSize:12
                });
            }
        }
        viewer.render();
    }'''

    def __init__(self, scene: GridScene):
        self.scene = scene
        self.viewer = None
        self._cell_positions: dict[int, dict[int, tuple[float, float, float]]] = {}

    def render(self):
        """Build and return the py3Dmol viewer."""

        import py3Dmol

        if not self.scene.cells:
            raise ValueError("Cannot render an empty GridScene.")

        columns = max(1, int(self.scene.columns))
        rows = math.ceil(len(self.scene.cells) / columns)
        width = self.scene.cell_size[0] * columns
        height = self.scene.cell_size[1] * rows

        self.viewer = py3Dmol.view(
            width=width,
            height=height,
            viewergrid=(rows, columns),
            linked=self.scene.linked,
        )

        bg_color, bg_alpha = self._background_parts()
        self.viewer.setBackgroundColor(bg_color, bg_alpha)

        for idx, cell in enumerate(self.scene.cells):
            position = (idx // columns, idx % columns)
            self._render_cell(idx, cell, position)

        return self.viewer

    def show(self):
        """Render, show, and return the py3Dmol viewer."""

        viewer = self.render()
        viewer.show()
        return viewer

    def write_html(self, path: str | os.PathLike) -> None:
        """Render and export the scene to HTML."""

        viewer = self.viewer or self.render()
        output = Path(path)
        if output.parent:
            output.parent.mkdir(parents=True, exist_ok=True)
        viewer.write_html(str(output))
        self._inject_js_view_helpers(output)

    def _background_parts(self) -> tuple[str, float]:
        background = self.scene.background_color
        if isinstance(background, tuple):
            return background[0], float(background[1])
        return str(background), 0.0 if self.scene.transparent else 1.0

    def _render_cell(
        self,
        cell_index: int,
        cell: SceneCell,
        viewer_position: tuple[int, int],
    ) -> None:
        positions: dict[int, tuple[float, float, float]] = {}
        for model in cell.models:
            positions.update(
                self._add_model(model, viewer_position, existing_positions=positions)
            )

        self._cell_positions[cell_index] = positions

        for animation in cell.animations:
            self._add_vibration(animation, viewer_position)

        for overlay in cell.overlays:
            self._add_overlay(overlay, positions, viewer_position)

        if cell.title:
            self._add_screen_label(cell.title, viewer_position)

        self.viewer.setClickable(
            {},
            True,
            self._CLICK_HANDLER,
            viewer=viewer_position,
        )
        self.viewer.zoomTo(viewer=viewer_position)

    def _add_model(
        self,
        model: MoleculeModel,
        viewer_position: tuple[int, int],
        *,
        existing_positions: dict[int, tuple[float, float, float]],
    ) -> dict[int, tuple[float, float, float]]:
        block, fmt, positions, rdkit_mol = self._model_block(model)
        self.viewer.addModel(block, fmt, viewer=viewer_position)
        self.viewer.setStyle(
            {},
            model.style or {"stick": {}, "sphere": {"radius": 0.3}},
            viewer=viewer_position,
        )

        if model.show_atom_labels:
            self._add_atom_index_labels(model, positions, viewer_position, rdkit_mol)

        if model.show_charges and rdkit_mol is not None:
            self._add_charge_labels(rdkit_mol, model.conf_id, viewer_position)

        offset_positions = dict(existing_positions)
        offset = len(offset_positions)
        for atom_idx, xyz in positions.items():
            offset_positions[offset + atom_idx] = xyz
        return offset_positions

    def _model_block(
        self,
        model: MoleculeModel,
    ) -> tuple[str, str, dict[int, tuple[float, float, float]], Chem.Mol | None]:
        if model.mol_block is not None:
            return model.mol_block, "mol", {}, None

        if model.mol is not None:
            mol = Chem.Mol(model.mol)
            mol = ensure_3d_mol(mol)
            if model.bonds_to_remove:
                editable = RWMol(mol)
                for atom1, atom2 in model.bonds_to_remove:
                    if editable.GetBondBetweenAtoms(int(atom1), int(atom2)):
                        editable.RemoveBond(int(atom1), int(atom2))
                mol = editable.GetMol()
            block = Chem.MolToMolBlock(
                mol,
                confId=model.conf_id,
                kekulize=model.kekulize,
            )
            return block, "mol", self._positions_from_mol(mol, model.conf_id), mol

        if model.atoms is None or model.coords is None:
            raise ValueError("MoleculeModel requires mol, mol_block, or atoms+coords.")

        if model.bonds:
            mol = self._mol_from_atoms(
                model.atoms,
                model.coords,
                model.bonds,
            )
            block = Chem.MolToMolBlock(mol, kekulize=model.kekulize)
            return block, "mol", self._positions_from_mol(mol, 0), mol

        block = ac2xyz(model.atoms, model.coords)
        positions = {
            idx: (float(coord[0]), float(coord[1]), float(coord[2]))
            for idx, coord in enumerate(model.coords)
        }
        return block, "xyz", positions, None

    @staticmethod
    def _mol_from_atoms(
        atoms: Sequence[str],
        coords: Sequence[Sequence[float]],
        bonds: Sequence[tuple[int, int]],
    ) -> Chem.Mol:
        editable = Chem.RWMol()
        for symbol in atoms:
            editable.AddAtom(Chem.Atom(str(symbol)))
        for begin, end in bonds:
            begin_i = int(begin)
            end_i = int(end)
            if begin_i == end_i:
                continue
            if editable.GetBondBetweenAtoms(begin_i, end_i) is None:
                editable.AddBond(begin_i, end_i, Chem.BondType.SINGLE)
        mol = editable.GetMol()
        mol.UpdatePropertyCache(strict=False)
        conformer = Chem.Conformer(len(atoms))
        for atom_idx, coord in enumerate(coords):
            conformer.SetAtomPosition(
                int(atom_idx),
                Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
            )
        mol.AddConformer(conformer, assignId=True)
        return mol

    @staticmethod
    def _positions_from_mol(
        mol: Chem.Mol,
        conf_id: int,
    ) -> dict[int, tuple[float, float, float]]:
        conf = mol.GetConformer(conf_id)
        return {
            atom_idx: (
                float(conf.GetAtomPosition(atom_idx).x),
                float(conf.GetAtomPosition(atom_idx).y),
                float(conf.GetAtomPosition(atom_idx).z),
            )
            for atom_idx in range(mol.GetNumAtoms())
        }

    def _add_atom_index_labels(
        self,
        model: MoleculeModel,
        positions: dict[int, tuple[float, float, float]],
        viewer_position: tuple[int, int],
        mol: Chem.Mol | None,
    ) -> None:
        for atom_idx, xyz in positions.items():
            text = str(atom_idx)
            if mol is not None:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.HasProp("atomNote"):
                    text = atom.GetProp("atomNote")
            self.viewer.addLabel(
                text,
                {
                    "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "borderThickness": 1,
                    "fontSize": 12,
                },
                viewer=viewer_position,
            )

    def _add_charge_labels(
        self,
        mol: Chem.Mol,
        conf_id: int,
        viewer_position: tuple[int, int],
    ) -> None:
        positions = self._positions_from_mol(mol, conf_id)
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge == 0:
                continue
            xyz = positions[atom.GetIdx()]
            color = "red" if charge > 0 else "blue"
            sign = f"{abs(charge)}+" if charge > 0 else f"{abs(charge)}-"
            self.viewer.addLabel(
                sign,
                {
                    "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                    "inFront": True,
                    "fontSize": 16,
                    "fontColor": color,
                    "fontWeight": "bold",
                    "backgroundColor": "rgba(255,255,255,0)",
                    "backgroundOpacity": 0.6,
                },
                viewer=viewer_position,
            )

    def _add_vibration(
        self,
        animation: VibrationAnimation,
        viewer_position: tuple[int, int],
    ) -> None:
        propmap = [
            {"index": atom_idx, "props": {"dx": mode[0], "dy": mode[1], "dz": mode[2]}}
            for atom_idx, mode in enumerate(animation.mode)
        ]
        interval_ms = (
            50 if animation.fps is None else max(1, int(1000.0 / animation.fps))
        )
        self.viewer.mapAtomProperties(propmap, viewer=viewer_position)
        self.viewer.vibrate(
            animation.num_frames,
            animation.amplitude,
            True,
            viewer=viewer_position,
        )
        self.viewer.animate(
            {
                "loop": "backAndForth",
                "interval": interval_ms,
                "reps": animation.reps,
            },
            viewer=viewer_position,
        )

    def _add_overlay(
        self,
        overlay: SceneOverlay,
        positions: dict[int, tuple[float, float, float]],
        viewer_position: tuple[int, int],
    ) -> None:
        if isinstance(overlay, AtomLabel):
            xyz = positions[int(overlay.atom)]
            self.viewer.addLabel(
                overlay.text,
                {
                    "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                    "fontColor": overlay.color,
                    "backgroundColor": overlay.background_color,
                    "borderThickness": 1,
                    "fontSize": overlay.font_size,
                },
                viewer=viewer_position,
            )
            return

        if isinstance(overlay, AtomHighlight):
            xyz = positions[int(overlay.atom)]
            self.viewer.addSphere(
                {
                    "center": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                    "radius": overlay.radius,
                    "color": overlay.color,
                    "alpha": overlay.alpha,
                },
                viewer=viewer_position,
            )
            return

        if isinstance(overlay, DistanceOverlay):
            xyz1 = positions[int(overlay.atom1)]
            xyz2 = positions[int(overlay.atom2)]
            self.viewer.addCylinder(
                {
                    "start": {"x": xyz1[0], "y": xyz1[1], "z": xyz1[2]},
                    "end": {"x": xyz2[0], "y": xyz2[1], "z": xyz2[2]},
                    "radius": overlay.radius,
                    "color": overlay.color,
                },
                viewer=viewer_position,
            )
            if overlay.label:
                midpoint = tuple((a + b) * 0.5 for a, b in zip(xyz1, xyz2))
                self.viewer.addLabel(
                    overlay.label,
                    {
                        "position": {
                            "x": midpoint[0],
                            "y": midpoint[1],
                            "z": midpoint[2],
                        },
                        "backgroundColor": "white",
                        "fontColor": overlay.color,
                        "fontSize": 12,
                    },
                    viewer=viewer_position,
                )
            return

        if isinstance(overlay, AngleOverlay):
            center = positions[int(overlay.atom2)]
            text = overlay.label or f"{overlay.atom1}-{overlay.atom2}-{overlay.atom3}"
            self.viewer.addLabel(
                text,
                {
                    "position": {"x": center[0], "y": center[1], "z": center[2]},
                    "backgroundColor": "white",
                    "fontColor": overlay.color,
                    "fontSize": 12,
                },
                viewer=viewer_position,
            )

    def _add_screen_label(
        self,
        title: str,
        viewer_position: tuple[int, int],
    ) -> None:
        self.viewer.addLabel(
            title,
            {
                "fontColor": "black",
                "fontSize": 13,
                "backgroundColor": "white",
                "borderColor": "black",
                "borderWidth": 1,
                "useScreen": True,
                "inFront": True,
                "screenOffset": {"x": 10, "y": 0},
            },
            viewer=viewer_position,
        )

    @staticmethod
    def _inject_js_view_helpers(export_html_path: str | os.PathLike) -> None:
        html_path = Path(export_html_path)
        html = html_path.read_text(encoding="utf-8")

        grid_var_match = re.search(
            r"var\s+(viewergrid_\d+)\s*=\s*null;",
            html,
        )
        viewer_var_match = re.search(
            r"var\s+(viewer_\d+)\s*=\s*null;",
            html,
        )

        if grid_var_match is not None:
            grid_var = grid_var_match.group(1)
            marker = f"{grid_var}[0][0].render();"
            helper_js = _grid_helper_js(grid_var)
        elif viewer_var_match is not None:
            viewer_var = viewer_var_match.group(1)
            marker = f"{viewer_var}.render();"
            helper_js = _viewer_helper_js(viewer_var)
        else:
            print(
                "Warning: could not find viewer variable in exported HTML. "
                "Skipping JS helper injection."
            )
            return

        if marker not in html:
            print(
                "Warning: could not find render marker in exported HTML. "
                "Skipping JS helper injection."
            )
            return

        html = html.replace(marker, helper_js + "\n" + marker, 1)
        html_path.write_text(html, encoding="utf-8")


def _grid_helper_js(grid_var: str) -> str:
    return f"""
window.saveView = function(row, col) {{
    const v = {grid_var}[row][col].getView();
    console.log(JSON.stringify(v));
    return v;
}};

window.saveAllViews = function() {{
    const out = {{}};
    for (let r = 0; r < {grid_var}.length; r++) {{
        for (let c = 0; c < {grid_var}[r].length; c++) {{
            if ({grid_var}[r][c]) {{
                out[`${{r}},${{c}}`] = {grid_var}[r][c].getView();
            }}
        }}
    }}
    console.log(JSON.stringify(out, null, 2));
    return out;
}};

window.setViewForCell = function(row, col, view) {{
    {grid_var}[row][col].setView(view);
    {grid_var}[row][col].render();
}};

window.applyViews = function(viewMap) {{
    for (const key in viewMap) {{
        const [r, c] = key.split(",").map(Number);
        if ({grid_var}[r] && {grid_var}[r][c]) {{
            {grid_var}[r][c].setView(viewMap[key]);
            {grid_var}[r][c].render();
        }}
    }}
}};

const fixedViews = {{
    // "0,0": [1, 2, 3, 4, 5, 6, 7, 8],
}};

applyViews(fixedViews);
"""


def _viewer_helper_js(viewer_var: str) -> str:
    return f"""
window.saveView = function() {{
    const v = {viewer_var}.getView();
    console.log(JSON.stringify(v));
    return v;
}};

window.saveAllViews = function() {{
    const out = {{"0,0": {viewer_var}.getView()}};
    console.log(JSON.stringify(out, null, 2));
    return out;
}};

window.setViewForCell = function(row, col, view) {{
    {viewer_var}.setView(view);
    {viewer_var}.render();
}};

window.applyViews = function(viewMap) {{
    if (viewMap["0,0"]) {{
        {viewer_var}.setView(viewMap["0,0"]);
        {viewer_var}.render();
    }}
}};

const fixedViews = {{
    // "0,0": [1, 2, 3, 4, 5, 6, 7, 8],
}};

applyViews(fixedViews);
"""
