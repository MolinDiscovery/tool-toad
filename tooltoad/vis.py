import os
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import py3Dmol
from matplotlib import patches
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdDetermineBonds
from rdkit.Geometry import Point2D
from scipy.interpolate import CubicSpline

from tooltoad.chemutils import ac2mol, ac2xyz

# load matplotlib style
plt.style.use(os.path.dirname(__file__) + "/data/paper.mplstyle")
PAPER = Path("/groups/kemi/julius/opt/tm-catalyst-paper/figures")


def oneColumnFig(square: bool = False):
    """Create a figure that is one column wide.

    Args:
        square (bool, optional): Square figure. Defaults to False.

    Returns:
        (fig, ax): Figure and axes.
    """
    if square:
        size = (6, 6)
    else:
        size = (6, 4.187)
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def twoColumnFig(**kwargs):
    """Create a figure that is two column wide.

    Args:
        square (bool, optional): Square figure. Defaults to False.

    Returns:
        (fig, ax): Figure and axes.
    """
    size = (12, 4.829)
    fig, axs = plt.subplots(figsize=size, **kwargs)
    return fig, axs


def draw2d(
    mol: Chem.Mol,
    legend: str = None,
    atomLabels: dict = None,
    atomHighlights: dict = None,
    bondLineWidth: int = 1,
    size=(800, 600),
    blackwhite=True,
):
    """Create 2D depiction of molecule for publication.

    Args:
        mol (Chem.Mol): Molecule to render
        legend (str, optional): Legend string. Defaults to None.
        atomLabels (dict, optional): Dictionary of atomindices and atomlabels, f.x.:
                                     {17: 'H<sub>1</sub>', 18: 'H<sub>2</sub>'}.
                                     Defaults to None.
        atomHighlights (dict, optional): List of atoms to highlight,, f.x.:
                                         [(9, False, (0.137, 0.561, 0.984)),
                                         (15, True, (0, 0.553, 0))]
                                         First item is the atomindex, second is whether
                                         or not the highlight should be filled, and third
                                         is the color.
                                         Defaults to None.
        size (tuple, optional): Size of the drawing canvas. Defaults to (800, 600).
        blackwhite (bool, optional): Black and white color palet. Defaults to True.

    Returns:
        PIL.PNG: Image of the molecule.
    """
    d2d = Draw.MolDraw2DCairo(*size)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.NormalizeDepiction(mol)
    rdDepictor.StraightenDepiction(mol)
    dopts = d2d.drawOptions()
    dopts.legendFraction = 0.15
    dopts.legendFontSize = 45
    dopts.baseFontSize = 0.8
    dopts.additionalAtomLabelPadding = 0.1
    dopts.bondLineWidth = bondLineWidth
    dopts.scaleBondWidth = False
    if blackwhite:
        dopts.useBWAtomPalette()
    if atomLabels:
        for key, value in atomLabels.items():
            dopts.atomLabels[key] = value
    if legend:
        d2d.DrawMolecule(mol, legend=legend)
    else:
        d2d.DrawMolecule(mol)

    alpha = 0.4
    positions = []
    radii = []
    colors = []
    filled_bools = []
    if atomHighlights:
        for h in atomHighlights:
            filled = False
            color = (0.137, 0.561, 0.984)
            if isinstance(h, int):
                atomIdx = h
            elif len(h) == 2:
                atomIdx, filled = h
            elif len(h) == 3:
                atomIdx, filled, color = h
            else:
                raise ValueError("Invalid atom highlight {}".format(h))
            point = mol.GetConformer().GetAtomPosition(int(atomIdx))
            positions.append(Point2D(point.x, point.y))
            radii.append(0.35)
            colors.append(color)
            filled_bools.append(bool(filled))

        # draw filled circles first
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            if filled:
                color = (color[0], color[1], color[2], alpha)
                d2d.SetColour(color)
                d2d.SetFillPolys(True)
                d2d.SetLineWidth(0)
                d2d.DrawArc(pos, radius, 0.0, 360.0)

        # # now draw molecule again
        d2d.SetLineWidth(3)
        if legend:
            d2d.DrawMolecule(mol, legend=legend)
        else:
            d2d.DrawMolecule(mol)

        # now draw ring highlights
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            d2d.SetColour(color)
            d2d.SetFillPolys(False)
            # d2d.SetLineWidth(2.5)
            d2d.SetLineWidth(5)
            d2d.DrawArc(pos, radius, 0.0, 360.0)

        # and draw molecule again for whatever reason
        d2d.SetLineWidth(1)
        if legend:
            d2d.DrawMolecule(mol, legend=legend)
        else:
            d2d.DrawMolecule(mol)

        # now draw ring highlights again
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            if not filled:
                d2d.SetColour(color)
                d2d.SetFillPolys(False)
                # d2d.SetLineWidth(2.5)
                d2d.SetLineWidth(5)
                d2d.DrawArc(pos, radius, 0.0, 360.0)
    # finish drawing
    d2d.FinishDrawing()
    d2d.GetDrawingText()
    bio = BytesIO(d2d.GetDrawingText())
    img = Image.open(bio)
    return img


def DrawMolSvg(
    mol_or_smiles,
    w=200,
    h=200,
    fixed_bond_length=40,
    bond_line_width=2,
    use_aromatic_circle=True,
    add_atom_indices=False,
    kekulize=True,
    show=True,
    to_var=False,
    highlight_atom_indices=None,
    highlight_color=(0.90, 0.62, 0.00, 0.85),
    highlight_radius=None,
    highlight_as_circles=False,
    fill_highlights=None,
    dim_others=False,
    dim_color=(0.6, 0.6, 0.6),
    dim_bonds=True,
    dim_atomic_nums=(6,),
):
    """Render a clean 2D SVG depiction of a molecule with options to reduce
    visual crowding and highlight specific atoms.

    This helper is intended for figures where default RDKit depictions can
    appear cluttered (e.g., dense carbon frameworks or overlapping highlights).
    It provides control over bond lengths, line widths, selective dimming, and
    atom highlighting so key features stand out clearly.

    Accepts either an RDKit ``Chem.Mol`` or a SMILES string, generates 2D
    coordinates, and renders the molecule using ``MolDraw2DSVG``. By default
    the SVG is displayed inline and can optionally be returned.

    Args:
        mol_or_smiles: An RDKit molecule (``Chem.Mol``) or a SMILES string.
        w: SVG canvas width in pixels.
        h: SVG canvas height in pixels.
        fixed_bond_length: Fixed bond length used by the drawer.
        bond_line_width: Bond line width.
        use_aromatic_circle: Depict aromatic rings with circles.
        add_atom_indices: If True, draw atom indices.
        kekulize: If True, attempt to kekulize before drawing.
        show: If True, display the SVG inline.
        to_var: If True, return the SVG object.

        highlight_atom_indices: Atom index or iterable of indices to highlight.
        highlight_color: RGBA tuple for highlighted atoms.
        highlight_radius: Optional radius override for highlights.
        highlight_as_circles: Draw highlights as circles.
        fill_highlights: Whether highlight regions are filled.
        dim_others: Dim specified elements (e.g., carbon) to de-emphasize them.
        dim_color: RGB(A) tuple for dimmed atoms/bonds (RGB used).
        dim_bonds: Also dim bonds when dimming atoms.
        dim_atomic_nums: Atomic numbers to dim (default: carbon).

    Returns:
        If ``to_var`` is True, an ``IPython.display.SVG`` object; otherwise None.

    Raises:
        ValueError: If the SMILES cannot be parsed or indices are invalid.
        TypeError: If highlight indices are not integers.
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES.")
    else:
        mol = mol_or_smiles
        if mol is None:
            raise ValueError("mol_or_smiles is None.")

    mol = Chem.Mol(mol)

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass

    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.useAromaticCircle = use_aromatic_circle
    opts.bondLineWidth = bond_line_width
    opts.fixedBondLength = fixed_bond_length
    opts.addAtomIndices = add_atom_indices
    opts.addStereoAnnotation = False
    opts.prepareMolsBeforeDrawing = False

    if dim_others:
        # Dim only selected elements (default: carbon) so hetero colors remain.
        if dim_atomic_nums is None:
            atomic_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
        else:
            atomic_nums = set(dim_atomic_nums)

        palette_update = {anum: dim_color[:3] for anum in atomic_nums}
        try:
            opts.updateAtomPalette(palette_update)
        except Exception:
            try:
                opts.useBWAtomPalette()
                opts.updateAtomPalette(palette_update)
            except Exception:
                pass

        if dim_bonds:
            for meth in (
                "setBondLineColour",
                "setDefaultBondColour",
                "setDefaultBondColor",
            ):
                if hasattr(opts, meth):
                    try:
                        getattr(opts, meth)(dim_color[:3])
                        break
                    except Exception:
                        pass
            else:
                for attr in (
                    "bondLineColour",
                    "defaultBondColour",
                    "defaultBondColor",
                ):
                    if hasattr(opts, attr):
                        try:
                            setattr(opts, attr, dim_color[:3])
                            break
                        except Exception:
                            pass

    highlight_atoms = None
    highlight_atom_colors = None
    highlight_atom_radii = None
    highlight_bonds = None

    if highlight_atom_indices is not None:
        if isinstance(highlight_atom_indices, int):
            highlight_atoms = [highlight_atom_indices]
        else:
            highlight_atoms = list(highlight_atom_indices)

        n_atoms = mol.GetNumAtoms()
        for idx in highlight_atoms:
            if not isinstance(idx, int):
                raise TypeError("highlight_atom_indices must contain ints.")
            if idx < 0 or idx >= n_atoms:
                raise ValueError(
                    f"Atom index {idx} is out of range (0..{n_atoms - 1})."
                )

        if highlight_color is not None:
            opts.setHighlightColour(highlight_color)
            highlight_atom_colors = {idx: highlight_color for idx in highlight_atoms}

        if highlight_radius is not None:
            highlight_atom_radii = {idx: highlight_radius for idx in highlight_atoms}

        if highlight_as_circles:
            opts.atomHighlightsAreCircles = True

        if fill_highlights is not None:
            opts.fillHighlights = fill_highlights

        highlight_bonds = []

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
        highlightBonds=highlight_bonds,
    )
    drawer.FinishDrawing()

    svg = SVG(drawer.GetDrawingText())
    if show:
        display(svg)
    if to_var:
        return svg


def molGrid(images: List, buffer: int = 5, out_file: str = None):
    """Creates a grid of images.

    Args:
        images (List): List of lists of images.
        buffer (int, optional): Buffer between images. Defaults to 5.
        out_file (str, optional): Filename to save image to. Defaults to None.
    """
    max_width = max([max([img.width for img in imgs]) for imgs in images])
    max_height = max([max([img.height for img in imgs]) for imgs in images])
    max_num_rows = max([len(imgs) for imgs in images])
    fig_width = max_width * max_num_rows + buffer * (max_num_rows - 1)
    fig_height = max_height * len(images) + buffer * (len(images) - 1)
    res = Image.new("RGBA", (fig_width, fig_height))

    y = 0
    for imgs in images:
        x = 0
        for img in imgs:
            res.paste(img, (x, y))
            x += img.width + buffer
        y += img.height + buffer
    if out_file:
        res.save(out_file)
    else:
        return res


def draw3d(
    mols: list,
    transparent: bool = True,
    overlay: bool = False,
    confId: int = -1,
    atomlabel: bool = False,
    kekulize: bool = True,
    width: float = 600,
    height: float = 400,
):
    """Draw 3D structures in Jupyter notebook using py3Dmol.

    Args:
        mols (list): List of RDKit molecules.
        overlay (bool, optional): Overlay molecules. Defaults to False.
        confId (int, optional): Conformer ID. Defaults to -1.
        atomlabel (bool, optional): Show all atomlabels. Defaults to False.

    Returns:
        Py3Dmol.view: 3D view object.
    """
    p = py3Dmol.view(width=width, height=height)
    if not isinstance(mols, list):
        mols = [mols]
    for mol in mols:
        if isinstance(mol, (str, Path)):
            p = Path(mol)
            if p.suffix == ".xyz":
                xyz_f = open(mol)
                line = xyz_f.read()
                xyz_f.close()
                p.addModel(line, "xyz")
            else:
                raise NotImplementedError("Only xyz file is supported")
        elif isinstance(mol, Chem.rdchem.Mol):  # if rdkit.mol
            if overlay:
                for conf in mol.GetConformers():
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId(), kekulize=kekulize)
                    p.addModel(mb, "sdf")
            else:
                mb = Chem.MolToMolBlock(mol, confId=confId, kekulize=kekulize)
                p.addModel(mb, "sdf")
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    p.setBackgroundColor("0xeeeeee", int(not (transparent)))
    if atomlabel:
        p.addPropertyLabels("index")
    else:
        p.setClickable(
            {},
            True,
            """function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: 'white', fontColor:'black'});
                   }}""",
        )
    p.zoomTo()
    return p


def show_irc(irc):
    view = py3Dmol.view(width=800, height=400, viewergrid=(1, 2))
    try:
        forward = ac2mol(
            irc["irc"]["forward"]["atoms"], irc["irc"]["forward"]["opt_coords"]
        )
        rdDetermineBonds.DetermineBondOrders(forward)
        sdf = Chem.MolToMolBlock(forward)
        view.addModel(sdf, "sdf", viewer=(0, 0))
        view.zoomTo(viewer=(0, 0))
        view.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
        view.setBackgroundColor("0xeeeeee", 0)
    except Exception as e:
        print(e)

    try:
        backward = ac2mol(
            irc["irc"]["backward"]["atoms"], irc["irc"]["backward"]["opt_coords"]
        )
        rdDetermineBonds.DetermineBondOrders(backward)
        sdf = Chem.MolToMolBlock(backward)
        view.addModel(sdf, "sdf", viewer=(0, 1))
        view.zoomTo(viewer=(0, 1))
        view.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
        view.setBackgroundColor("0xeeeeee", 0)
    except Exception as e:
        print(e)

    return view


def show_traj(input: str | dict, width: float = 600, height: float = 400):
    """Show xyz trajectory.

    Args:
        input (str | dict): Trajectory, either as a string or a dict["traj"].
        width (float, optional): Width of py3dmol. Defaults to 600.
        height (float, optional): Height of py3dmol. Defaults to 400.
    """
    if isinstance(input, str):
        traj = input
    elif isinstance(input, dict):
        traj = input.get("traj")
        if not traj:
            raise ValueError
    else:
        raise ValueError
    p = py3Dmol.view(width=width, height=height)
    p.addModelsAsFrames(traj, "xyz")
    p.animate({"loop": "forward", "reps": 3})
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    return p


def interpolate_trajectory(positions, frame_multiplier: int = 10):
    positions = np.asarray(positions)
    num_frames = positions.shape[0]
    frames = np.arange(num_frames)
    new_frames = np.linspace(0, num_frames - 1, num=num_frames * frame_multiplier)

    # Vectorized cubic spline interpolation
    spline = CubicSpline(frames, positions, axis=0)
    return spline(new_frames)


def show_traj_v2(
    atoms, coords, width: float = 600, height: float = 400, interpolation: int = 10
):
    """Show xyz trajectory.

    Args:
        atoms (list): List of atom symbols.
        coords (list): List of coordinates.
        width (float, optional): Width of py3dmol. Defaults to 600.
        height (float, optional): Height of py3dmol. Defaults to 400.
    """
    if interpolation:
        coords = interpolate_trajectory(coords, frame_multiplier=interpolation)
    traj = ""
    for coord in coords:
        traj += ac2xyz(atoms, coord)
    p = py3Dmol.view(width=width, height=height)
    p.addModelsAsFrames(traj, "xyz")
    p.animate({"loop": "forward", "reps": 3})
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    return p


def show_vibs(
    results: dict,
    vId: int = 0,
    width: float = 600,
    height: float = 400,
    numFrames: int = 20,
    amplitude: float = 1.0,
    transparent: bool = True,
    fps: float | None = None,
    reps: int = 100,
    background_color: str | None = None,
    vIds: list[int] | int | None = None,
    viewergrid: tuple[int, int] | None = None,
    linked: bool = True,
):
    """Show normal mode vibration."""
    import math
    import py3Dmol

    input = results
    atoms = input["atoms"]
    opt_coords = input["opt_coords"]
    xyz = ac2xyz(atoms, opt_coords)

    if fps is None:
        interval_ms = 50
    else:
        interval_ms = max(1, int(1000.0 / fps))

    color = background_color or "0xeeeeee"
    alpha = 0 if transparent else 1

    # ---- Single-view path (original behavior) ----
    if vIds is None or (isinstance(vIds, list) and len(vIds) == 1) \
            or isinstance(vIds, int):
        mode_index = vId
        if isinstance(vIds, int):
            mode_index = vIds
        elif isinstance(vIds, list) and len(vIds) == 1:
            mode_index = vIds[0]

        vib = input.get("vibs")[mode_index]
        mode = vib["mode"]
        frequency = vib["frequency"]

        p = py3Dmol.view(width=width, height=height)
        p.addModel(xyz, "xyz")

        propmap = []
        for j, m in enumerate(mode):
            propmap.append({"index": j, "props": {
                "dx": m[0], "dy": m[1], "dz": m[2]}})
        p.mapAtomProperties(propmap)
        p.vibrate(numFrames, amplitude, True)
        p.animate({"loop": "backAndForth",
                   "interval": interval_ms, "reps": reps})
        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})

        p.setBackgroundColor(color, alpha)
        p.zoomTo()
        print(f"Normal mode {mode_index} with frequency {frequency} cm^-1")
        return p

    # ---- Grid path (multiple modes) ----
    if isinstance(vIds, int):
        mode_indices = [vIds]
    else:
        mode_indices = list(vIds)

    n = len(mode_indices)
    if viewergrid is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = viewergrid

    p = py3Dmol.view(width=width, height=height,
                     viewergrid=(rows, cols), linked=linked)

    # Build each cell
    for i, mi in enumerate(mode_indices):
        vib = input.get("vibs")[mi]
        mode = vib["mode"]
        frequency = vib["frequency"]

        r, c = divmod(i, cols)
        p.addModel(xyz, "xyz", viewer=(r, c))

        propmap = []
        for j, m in enumerate(mode):
            propmap.append({"index": j, "props": {
                "dx": m[0], "dy": m[1], "dz": m[2]}})
        p.mapAtomProperties(propmap, viewer=(r, c))

        # per-cell vibrate + animate
        p.vibrate(numFrames, amplitude, True, viewer=(r, c))
        p.animate({"loop": "backAndForth",
                   "interval": interval_ms, "reps": reps}, viewer=(r, c))

        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}}, viewer=(r, c))
        p.setBackgroundColor(color, alpha, viewer=(r, c))
        p.zoomTo(viewer=(r, c))
        print(f"Normal mode {mi} with frequency {frequency} cm^-1")

    return p


dopts = Chem.Draw.rdMolDraw2D.MolDrawOptions()
try:
    dopts.legendFontSize = 18
    dopts.minFontSize = 30
    dopts.padding = 0.05
    dopts.atomLabelFontSize = 40
    dopts.drawMolsSameScale = True
    dopts.centreMoleculesBeforeDrawing = True
    dopts.prepareMolsForDrawing = True
except AttributeError:
    pass


def drawMolInsert(
    ax_below: plt.axes,
    mol: Chem.Mol,
    pos: tuple,
    xSize: float = 0.5,
    aspect: float = 0.33,
    zorder: int = 5,
) -> plt.axes:
    """Draw molecule in a subplot.

    Args:
        ax_below (plt.axes): Axes to draw molecule on top of.
        mol (Chem.Mol): RDKit molecule to draw.
        pos (tuple): (x0, y0) position of insert.
        xSize (float, optional): Size of x dimension of insert. Defaults to 0.5.
        aspect (float, optional): Aspect ratio of insert. Defaults to 0.33.

    Returns:
        plt.axes: Axes of insert.
    """
    ax = ax_below.inset_axes([*pos, xSize, xSize * aspect])
    resolution = 1000
    im = Draw.MolToImage(
        mol,
        size=(int(resolution * xSize), int(resolution * xSize * aspect)),
        options=dopts,
    )
    ax.imshow(im, origin="upper", zorder=zorder)
    ax.axis("off")
    return ax


def addFrame(
    ax_around: plt.axes,
    ax_below: plt.axes,
    linewidth: int = 6,
    edgecolor: str = "crimson",
    nShadows: int = 25,
    shadowLinewidth: float = 0.05,
    molZorder: int = 4,
) -> None:
    """Draw Frame around axes.

    Args:
        ax_around (plt.axes): Axes to draw frame around.
        ax_below (plt.axes): Axes to draw frame on.
        linewidth (int, optional): Linewidth of frame. Defaults to 6.
        edgecolor (str, optional): Color of frame. Defaults to "crimson".
        nShadows (int, optional): Resolution of shadow. Defaults to 25.
        shadowLinewidth (float, optional): Extend of shadow. Defaults to 0.05.
        molZorder (int, optional): ZOrder of Mol. Defaults to 4.
    """
    images = ax_around.get_images()
    assert len(images) == 1, f"Found {len(images)} images in {ax_around}, expected 1"
    img = images[0]
    frame = patches.FancyBboxPatch(
        (0, 0),
        *reversed(img.get_size()),
        boxstyle="round",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
        transform=ax_around.transData,
        zorder=molZorder,
    )
    ax_below.add_patch(frame)
    if nShadows:
        for i in range(nShadows):
            shadow = patches.FancyBboxPatch(
                (0, 0),
                *reversed(img.get_size()),
                boxstyle="round",
                linewidth=shadowLinewidth * i**2 + 0.2,
                edgecolor="black",
                facecolor="none",
                alpha=0.7 / nShadows,
                transform=ax_around.transData,
                zorder=frame.get_zorder() - 1,
            )
            ax_below.add_patch(shadow)


def plot_parity(ax: plt.axes, tick_base: int = None, **kwargs) -> None:
    """Make square plot with parity line.

    Args:
        ax (plt.axes): Axes to plot on.
        tick_base (int, optional): Tick base. Defaults to 10.
    """
    ax.set_aspect("equal")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax_min = min(xlim[0], ylim[0])
    ax_max = max(xlim[1], ylim[1])

    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        c="grey",
        linestyle="dashed",
        zorder=0,
        **kwargs,
    )
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    if tick_base is None:
        ax_len = ax_max - ax_min
        tick_base = round(ax_len / 4)

    loc = plticker.MultipleLocator(base=tick_base)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)


def plot_residual_histogram(
    ax: plt.axes,
    x: np.ndarray,
    y: np.ndarray,
    loc: list = [0.58, 0.13, 0.4, 0.4],
    bins: int = 15,
    xlabel: str = "Residual (kcal/mol)",
    **kwargs,
) -> plt.axes:
    """Plot Histogram insert of residuals.

    Args:
        ax (plt.axes): Axes to plot on.
        x (np.ndarray): x data
        y (np.ndarray): y data
        loc (list, optional): Location of insert. Defaults to [0.58, 0.13, 0.4, 0.4].
        bins (int, optional): Number of bins in histogram. Defaults to 15.
        xlabel (str, optional): Label on x axis. Defaults to "Residual (kcal/mol)".

    Returns:
        plt.axes: _description_
    """
    insert = ax.inset_axes(loc)
    diff = y - x
    insert.hist(diff, bins=bins, **kwargs)
    insert.set_xlim(-np.max(abs(diff)), np.max(abs(diff)))
    insert.set_xlabel(xlabel)
    insert.set_ylabel("Count")
    return insert


def align_axes(axes, align_values):
    # keep format of first axes
    nTicks = len(axes[0].get_yticks())

    idx1 = (np.abs(axes[0].get_yticks() - align_values[0][0])).argmin()
    shiftAx1 = axes[0].get_yticks()[idx1] - align_values[0][0]
    ticksAx1 = axes[0].get_yticks() - shiftAx1
    dy1 = np.mean(
        [
            axes[0].get_yticks()[i] - axes[0].get_yticks()[i - 1]
            for i in range(1, len(axes[0].get_yticks()))
        ]
    )
    ylim1 = (ticksAx1[1] - dy1 / 2, ticksAx1[-2] + dy1 / 2)
    axes[0].set_yticks(ticksAx1)

    for i, ax in enumerate(axes[1:]):
        tmp = np.linspace(align_values[1][i + 1], align_values[0][i + 1], nTicks - 2)
        dy2 = np.mean([tmp[i] - tmp[i - 1] for i in range(1, len(tmp))])
        ticksAx2 = np.linspace(tmp[0] - dy2, tmp[-1] + dy2, nTicks)
        ylim2 = (ticksAx2[1] - dy2 / 2, ticksAx2[-2] + dy2 / 2)
        ax.set_yticks(ticksAx2)
        ax.set_ylim(ylim2)

    axes[0].set_ylim(ylim1)


def MolTo3DGrid(
    mols,
    show_labels=False,
    show_confs: bool = True,
    background_color=('blue', 0.1),
    export_HTML='none',
    cell_size=(400, 400),
    columns=3,
    linked=False,
    kekulize=True,
    legends=None,
    highlightAtoms=None,
    bonds_to_remove=None,
    show_charges=True,
):
    """
    Displays either:
    1) All conformers of a single RDKit molecule (or a SMILES string), or
    2) A list of RDKit molecules (each with one or more conformers),
    in a grid using py3Dmol. In addition to 3D rendering, this function can:

      - Overlay a legend label on each cell (with automatic numbering for
        multiple conformers).
      - Highlight specified atoms (via `highlightAtoms`).
      - Remove specified bonds before rendering (via `bonds_to_remove`).
      - Toggle per-atom labels on click.
      - Measure and label the distance between two atoms on Ctrl-click.

    Args:
        mols (rdkit.Chem.Mol or str or list[rdkit.Chem.Mol | str]):
            A single RDKit molecule, a SMILES string, an '.xyz' filepath,
            or a list of these. Each resulting molecule may have 0+ 3D
            conformers. If a molecule has no conformers, one will be embedded.
        show_labels (bool):
            If True, pre-draw atom labels (from `atomNote` or atom index).
            Clicking atoms toggles a label regardless.
        show_confs (bool):
            If True (default), display every conformer of each mol.
            If False, only the first conformer (confId=0) is shown.
        background_color (tuple[str, float]):
            (color, opacity) for the viewer background, e.g. ('white', 1.0).
        export_HTML (str):
            If not 'none', path used to write out an HTML file of the grid
            view.
        cell_size (tuple of int):
            (width, height) in pixels for each grid cell.
        columns (int):
            Number of columns; rows auto-computed.
        linked (bool):
            If True, link all cells for simultaneous rotation/zoom.
        kekulize (bool):
            If True, use Kekulé form when generating MolBlocks.
        legends (list of str):
            Legend text for each molecule; defaults to ["Mol 1", "Mol 2", …].
        highlightAtoms (list of int or list of list of int):
            Zero-based atom indices to highlight per molecule.
        bonds_to_remove (list of tuple of int):
            Pairs of atom indices whose bond should be removed before display.
            Applied to every molecule.
            e.g. [(10, 41), (10, 12), (11, 41)]
        show_charges (bool):
            Show charges in 3D space.

    Returns:
        None
    """
    import os
    import math
    from pathlib import Path
    import py3Dmol
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from rdkit.Chem.rdchem import RWMol

    try:
        from tooltoad.utils import chemutils as _chemutils
    except Exception:
        _chemutils = None

    # --- helper: coerce any RDKit Mol / '.xyz' path to RDKit Mol ----------------
    def _coerce_to_mol(item):
        if isinstance(item, Chem.Mol):
            return item

        if isinstance(item, (str, os.PathLike)):
            s = str(item)
            p = Path(s)

            # If it's an existing .xyz file, treat it as XYZ
            if p.suffix.lower() == ".xyz" and p.is_file():
                # Prefer tooltoad chemutils if available
                if _chemutils is not None:
                    try:
                        return _chemutils.read_xyz_file(
                            str(p), return_mol=True, useHueckel=True
                        )
                    except Exception:
                        pass

                # Fallback: raw RDKit + bond perception
                block = p.read_text()
                mol = Chem.MolFromXYZBlock(block)
                if mol is None:
                    raise ValueError(f"Could not parse XYZ file: {p}")
                try:
                    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=True)
                except Exception:
                    # If Hueckel fails, leave as-is; viewer can still show coords
                    pass
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                return mol

            # Otherwise: treat it as a SMILES string
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(
                    f"Could not parse SMILES string (and not an .xyz file): {s}"
                )
            return mol

        raise TypeError(
            "mols must be an RDKit Mol, a list of Mols, a SMILES string, "
            "a '.xyz' filepath, or a list of those."
        )

    # Wrap single input into list
    if not isinstance(mols, list):
        mols = [mols]

    # Coerce xyz paths → RDKit Mols (via tooltoad when available)
    mols = [_coerce_to_mol(m) for m in mols]

    # Normalize highlightAtoms into list of lists
    if highlightAtoms is None:
        normalized_highlights = None
    else:
        if all(isinstance(x, int) for x in highlightAtoms):
            normalized_highlights = [list(highlightAtoms)]
        else:
            try:
                normalized_highlights = [list(seq) for seq in highlightAtoms]
            except TypeError:
                raise ValueError(
                    "highlightAtoms must be a sequence of ints or sequence of "
                    "sequences."
                )
        if len(normalized_highlights) != len(mols):
            raise ValueError(
                "Length of highlightAtoms must match number of molecules."
            )

    # Prepare legends
    if legends is None:
        legends = []
    if not legends:
        legends = [f"Mol {i+1}" for i in range(len(mols))]
    if len(legends) != len(mols):
        raise ValueError(
            "Length of legends must match the number of molecules."
        )

    # Ensure 3D conformers and gather pairs
    mol_conf_pairs = []
    conf_counts = []
    mols_with_multiple_confs = False

    for i, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            AllChem.EmbedMolecule(mol, params)

        total_confs = mol.GetNumConformers()
        if show_confs:
            conf_list = list(range(total_confs))
            if total_confs > 1:
                mols_with_multiple_confs = True
        else:
            conf_list = [0]

        conf_counts.append(len(conf_list))

        for conf_id in conf_list:
            mol_conf_pairs.append((i, conf_id))

    # Compute grid dimensions
    if len(mols) == 1 and not mols_with_multiple_confs:
        columns = 1
    if len(mols) == 2 and not mols_with_multiple_confs:
        columns = 2
    total_pairs = len(mol_conf_pairs)
    rows = math.ceil(total_pairs / columns)
    total_width = cell_size[0] * columns
    total_height = cell_size[1] * rows

    # Create viewer
    viewer = py3Dmol.view(
        width=total_width, height=total_height,
        viewergrid=(rows, columns), linked=linked
    )
    viewer.setBackgroundColor(background_color[0], background_color[1])

    # Display each conformer
    for idx, (m_idx, conf_id) in enumerate(mol_conf_pairs):
        mol = mols[m_idx]
        row = idx // columns
        col = idx % columns

        # 1) copy & remove bonds
        mol_edit = RWMol(mol)
        if bonds_to_remove:
            for i_b, j_b in bonds_to_remove:
                mol_edit.RemoveBond(i_b, j_b)

        # 2) render the edited mol
        mol_block = Chem.MolToMolBlock(
            mol_edit, confId=conf_id, kekulize=kekulize
        )
        viewer.addModel(mol_block, 'mol', viewer=(row, col))
        viewer.setStyle(
            {}, {'stick': {}, 'sphere': {'radius': 0.3}}, viewer=(row, col)
        )
        viewer.zoomTo(viewer=(row, col))

        # Legend
        label = legends[m_idx]
        if conf_counts[m_idx] > 1:
            label += f" c{conf_id+1}"
        viewer.addLabel(
            label,
            {'fontColor': 'black', 'fontSize': 13, 'backgroundColor': 'white',
             'borderColor': 'black', 'borderWidth': 1, 'useScreen': True,
             'inFront': True, 'screenOffset': {'x': 10, 'y': 0}},
            viewer=(row, col)
        )

        # Per-atom labels
        if show_labels:
            conf = mol.GetConformer(conf_id)
            for atom in mol.GetAtoms():
                idx0 = atom.GetIdx()
                pos = conf.GetAtomPosition(idx0)
                text = (atom.GetProp('atomNote') if atom.HasProp('atomNote')
                        else str(idx0))
                viewer.addLabel(
                    text,
                    {'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                     'fontColor': 'black', 'backgroundColor': 'white',
                     'borderThickness': 1, 'fontSize': 12},
                    viewer=(row, col)
                )

        if show_charges:
            conf = mol.GetConformer(conf_id)
            for atom in mol.GetAtoms():
                fc = atom.GetFormalCharge()
                if fc == 0:
                    continue

                i_a = atom.GetIdx()
                pos = conf.GetAtomPosition(i_a)
                atom_pos = {'x': pos.x, 'y': pos.y, 'z': pos.z}
                label_pos = {'x': pos.x, 'y': pos.y, 'z': pos.z}

                color = 'red' if fc > 0 else 'blue'
                sign = f"{abs(fc)}+" if fc > 0 else f"{abs(fc)}-"

                viewer.addCylinder(
                    {'start': atom_pos, 'end': label_pos, 'radius': 0.05,
                     'color': color, 'fromCap': False, 'toCap': False},
                    viewer=(row, col)
                )

                viewer.addLabel(
                    sign,
                    {'position': label_pos, 'inFront': True, 'fontSize': 16,
                     'fontColor': color, 'fontWeight': 'bold',
                     'backgroundColor': 'rgba(255,255,255,0)',
                     'backgroundOpacity': 0.6},
                    viewer=(row, col)
                )

        viewer.setClickable(
            {}, True,
            '''function(atom, viewer, event, container) {
                if(!viewer._picks)       viewer._picks = [];
                if(!viewer._distLabels)  viewer._distLabels = {};
                if(!viewer._anglePicks)  viewer._anglePicks = [];
                if(!viewer._angleLabels) viewer._angleLabels = {};

                // Shift-click: angle
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
                            function vec(u,v){return{x:u.x-v.x,y:u.y-v.y,z:u.z-v.z}};
                            var vBA=vec(A,B), vBC=vec(C,B);
                            var dot=vBA.x*vBC.x+vBA.y*vBC.y+vBA.z*vBC.z;
                            var magBA=Math.sqrt(vBA.x*vBA.x+vBA.y*vBA.y+vBA.z*vBA.z);
                            var magBC=Math.sqrt(vBC.x*vBC.x+vBC.y*vBC.y+vBC.z*vBC.z);
                            var angle=(Math.acos(dot/(magBA*magBC))*(180/Math.PI))
                                      .toFixed(2)+'°';
                            var lbl = viewer.addLabel(angle,
                                {position:{x:B.x,y:B.y,z:B.z},
                                 backgroundColor:'blue', fontColor:'white',
                                 fontSize:12});
                            viewer._angleLabels[key] = lbl;
                        }
                        viewer._anglePicks = [];
                    }
                }
                // Ctrl-click: distance
                else if(event.ctrlKey) {
                    viewer._picks.push(atom);
                    if(viewer._picks.length === 2) {
                        var a=viewer._picks[0], b=viewer._picks[1];
                        var key=[Math.min(a.index,b.index),
                                 Math.max(a.index,b.index)].join('-');
                        if(key in viewer._distLabels) {
                            viewer.removeLabel(viewer._distLabels[key]);
                            delete viewer._distLabels[key];
                        } else {
                            var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
                            var dist=Math.sqrt(dx*dx+dy*dy+dz*dz)
                                      .toFixed(3)+' Å';
                            var mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:(a.z+b.z)/2};
                            var lbl=viewer.addLabel(dist,
                                {position:mid,backgroundColor:'grey',
                                 fontColor:'white',fontSize:12});
                            viewer._distLabels[key]=lbl;
                        }
                        viewer._picks = [];
                    }
                }
                // Click: toggle label
                else {
                    if(atom.label) {
                        viewer.removeLabel(atom.label); delete atom.label;
                    } else {
                        atom.label = viewer.addLabel(
                            atom.index,
                            {position:atom, backgroundColor:'white',
                             fontColor:'black', fontSize:12}
                        );
                    }
                }
                viewer.render();
            }'''
        )

        # Highlight atoms if requested
        if normalized_highlights is not None:
            atoms_sel = normalized_highlights[m_idx]
            viewer.setStyle(
                {'serial': atoms_sel},
                {'stick': {'radius': 0.2, 'color': 'red'},
                 'sphere': {'radius': 0.4, 'color': 'red'}},
                viewer=(row, col)
            )

    # Export HTML if requested
    if export_HTML != 'none':
        try:
            os.makedirs(os.path.dirname(export_HTML), exist_ok=True)
            viewer.write_html(export_HTML)
            print(f"HTML export successful: {export_HTML}")
        except Exception as e:
            print(f"Error exporting HTML to '{export_HTML}': {e}")

    viewer.show()


def RxnTo3DGrid(
    rxn,
    show_labels: bool = False,
    background_color: tuple = ('blue', 0.1),
    export_HTML: str = 'none',
    cell_size=(400, 400),
    linked: bool = False,
    kekulize: bool = True,
    show_charges: bool = True,
    legends=None,
    show_bond_changes: bool = False,
    h_mode: str = "reactive",          # "none" | "reactive" | "all"
    show_charge_changes: bool = False,
    check_reaction_stoichiometry: bool = False
):
    """
    Displays a reaction in 3D using py3Dmol.

    Layout:
        [ Reactants ]   [ 3D arrow ]   [ Products ]

    Args:
        rxn:
            RDKit ChemicalReaction or a reaction SMILES/SMARTS string.
            Strings are parsed with ReactionFromSmarts(..., useSmiles=True).
        show_labels:
            If True, pre-draw atom labels (index or atomNote) and allow
            click-toggle for labels.
        background_color:
            Background color for the viewer (e.g. 'white', 'black').
        export_HTML:
            If not 'none', path used to write out an HTML file of the grid
            view.
        cell_size:
            (width, height) in pixels for each of the three cells.
        linked:
            If True, link cells for simultaneous rotation/zoom.
        kekulize:
            If True, use Kekulé form when generating MolBlocks.
        show_charges:
            Show formal charges in 3D space (per-atom annotation).
        legends:
            Optional two labels: [reactant_label, product_label].
            Defaults to ["Reactants", "Products"].
        show_bond_changes:
            If True and the reaction is atom-mapped, highlight bonds broken
            (red), formed (green), and with changed bond order (orange).
        h_mode:
            "none"      → hide all hydrogens (even explicit ones).
            "reactive"  → show only the hydrogens explicitly present in the
                          reaction (typically reactive/proton-transfer H).
            "all"       → Add explicit hydrogens to all molecules and show
                          them.
        show_charge_changes:
            If True and the reaction is atom-mapped, highlight atoms whose
            formal charge changes between reactants and products (red for
            more positive, blue for more negative).
        check_reaction_stoichiometry:
            If True a summary of the stoichiometry will be printed.
    """
    import os
    from pathlib import Path

    import py3Dmol
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdChemReactions, rdDepictor
    from rdkit.Chem.rdchem import RWMol
    from rdkit.Geometry import Point3D
    from collections import defaultdict, Counter

    try:
        from tooltoad.utils import chemutils as _chemutils
    except Exception:
        _chemutils = None

    h_mode = h_mode.lower()
    if h_mode not in ("none", "reactive", "all"):
        raise ValueError(
            "h_mode must be one of 'none', 'reactive', or 'all'."
        )

    # --- helpers -------------------------------------------------------------
    def _coerce_to_mol(item):
        """Coerce RDKit Mol / xyz path to RDKit Mol (similar to MolTo3DGrid)."""
        if isinstance(item, Chem.Mol):
            return item
        if isinstance(item, (str, os.PathLike)):
            p = Path(item)
            if p.suffix.lower() == ".xyz" and p.is_file():
                if _chemutils is not None:
                    try:
                        return _chemutils.read_xyz_file(
                            str(p), return_mol=True, useHueckel=True
                        )
                    except Exception:
                        pass
                block = p.read_text()
                mol = Chem.MolFromXYZBlock(block)
                if mol is None:
                    raise ValueError(f"Could not parse XYZ file: {p}")
                try:
                    from rdkit.Chem import rdDetermineBonds

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
        raise TypeError(
            "rxn must be a ChemicalReaction, a reaction SMILES/SMARTS string, "
            "or xyz-based molecule definitions."
        )

    def _ensure_3d(mol):
        """Ensure mol has at least one conformer, suppressing RDKit log spam."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDepictor

        if mol.GetNumConformers() > 0:
            return

        # Temporarily suppress RDKit logs (all rdApp.*)
        with rdBase.BlockLogs():
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 0xF00D
                AllChem.EmbedMolecule(mol, params)
                if mol.GetNumConformers() > 0:
                    return
            except Exception:
                pass

            # 2D fallback
            try:
                rdDepictor.Compute2DCoords(mol)
            except Exception:
                pass

    def _offset_mol(mol, dx):
        mol = RWMol(mol)
        if mol.GetNumConformers() == 0:
            _ensure_3d(mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x + dx, pos.y, pos.z))
        return mol

    def _prepare_side(mols, gap: float = 3.0):
        """Return 3D mols laid out along x without overlapping (bbox-based)."""
        prepared = []
        bounds = []
        current_x = 0.0

        for m in mols:
            m = _coerce_to_mol(Chem.Mol(m))
            _ensure_3d(m)

            conf = m.GetConformer()
            xs = [conf.GetAtomPosition(i).x for i in range(m.GetNumAtoms())]
            if not xs:
                prepared.append(m)
                continue
            xmin, xmax = min(xs), max(xs)

            dx = current_x - xmin
            m_off = _offset_mol(m, dx)
            xmin_off = xmin + dx
            xmax_off = xmax + dx
            bounds.append((xmin_off, xmax_off))
            prepared.append(m_off)

            current_x = xmax_off + gap

        if not prepared or not bounds:
            return prepared

        all_min = min(b[0] for b in bounds)
        all_max = max(b[1] for b in bounds)
        center = 0.5 * (all_min + all_max)

        for m in prepared:
            conf = m.GetConformer()
            for i in range(m.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(
                    i, Point3D(pos.x - center, pos.y, pos.z)
                )

        return prepared

    def _add_side_to_viewer(side_mols, viewer, grid_pos, label):
        """Add all mols for one side (reactants or products) to a given cell."""
        row, col = grid_pos
        for m in side_mols:
            block = Chem.MolToMolBlock(m, kekulize=kekulize)
            viewer.addModel(block, 'mol', viewer=(row, col))

        viewer.setStyle(
            {}, {'stick': {}, 'sphere': {'radius': 0.3}}, viewer=(row, col)
        )

        # Hydrogen visibility control
        if h_mode == "none":
            viewer.setStyle({'elem': 'H'}, {}, viewer=(row, col))

        viewer.zoomTo(viewer=(row, col))

        if label:
            viewer.addLabel(
                label,
                {
                    'fontColor': 'black',
                    'backgroundColor': 'white',
                    'borderColor': 'black',
                    'borderWidth': 1,
                    'useScreen': True,
                    'inFront': True,
                    'screenOffset': {'x': 10, 'y': 0},
                },
                viewer=(row, col),
            )

        if show_labels or show_charges:
            for m in side_mols:
                conf = m.GetConformer()
                for atom in m.GetAtoms():
                    if h_mode == "none" and atom.GetSymbol() == "H":
                        continue

                    idx0 = atom.GetIdx()
                    pos = conf.GetAtomPosition(idx0)
                    apos = {'x': pos.x, 'y': pos.y, 'z': pos.z}

                    if show_labels:
                        text = (
                            atom.GetProp('atomNote')
                            if atom.HasProp('atomNote')
                            else str(idx0)
                        )
                        viewer.addLabel(
                            text,
                            {
                                'position': apos,
                                'fontColor': 'black',
                                'backgroundColor': 'white',
                                'borderThickness': 1,
                                'fontSize': 12,
                            },
                            viewer=(row, col),
                        )

                    if show_charges:
                        fc = atom.GetFormalCharge()
                        if fc != 0:
                            color = 'red' if fc > 0 else 'blue'
                            sign = (
                                f"{abs(fc)}+"
                                if fc > 0
                                else f"{abs(fc)}-"
                            )
                            viewer.addLabel(
                                sign,
                                {
                                    'position': apos,
                                    'inFront': True,
                                    'fontSize': 16,
                                    'fontColor': color,
                                    'fontWeight': 'bold',
                                    'backgroundColor': 'rgba(255,255,255,0)',
                                    'backgroundOpacity': 0.6,
                                },
                                viewer=(row, col),
                            )

    def _collect_mapped_bonds(mol_list):
        """Return dict[(map1,map2)] -> bondType for mapped bonds."""
        bonds = {}
        for m in mol_list:
            for b in m.GetBonds():
                a1 = b.GetBeginAtom()
                a2 = b.GetEndAtom()
                if (
                    a1.HasProp('molAtomMapNumber')
                    and a2.HasProp('molAtomMapNumber')
                ):
                    m1 = a1.GetIntProp('molAtomMapNumber')
                    m2 = a2.GetIntProp('molAtomMapNumber')
                    key = tuple(sorted((m1, m2)))
                    bonds[key] = b.GetBondType()
        return bonds

    def _build_mapnum_index(side_mols):
        """mapNum -> (mol_idx, atom_idx) for displayed side mols."""
        mapping = {}
        for m_idx, m in enumerate(side_mols):
            for a in m.GetAtoms():
                if a.HasProp('molAtomMapNumber'):
                    mnum = a.GetIntProp('molAtomMapNumber')
                    mapping[mnum] = (m_idx, a.GetIdx())
        return mapping

    def _build_mapnum_symbol(side_mols):
        """mapNum -> atomic symbol."""
        mapping = {}
        for m in side_mols:
            for a in m.GetAtoms():
                if a.HasProp('molAtomMapNumber'):
                    mnum = a.GetIntProp('molAtomMapNumber')
                    mapping[mnum] = a.GetSymbol()
        return mapping

    def _add_bond_change_cylinders(
        pairs, side_mols, map_index, color, viewer, grid_pos
    ):
        """Overlay cylinders for changed bonds."""
        row, col = grid_pos
        for m1, m2 in pairs:
            info1 = map_index.get(m1)
            info2 = map_index.get(m2)
            if info1 is None or info2 is None:
                continue
            m_idx1, a_idx1 = info1
            m_idx2, a_idx2 = info2

            m1_mol = side_mols[m_idx1]
            m2_mol = side_mols[m_idx2]
            conf1 = m1_mol.GetConformer()
            conf2 = m2_mol.GetConformer()
            p1 = conf1.GetAtomPosition(a_idx1)
            p2 = conf2.GetAtomPosition(a_idx2)

            start = {'x': p1.x, 'y': p1.y, 'z': p1.z}
            end = {'x': p2.x, 'y': p2.y, 'z': p2.z}

            viewer.addCylinder(
                {
                    'start': start,
                    'end': end,
                    'radius': 0.18,
                    'color': color,
                },
                viewer=(row, col),
            )

    def _add_charge_change_spheres(
        side_mols, map_index, charge_delta, viewer, grid_pos
    ):
        """Overlay spheres at atoms whose formal charge changes."""
        row, col = grid_pos
        for mnum, delta in charge_delta.items():
            info = map_index.get(mnum)
            if info is None:
                continue
            m_idx, a_idx = info
            m = side_mols[m_idx]
            atom = m.GetAtomWithIdx(a_idx)
            if h_mode == "none" and atom.GetSymbol() == "H":
                continue

            conf = m.GetConformer()
            pos = conf.GetAtomPosition(a_idx)
            color = 'red' if delta > 0 else 'blue'
            viewer.addSphere(
                {
                    'center': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                    'radius': 0.35,
                    'color': color,
                    'alpha': 0.7,
                },
                viewer=(row, col),
            )

    def _add_hs_safe(mol_in):
        """Best-effort AddHs that won't crash on weird valence/query atoms."""
        mol = Chem.Mol(mol_in)  # work on a copy

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        try:
            mh = Chem.AddHs(mol)
        except Exception:
            # If AddHs fails, just return the sanitized (or original) mol
            mh = mol

        return mh

    def _check_reaction_stoichiometry(reactants, products, verbose: bool = True):
        """Internal stoichiometry check on lists of RDKit mols."""
        def _side_counts(mols):
            elem = Counter()
            total_charge = 0
            for m in mols:
                mol = Chem.Mol(m)  # work on a copy

                # try to sanitize, but don't die if it fails
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass

                # try to add explicit H; if that fails, just use mol as-is
                try:
                    mh = Chem.AddHs(mol)
                except Exception:
                    mh = mol

                for atom in mh.GetAtoms():
                    elem[atom.GetSymbol()] += 1
                    total_charge += atom.GetFormalCharge()

            return elem, total_charge

        r_counts, r_charge = _side_counts(reactants)
        p_counts, p_charge = _side_counts(products)

        elements = sorted(set(r_counts) | set(p_counts))
        element_diffs = {}
        atoms_balanced = True
        for el in elements:
            r_n = r_counts.get(el, 0)
            p_n = p_counts.get(el, 0)
            if r_n != p_n:
                atoms_balanced = False
            element_diffs[el] = (r_n, p_n, p_n - r_n)

        charges_balanced = (r_charge == p_charge)

        result = {
            "atoms_balanced": atoms_balanced,
            "charges_balanced": charges_balanced,
            "reactant_counts": r_counts,
            "product_counts": p_counts,
            "element_diffs": element_diffs,
            "reactant_charge": r_charge,
            "product_charge": p_charge,
        }

        if verbose:
            print("=== Stoichiometry check (RxnTo3DGrid) ===")
            for el in elements:
                r_n, p_n, delta = element_diffs[el]
                print(
                    f"{el:>2}: reactants={r_n:3d}  products={p_n:3d}  "
                    f"Δ={delta:+d}"
                )
            print(
                f"\nTotal charge: reactants={r_charge:+d}, "
                f"products={p_charge:+d}"
            )
            print(f"\nAtoms balanced:   {atoms_balanced}")
            print(f"Charges balanced: {charges_balanced}\n")

        return result        

    # --- coerce rxn ----------------------------------------------------------
    if isinstance(rxn, (str, os.PathLike)):
        rxn_str = str(rxn)
        rxn_obj = rdChemReactions.ReactionFromSmarts(
            rxn_str, useSmiles=True
        )
        if rxn_obj is None:
            raise ValueError(
                "Could not parse reaction string. Make sure it is a valid "
                "reaction SMILES/SMARTS."
            )
    elif hasattr(rxn, "GetReactants") and hasattr(rxn, "GetProducts"):
        rxn_obj = rxn
    else:
        raise TypeError(
            "rxn must be a ChemicalReaction or a reaction SMILES/SMARTS "
            "string."
        )

    reactant_mols = [Chem.Mol(m) for m in rxn_obj.GetReactants()]
    product_mols = [Chem.Mol(m) for m in rxn_obj.GetProducts()]

    # --- stoichiometry check (before any AddHs / coordinate tweaks) -------
    balance_label = None
    balance_color = "black"

    if check_reaction_stoichiometry:
        stoich_result = _check_reaction_stoichiometry(
            reactant_mols, product_mols, verbose=True  # or False if you hate prints
        )

        atoms_ok = stoich_result["atoms_balanced"]
        charge_ok = stoich_result["charges_balanced"]
    else:
        stoich_result = _check_reaction_stoichiometry(
            reactant_mols, product_mols, verbose=False  # or False if you hate prints
        )
        
        atoms_ok = stoich_result["atoms_balanced"]
        charge_ok = stoich_result["charges_balanced"]

    if atoms_ok and charge_ok:
        balance_label = "atoms ✓, charge ✓"
        balance_color = "green"
    elif atoms_ok and not charge_ok:
        balance_label = "atoms ✓, charge ✗"
        balance_color = "orange"
    elif not atoms_ok and charge_ok:
        balance_label = "atoms ✗, charge ✓"
        balance_color = "orange"
    else:
        balance_label = "atoms ✗, charge ✗"
        balance_color = "red"

    # hydrogen handling at molecule level
    if h_mode == "all":
        reactant_mols = [_add_hs_safe(m) for m in reactant_mols]
        product_mols = [_add_hs_safe(m) for m in product_mols]
    # "reactive" and "none" keep only explicitly present H (typically
    # reactive ones if they are written in the reaction SMILES).

    # --- bond-change and charge-change analysis ------------------------------
    broken_bonds = set()
    formed_bonds = set()
    changed_bonds = set()
    have_mapping = False

    charge_delta = {}  # mapNum -> (p_charge - r_charge)

    if show_bond_changes or show_charge_changes:
        # bond changes
        rb = _collect_mapped_bonds(reactant_mols)
        pb = _collect_mapped_bonds(product_mols)
        if rb or pb:
            have_mapping = True
            r_keys = set(rb.keys())
            p_keys = set(pb.keys())
            broken_bonds = r_keys - p_keys
            formed_bonds = p_keys - r_keys
            common = r_keys & p_keys
            changed_bonds = {k for k in common if rb[k] != pb[k]}

        # charge changes
        r_charge_map = defaultdict(int)
        p_charge_map = defaultdict(int)
        for m in reactant_mols:
            for a in m.GetAtoms():
                if a.HasProp('molAtomMapNumber'):
                    mnum = a.GetIntProp('molAtomMapNumber')
                    r_charge_map[mnum] = a.GetFormalCharge()
        for m in product_mols:
            for a in m.GetAtoms():
                if a.HasProp('molAtomMapNumber'):
                    mnum = a.GetIntProp('molAtomMapNumber')
                    p_charge_map[mnum] = a.GetFormalCharge()

        all_maps = set(r_charge_map) | set(p_charge_map)
        for mnum in all_maps:
            rc = r_charge_map.get(mnum, 0)
            pc = p_charge_map.get(mnum, 0)
            if rc != pc:
                charge_delta[mnum] = pc - rc

        if (show_bond_changes or show_charge_changes) and not have_mapping:
            print(
                "RxnTo3DGrid: mapping not found; bond/charge change "
                "highlighting limited."
            )

    # --- prepare 3D mols for each side --------------------------------------
    left_side = _prepare_side(reactant_mols, gap=4.0)
    right_side = _prepare_side(product_mols, gap=4.0)

    if legends is None:
        legends = ["Reactants", "Products"]
    if len(legends) != 2:
        raise ValueError("legends must be a list of two strings.")

    # --- build viewer (1 row, 3 columns) ------------------------------------
    cols = 3
    rows = 1
    total_width = cell_size[0] * cols
    total_height = cell_size[1] * rows

    viewer = py3Dmol.view(
        width=total_width,
        height=total_height,
        viewergrid=(rows, cols),
        linked=linked,
    )
    viewer.setBackgroundColor(background_color[0], background_color[1])

    # left: reactants
    _add_side_to_viewer(left_side, viewer, (0, 0), legends[0])

    # middle: arrow
    viewer.addArrow(
        {
            'start': {'x': -1.5, 'y': 0.0, 'z': 0.0},
            'end': {'x': 1.5, 'y': 0.0, 'z': 0.0},
            'radius': 0.15,
            'color': 'black',
        },
        viewer=(0, 1),
    )
    viewer.zoomTo(viewer=(0, 1))

    if balance_label is not None:
        viewer.addLabel(
            balance_label,
            {
                'fontColor': balance_color,
                'backgroundColor': 'white',
                'borderColor': balance_color,
                'borderWidth': 1,
                'useScreen': True,
                'inFront': True,
                # negative y to place label slightly above the arrow
                'screenOffset': {'x': 120, 'y': -125},
            },
            viewer=(0, 1),
        )    

    # right: products
    _add_side_to_viewer(right_side, viewer, (0, 2), legends[1])

    # --- overlay bond changes -----------------------------------------------
    if show_bond_changes and have_mapping:
        left_index = _build_mapnum_index(left_side)
        right_index = _build_mapnum_index(right_side)

        _add_bond_change_cylinders(
            broken_bonds, left_side, left_index, 'red', viewer, (0, 0)
        )
        _add_bond_change_cylinders(
            changed_bonds, left_side, left_index, 'orange', viewer, (0, 0)
        )

        _add_bond_change_cylinders(
            formed_bonds, right_side, right_index, 'green', viewer, (0, 2)
        )
        _add_bond_change_cylinders(
            changed_bonds, right_side, right_index, 'orange', viewer, (0, 2)
        )

    # --- overlay charge changes ---------------------------------------------
    if show_charge_changes and charge_delta:
        left_index = _build_mapnum_index(left_side)
        right_index = _build_mapnum_index(right_side)
        _add_charge_change_spheres(
            left_side, left_index, charge_delta, viewer, (0, 0)
        )
        _add_charge_change_spheres(
            right_side, right_index, charge_delta, viewer, (0, 2)
        )

    # --- clickable distances / angles / labels ------------------------------
    viewer.setClickable(
        {}, True,
        '''function(atom, viewer, event, container) {
            if(!viewer._picks)       viewer._picks = [];
            if(!viewer._distLabels)  viewer._distLabels = {};
            if(!viewer._anglePicks)  viewer._anglePicks = [];
            if(!viewer._angleLabels) viewer._angleLabels = {};

            // Shift-click: angle
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
                        function vec(u,v){return{x:u.x-v.x,y:u.y-v.y,z:u.z-v.z}};
                        var vBA=vec(A,B), vBC=vec(C,B);
                        var dot=vBA.x*vBC.x+vBA.y*vBC.y+vBA.z*vBC.z;
                        var magBA=Math.sqrt(
                            vBA.x*vBA.x+vBA.y*vBA.y+vBA.z*vBA.z
                        );
                        var magBC=Math.sqrt(
                            vBC.x*vBC.x+vBC.y*vBC.y+vBC.z*vBC.z
                        );
                        var angle = (
                            Math.acos(dot/(magBA*magBC))*(180/Math.PI)
                        ).toFixed(2)+'°';
                        var lbl = viewer.addLabel(angle,
                            {position:{x:B.x,y:B.y,z:B.z},
                             backgroundColor:'blue', fontColor:'white',
                             fontSize:12});
                        viewer._angleLabels[key] = lbl;
                    }
                    viewer._anglePicks = [];
                }
            }
            // Ctrl-click: distance
            else if(event.ctrlKey) {
                viewer._picks.push(atom);
                if(viewer._picks.length === 2) {
                    var a=viewer._picks[0], b=viewer._picks[1];
                    var key=[Math.min(a.index,b.index),
                             Math.max(a.index,b.index)].join('-');
                    if(key in viewer._distLabels) {
                        viewer.removeLabel(viewer._distLabels[key]);
                        delete viewer._distLabels[key];
                    } else {
                        var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
                        var dist=Math.sqrt(dx*dx+dy*dy+dz*dz)
                                  .toFixed(3)+' Å';
                        var mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:(a.z+b.z)/2};
                        var lbl=viewer.addLabel(dist,
                            {position:mid,backgroundColor:'grey',
                             fontColor:'white',fontSize:12});
                        viewer._distLabels[key]=lbl;
                    }
                    viewer._picks = [];
                }
            }
            // Click: toggle label
            else {
                if(atom.label) {
                    viewer.removeLabel(atom.label); delete atom.label;
                } else {
                    atom.label = viewer.addLabel(
                        atom.index,
                        {position:atom, backgroundColor:'white',
                         fontColor:'black', fontSize:12}
                    );
                }
            }
            viewer.render();
        }'''
    )

    # --- export --------------------------------------------------------------
    if export_HTML != 'none':
        try:
            os.makedirs(os.path.dirname(export_HTML), exist_ok=True)
            viewer.write_html(export_HTML)
            print(f"HTML export successful: {export_HTML}")
        except Exception as e:
            print(f"Error exporting HTML to '{export_HTML}': {e}")

    viewer.show()