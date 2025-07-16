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
    fps: float | None = None
):
    """Show normal mode vibration."""
    input = results
    vib = input.get("vibs")[vId]
    mode = vib["mode"]
    frequency = vib["frequency"]
    atoms = results["atoms"]
    opt_coords = results["opt_coords"]
    xyz = ac2xyz(atoms, opt_coords)

    if fps is None:
        interval_ms = 50
    else:
        interval_ms = max(1, int(1000.0 / fps))

    p = py3Dmol.view(width=width, height=height)
    p.addModel(xyz, "xyz")
    propmap = []
    for j, m in enumerate(mode):
        propmap.append(
            {
                "index": j,
                "props": {
                    "dx": m[0],
                    "dy": m[1],
                    "dz": m[2],
                },
            }
        )
    p.mapAtomProperties(propmap)
    p.vibrate(numFrames, amplitude, True)
    p.animate({"loop": "backAndForth", "interval": interval_ms, "reps": 0})
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    p.setBackgroundColor("0xeeeeee", int(~transparent))
    p.zoomTo()
    print(f"Normal mode {vId} with frequency {frequency} cm^-1")
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
    background_color='white',
    export_HTML='none',
    cell_size=(400, 400),
    columns=3,
    linked=False,
    kekulize=True,
    legends=None,
    highlightAtoms=None,
    bonds_to_remove=None,
):
    """
    Displays either:
    1) All conformers of a single RDKit molecule, or
    2) A list of RDKit molecules (each with one or more conformers),
    in a grid using py3Dmol. In addition to 3D rendering, this function can:

      - Overlay a legend label on each cell (with automatic numbering for multiple conformers).
      - Highlight specified atoms (via `highlightAtoms`).
      - Remove specified bonds before rendering (via `bonds_to_remove`).
      - Toggle per-atom labels on click.
      - Measure and label the distance between two atoms on Ctrl-click.

    Args:
        mols (rdkit.Chem.Mol or list of rdkit.Chem.Mol):
            A single RDKit molecule or a list of molecules, each with 0+ 3D conformers.
        show_labels (bool):
            If True, pre-draw atom labels (from `atomNote` or atom index) and enable click-toggle.
            If False, only click-toggle is enabled.
        show_confs (bool):
            If True (default), display every conformer of each mol.
            If False, only the first conformer (confId=0) is shown.
        background_color (str):
            Background color for the viewer (e.g. 'white', 'black').
        export_HTML (str):
            If not 'none', path used to write out an HTML file of the grid view.
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
            e.g. [(10, 41), (10, 12), (11, 41)]

    Returns:
        None
    """
    import os
    import math
    import py3Dmol
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import RWMol

    # Wrap single molecule into list
    if not isinstance(mols, list):
        mols = [mols]

    # Normalize highlightAtoms into list of lists
    # Accept: None, flat list/tuple, or sequence of sequences
    if highlightAtoms is None:
        normalized_highlights = None
    else:
        # detect flat sequence of ints
        if all(isinstance(x, int) for x in highlightAtoms):
            normalized_highlights = [list(highlightAtoms)]
        else:
            # assume sequence of sequences
            try:
                normalized_highlights = [list(seq) for seq in highlightAtoms]
            except TypeError:
                raise ValueError("highlightAtoms must be a sequence of ints or sequence of sequences.")
        if len(normalized_highlights) != len(mols):
            raise ValueError("Length of highlightAtoms must match number of molecules.")

    # Prepare legends
    if legends is None:
        legends = []
    if not legends:
        legends = [f"Mol {i+1}" for i in range(len(mols))]
    if len(legends) != len(mols):
        raise ValueError("Length of legends must match the number of molecules.")

    # Ensure 3D conformers and gather pairs
    mol_conf_pairs = []
    conf_counts = []
    mols_with_multiple_confs = False

    for i, mol in enumerate(mols):
        # ensure at least one 3D conf
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            AllChem.EmbedMolecule(mol, params)

        total_confs = mol.GetNumConformers()
        # decide which confs to display
        if show_confs:
            conf_list = list(range(total_confs))
            if total_confs > 1:
                mols_with_multiple_confs = True
        else:
            # only show the first conformer
            conf_list = [0]

        # record how many *we are actually* displaying
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
    viewer = py3Dmol.view(width=total_width, height=total_height, viewergrid=(rows, columns), linked=linked)
    viewer.setBackgroundColor(background_color)

    # Display each conformer
    for idx, (m_idx, conf_id) in enumerate(mol_conf_pairs):
        mol = mols[m_idx]
        row = idx // columns
        col = idx % columns

        # 1) copy & remove bonds
        mol_edit = RWMol(mol)
        if bonds_to_remove:
            for i, j in bonds_to_remove:
                mol_edit.RemoveBond(i, j)

        # 2) render the edited mol
        mol_block = Chem.MolToMolBlock(mol_edit, confId=conf_id, kekulize=kekulize)
        viewer.addModel(mol_block, 'mol', viewer=(row, col))
        viewer.setStyle({}, {'stick': {}, 'sphere': {'radius': 0.3}}, viewer=(row, col))
        viewer.zoomTo(viewer=(row, col))

        # Legend
        label = legends[m_idx]
        if conf_counts[m_idx] > 1:
            label += f" c{conf_id+1}"
        viewer.addLabel(label,
            {'fontColor':'black', 'backgroundColor':'white', 'borderColor':'black', 'borderWidth':1,
             'useScreen':True, 'inFront':True, 'screenOffset':{'x':10,'y':0}},
            viewer=(row, col)
        )

        # Per-atom labels
        if show_labels:
            conf = mol.GetConformer(conf_id)
            for atom in mol.GetAtoms():
                idx0 = atom.GetIdx()
                pos = conf.GetAtomPosition(idx0)
                text = atom.GetProp('atomNote') if atom.HasProp('atomNote') else str(idx0)
                viewer.addLabel(text,
                    {'position':{'x':pos.x,'y':pos.y,'z':pos.z}, 'fontColor':'black', 'backgroundColor':'white',
                     'borderThickness':1, 'fontSize':12},
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
                            var A = viewer._anglePicks[0], B = viewer._anglePicks[1], C = viewer._anglePicks[2];
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
                                var angle=(Math.acos(dot/(magBA*magBC))*(180/Math.PI)).toFixed(2)+'°';
                                var lbl = viewer.addLabel(angle,
                                    {position:{x:B.x,y:B.y,z:B.z}, backgroundColor:'blue', fontColor:'white', fontSize:12});
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
                            var key=[Math.min(a.index,b.index),Math.max(a.index,b.index)].join('-');
                            if(key in viewer._distLabels) {
                                viewer.removeLabel(viewer._distLabels[key]); delete viewer._distLabels[key];
                            } else {
                                var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
                                var dist=Math.sqrt(dx*dx+dy*dy+dz*dz).toFixed(3)+' Å';
                                var mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:(a.z+b.z)/2};
                                var lbl=viewer.addLabel(dist,{position:mid,backgroundColor:'grey',fontColor:'white',fontSize:12});
                                viewer._distLabels[key]=lbl;
                            }
                            viewer._picks = [];
                        }
                    }
                    // Click: toggle label
                    else {
                        if(atom.label) {viewer.removeLabel(atom.label); delete atom.label;} 
                        else {atom.label = viewer.addLabel(atom.index,{position:atom, backgroundColor:'white', fontColor:'black', fontSize:12});}
                    }
                    viewer.render();
                }'''
            )

        # Highlight atoms if requested
        if highlightAtoms is not None:
            atoms = normalized_highlights[m_idx]
            viewer.setStyle(
                {'serial': atoms},
                {'stick':{'radius':0.2,'color':'red'}, 'sphere':{'radius':0.4,'color':'red'}},
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