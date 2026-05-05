from __future__ import annotations

from pathlib import Path
from typing import Iterable


ANGSTROM_TO_BOHR = 1.8897259886


ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "I": 126.904,
}


def read_xtb_hessian(path: str | Path, natoms: int) -> list[list[float]]:
    """Read xTB/g-xTB Cartesian Hessian matrix from the plain ``hessian`` file."""
    path = Path(path)
    values: list[float] = []
    in_block = False
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("$hessian"):
                in_block = True
                continue
            if in_block and stripped.startswith("$"):
                break
            if in_block:
                values.extend(float(field.replace("D", "E")) for field in stripped.split())

    ndim = 3 * natoms
    expected = ndim * ndim
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} Hessian values for {natoms} atoms, got {len(values)}."
        )
    return [values[i * ndim : (i + 1) * ndim] for i in range(ndim)]


def _format_matrix_block(matrix: list[list[float]]) -> str:
    nrows = len(matrix)
    lines: list[str] = []
    for start in range(0, nrows, 5):
        cols = list(range(start, min(start + 5, nrows)))
        lines.append("          " + "".join(f"{col:18d}" for col in cols))
        for row_idx, row in enumerate(matrix):
            vals = "".join(f"{row[col]:18.10E}" for col in cols)
            lines.append(f"{row_idx:5d} {vals}")
    return "\n".join(lines)


def _atom_lines(atoms: Iterable[str], coords: Iterable[Iterable[float]]) -> list[str]:
    lines: list[str] = []
    for atom, coord in zip(atoms, coords):
        mass = ATOMIC_MASSES.get(atom)
        if mass is None:
            raise ValueError(f"No atomic mass configured for element {atom!r}.")
        x, y, z = [float(c) * ANGSTROM_TO_BOHR for c in coord]
        lines.append(f" {atom:<2} {mass:11.5f} {x:18.12f} {y:18.12f} {z:18.12f}")
    return lines


def write_orca_hessian_string(
    *,
    atoms: list[str],
    coords: list[list[float]],
    hessian: list[list[float]],
    energy: float = 0.0,
    multiplicity: int = 1,
) -> str:
    """Write the minimal text ORCA ``.hess`` format accepted by ``InHess Read``."""
    natoms = len(atoms)
    ndim = 3 * natoms
    if len(coords) != natoms:
        raise ValueError("Number of coordinate rows must match number of atoms.")
    if len(hessian) != ndim or any(len(row) != ndim for row in hessian):
        raise ValueError(f"Hessian must be a {ndim}x{ndim} matrix.")

    lines = [
        "$orca_hessian_file",
        "",
        "$act_atom",
        "  0",
        "",
        "$act_coord",
        "  0",
        "",
        "$act_energy",
        f" {energy:15.6f}",
        "",
        "$multiplicity",
        f"  {multiplicity}",
        "",
        "$hessian",
        str(ndim),
        _format_matrix_block(hessian),
        "",
        "$atoms",
        str(natoms),
        *_atom_lines(atoms, coords),
        "",
        "$end",
        "",
    ]
    return "\n".join(lines)


def xtb_hessian_to_orca_hessian(
    *,
    hessian_path: str | Path,
    atoms: list[str],
    coords: list[list[float]],
    energy: float = 0.0,
    multiplicity: int = 1,
) -> str:
    matrix = read_xtb_hessian(hessian_path, natoms=len(atoms))
    return write_orca_hessian_string(
        atoms=atoms,
        coords=coords,
        hessian=matrix,
        energy=energy,
        multiplicity=multiplicity,
    )
