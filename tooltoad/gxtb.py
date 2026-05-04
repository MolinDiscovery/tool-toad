import os
from datetime import datetime

from tooltoad.config import find_and_load_dotenv
from tooltoad.xtb import xtb_calculate

find_and_load_dotenv()


def _resolve_gxtb_cmd(gxtb_cmd: str | None = None) -> str:
    cmd = gxtb_cmd or os.getenv("GXTB_EXE")
    if not cmd:
        raise RuntimeError(
            "GXTB_EXE is not set. Set it to the special g-xTB v2 xtb executable."
        )
    return cmd


def _with_gxtb_option(options: dict | None) -> dict:
    merged = {"gxtb": None}
    if options:
        merged.update(options)
    return merged


def gxtb_calculate(
    atoms: list[str],
    coords: list[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict | None = None,
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: None | dict = None,
    detailed_input_str: None | str = None,
    calc_dir: None | str = None,
    gxtb_cmd: str | None = None,
    force: bool = False,
    data2file: None | dict = None,
) -> dict:
    """Run g-xTB v2 through its special xtb executable.

    The current g-xTB v2 distribution is invoked as ``xtb mol.xyz --gxtb``.
    This wrapper keeps Tooltoad's xTB result contract while requiring a separate
    ``GXTB_EXE`` so regular xTB installations are not confused with g-xTB.
    """
    return xtb_calculate(
        atoms=atoms,
        coords=coords,
        charge=charge,
        multiplicity=multiplicity,
        options=_with_gxtb_option(options),
        scr=scr,
        n_cores=n_cores,
        detailed_input=detailed_input,
        detailed_input_str=detailed_input_str,
        calc_dir=calc_dir,
        xtb_cmd=_resolve_gxtb_cmd(gxtb_cmd),
        force=force,
        data2file=data2file,
    )


def mock_gxtb_calculate(
    atoms: list[str],
    coords: list[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict | None = None,
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: None | dict = None,
    detailed_input_str: None | str = None,
    calc_dir: None | str = None,
    gxtb_cmd: str | None = None,
    force: bool = False,
    data2file: None | dict = None,
) -> dict:
    """Mock g-xTB result matching the shape of the real calculator."""
    time = datetime.now()
    results = {
        "normal_termination": True,
        "electronic_energy": 0.0,
        "atoms": atoms,
        "coords": coords,
        "charge": charge,
        "multiplicity": multiplicity,
        "options": _with_gxtb_option(options),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.timestamp(),
    }
    if options and any(key in options for key in ("opt", "ohess")):
        results["opt_coords"] = coords
    return results
