# Performing QM-calculations using xTB and ORCA

> [!IMPORTANT]
> These functions require xTB or ORCA be be installed, see instructions for [xTB](https://xtb-docs.readthedocs.io/en/latest/setup.html) or [ORCA](https://www.faccts.de/docs/orca/6.0/tutorials/first_steps/install.html).
Furthermore, follow the instructions below to set all relevant paths in `.env`.

<details>
  <summary>Setting paths to executables</summary>

  For `xtb_calculate`, the path to the xTB executable should be in PATH, otherwise, the path to the exectuable can be parsed to `xtb_calculate` like this:

```python
results = xtb_calculate(
    atoms=atoms,
    coords=coords,
    options={"gfn2" None},
    xtb_cmd="path/to/xtb"
)
```

For `orca_calculate`, put the following paths in `~/.env`:

```bash
ORCA_EXE=/path/to/orca_version/orca
OPEN_MPI_DIR=${HOME}/openmpi
XTBPATH=/path/to/xtb/share/xtb
XTB_EXE=xtb # only required for xTB calculations via ORCA
```

For g-xTB or downstream FRUST/OET workflows, the same `~/.env` file can also
hold:

```bash
OET_TOOLS=/path/to/orca-external-tools
GXTB_ROOT=/path/to/g-xtb
GXTB_EXE=/path/to/g-xtb/bin/xtb
```

Use `${HOME}`, not bare `$HOME`, inside `.env` files. Tooltoad loads dotenv
files in this order:

1. path set by `TOOLTOAD_DOTENV_PATH`;
2. `.env` in the current working directory;
3. `~/.env`.

Shell `export ...` commands still work for temporary sessions, CI, and job
scripts, but stable local tool paths should normally live in `~/.env`.
</details>


## Performing xTB calculations

Any [xTB arguments](https://xtb-docs.readthedocs.io/en/latest/commandline.html) are parsed as a `dict` to `options`:

```python
from tooltoad.xtb import xtb_calculate

results = xtb_calculate(
    atoms=atoms,
    coords=coords,
    charge=0,
    multiplicity=1,
    options={"opt": None, "alpb": "water"},
    n_cores=4,
)
```

The function returns a dictionary of the results with keys such as:

* bool: `normal_termination`
* float: `electronic_energy`
* list[list[float]]: `opt_coords` (only when `opt` requested)

The whole xTB output is available under `log` when the calculation failed:

```python
results = xtb_calculate(
    atoms=atoms,
    coords=coords,
    charge=-1, # wrong charge
    multiplicity=1,
    options={"opt": None, "alpb": "water"},
    n_cores=4,
)

print(results["log"])
```

Furthermore, the output can be accessed with python loggers:

```python
import logging

xtb_logger = logging.get_logger("xtb")
xtb_logger.setLevel(logging.DEBUG)
xtb_logger.addHandler(logging.FileHandler("xtb.log"))
```


## Performing ORCA calculations

Any [ORCA arguments](https://www.faccts.de/docs/orca/6.0/manual/index.html) are parsed as a `dict` to `options`:

```python
from tooltoad.orca import orca_calculate

results = orca_calculate(
    atoms=atoms,
    coords=coords,
    charge=0,
    multiplicity=1,
    options={"r2scan-3c": None, "SMD": "water", "opt": None},
    n_cores=4,
)
```

Again, a dictionary of the results is returned with keys such as:

* bool: `normal_termination`
* float: `electronic_energy`
* list[list[float]]: `opt_coords` (only when `opt` requested)

Additional input for ORCA can be parsed to `xtra_inp_str`, such as options for constraints, scans, molecular dynamics, etc., for example:

```python
detailed_input = """%geom
TS_Mode { A 1 0 2 } end
modify_internal { B 0 5 A } end
Calc_Hess true
end"""

ts_opt_results = orca_calculate(
    atoms=atoms,
    coords=coords,
    charge=charge,
    options={"OptTS": None, "Freq": None, "r2scan-3c": None},
    xtra_inp_str=detailed_input
)
```

The ORCA output can again be accessed using the python logger:

```python
orca_logger = logging.get_logger("orca")
```
