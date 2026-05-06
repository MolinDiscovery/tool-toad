# Tool Toad

### Installation
```bash
# clone repository
pip install .
```

> [!NOTE]
> Several function utilize [xTB](https://xtb-docs.readthedocs.io/en/latest/) and [CREST](https://crest-lab.github.io/crest-docs/) which can be installed using `conda install -c conda-forge xtb crest`
> Furthermore, [ORCA](https://www.faccts.de/orca/) is required for some routines and must be installed manually.
> Tool paths should usually be stored in `~/.env`, including `ORCA_EXE`, `OPEN_MPI_DIR`, `XTB_EXE`, and optional downstream paths such as `OET_TOOLS` and `GXTB_EXE`; see [here](./recipes/qm_calculation.md).

### Examples
Check the [recipes](./recipes/) for example use cases.
