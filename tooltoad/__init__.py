#__init__.py

# core utility submodules
from .chemutils      import *
from .vis            import *

# quantum‐chemistry interfaces
from .orca           import orca_calculate
from .xtb            import xtb_calculate

from .orca import orca
from .xtb  import xtb

# data providers
from .periodictable  import PeriodicTable

# convenience imports
from .crest          import run_crest
from .thermochemistry import *

__all__ = [
    *chemutils.__all__,
    *vis.__all__,

    # direct functions
    "orca_calculate",
    "xtb_calculate",
    "PeriodicTable",
    "CM5Charges",
    "run_crest",
]