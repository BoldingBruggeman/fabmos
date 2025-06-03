import importlib.metadata
__version__ = importlib.metadata.version(__name__)

from pygetm import TimeUnit, CoordinateType
from pygetm import vertical_coordinates
from pygetm.core import Array, Grid

from . import domain
from . import input
from .simulator import Simulator
from . import environment
