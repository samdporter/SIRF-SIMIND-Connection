"""
Type definitions, enums, exceptions, and constants for SIMIND simulation.

This module contains types that don't depend on SIRF, allowing them to be
imported without triggering SIRF dependencies.
"""

from enum import Enum


# =============================================================================
# EXCEPTIONS
# =============================================================================


class SimindError(Exception):
    """Base exception for SIMIND simulation errors."""


class ValidationError(SimindError):
    """Raised when validation fails."""


class SimulationError(SimindError):
    """Raised when simulation execution fails."""


class OutputError(SimindError):
    """Raised when output processing fails."""


# =============================================================================
# ENUMS
# =============================================================================


class RotationDirection(Enum):
    """Rotation direction for SPECT acquisition."""

    CCW = "ccw"
    CW = "cw"


class ScatterType(Enum):
    """SIMIND scatter output types."""

    TOTAL = "tot"
    SCATTER = "sca"
    PRIMARY = "pri"
    AIR = "air"


class ScoringRoutine(Enum):
    """Enum for different SIMIND scoring routines."""

    DUMMY = 0
    SCATTWIN = 1
    LIST_MODE = 2
    FORCED_COLLIMATION = 3
    PENETRATE = 4


class PenetrateOutputType(Enum):
    """Enum for different penetrate routine output components."""

    # Without backscatter (*.b10-*.b17)
    ALL_INTERACTIONS = 1  # *.b01
    GEOM_COLL_PRIMARY_ATT = 2  # *.b02
    SEPTAL_PENETRATION_PRIMARY_ATT = 3  # *.b03
    COLL_SCATTER_PRIMARY_ATT = 4  # *.b04
    COLL_XRAY_PRIMARY_ATT = 5  # *.b05
    GEOM_COLL_SCATTERED = 6  # *.b06
    SEPTAL_PENETRATION_SCATTERED = 7  # *.b07
    COLL_SCATTER_SCATTERED = 8  # *.b08
    COLL_XRAY_SCATTERED = 9  # *.b09
    # With backscatter (*.b10-*.b17)
    GEOM_COLL_PRIMARY_ATT_BACK = 10  # *.b10
    SEPTAL_PENETRATION_PRIMARY_ATT_BACK = 11  # *.b11
    COLL_SCATTER_PRIMARY_ATT_BACK = 12  # *.b12
    COLL_XRAY_PRIMARY_ATT_BACK = 13  # *.b13
    GEOM_COLL_SCATTERED_BACK = 14  # *.b14
    SEPTAL_PENETRATION_SCATTERED_BACK = 15  # *.b15
    COLL_SCATTER_SCATTERED_BACK = 16  # *.b16
    COLL_XRAY_SCATTERED_BACK = 17  # *.b17
    ALL_UNSCATTERED_UNATTENUATED = 18  # *.b18
    ALL_UNSCATTERED_UNATTENUATED_GEOM_COLL = 19  # *.b19


# =============================================================================
# CONSTANTS
# =============================================================================

SIMIND_VOXEL_UNIT_CONVERSION = 10  # mm to cm
# Maximum normalised value of source image
# You would have expected this to be 65535, but it is not
# I have no understanding why, but it is the case
# 500 seems a reasonable value that improves precision
# whilst not exceeding the maximum value (weird things happen)
MAX_SOURCE = 500
ORBIT_FILE_EXTENSION = ".cor"
OUTPUT_EXTENSIONS = [".h00", ".a00", ".hs"]


__all__ = [
    # Exceptions
    "SimindError",
    "ValidationError",
    "SimulationError",
    "OutputError",
    # Enums
    "RotationDirection",
    "ScatterType",
    "ScoringRoutine",
    "PenetrateOutputType",
    # Constants
    "SIMIND_VOXEL_UNIT_CONVERSION",
    "MAX_SOURCE",
    "ORBIT_FILE_EXTENSION",
    "OUTPUT_EXTENSIONS",
]
