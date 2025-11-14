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

    def __new__(cls, value: int, slug: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.slug = slug
        obj.description = description
        return obj

    # Without backscatter (*.b01-*.b09)
    ALL_INTERACTIONS = (
        1,
        "all_interactions",
        "All type of interactions",
    )
    GEOM_COLL_PRIMARY_ATT = (
        2,
        "geom_coll_primary",
        "Geometrically collimated primary attenuated photons",
    )
    SEPTAL_PENETRATION_PRIMARY_ATT = (
        3,
        "septal_pen_primary",
        "Septal penetration from primary attenuated photons",
    )
    COLL_SCATTER_PRIMARY_ATT = (
        4,
        "coll_scatter_primary",
        "Collimator scatter from primary attenuated photons",
    )
    COLL_XRAY_PRIMARY_ATT = (
        5,
        "coll_xray_primary",
        "X-rays from collimator (primary attenuated photons)",
    )
    GEOM_COLL_SCATTERED = (
        6,
        "geom_coll_scattered",
        "Geometrically collimated scattered photons",
    )
    SEPTAL_PENETRATION_SCATTERED = (
        7,
        "septal_pen_scattered",
        "Septal penetration from scattered photons",
    )
    COLL_SCATTER_SCATTERED = (
        8,
        "coll_scatter_scattered",
        "Collimator scatter from scattered photons",
    )
    COLL_XRAY_SCATTERED = (
        9,
        "coll_xray_scattered",
        "X-rays from collimator (scattered photons)",
    )
    # With backscatter (*.b10-*.b17)
    GEOM_COLL_PRIMARY_ATT_BACK = (
        10,
        "geom_coll_primary_back",
        "Geometrically collimated primary attenuated photons (with backscatter)",
    )
    SEPTAL_PENETRATION_PRIMARY_ATT_BACK = (
        11,
        "septal_pen_primary_back",
        "Septal penetration from primary attenuated photons (with backscatter)",
    )
    COLL_SCATTER_PRIMARY_ATT_BACK = (
        12,
        "coll_scatter_primary_back",
        "Collimator scatter from primary attenuated photons (with backscatter)",
    )
    COLL_XRAY_PRIMARY_ATT_BACK = (
        13,
        "coll_xray_primary_back",
        "X-rays from collimator, primary attenuated photons (with backscatter)",
    )
    GEOM_COLL_SCATTERED_BACK = (
        14,
        "geom_coll_scattered_back",
        "Geometrically collimated scattered photons (with backscatter)",
    )
    SEPTAL_PENETRATION_SCATTERED_BACK = (
        15,
        "septal_pen_scattered_back",
        "Septal penetration from scattered photons (with backscatter)",
    )
    COLL_SCATTER_SCATTERED_BACK = (
        16,
        "coll_scatter_scattered_back",
        "Collimator scatter from scattered photons (with backscatter)",
    )
    COLL_XRAY_SCATTERED_BACK = (
        17,
        "coll_xray_scattered_back",
        "X-rays from collimator, scattered photons (with backscatter)",
    )
    ALL_UNSCATTERED_UNATTENUATED = (
        18,
        "unscattered_unattenuated",
        "Photons without scattering and attenuation in phantom",
    )
    ALL_UNSCATTERED_UNATTENUATED_GEOM_COLL = (
        19,
        "unscattered_unattenuated_geom_coll",
        "Photons without scattering/attenuation, geometrically collimated",
    )


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
