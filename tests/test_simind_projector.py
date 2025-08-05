import pytest

from sirf_simind_connection import SimindProjector

# All tests in this file require SIRF since SimindProjector uses SIRF
pytestmark = pytest.mark.requires_sirf


def test_projector_initialization():
    """Test initialization of SimindProjector."""
    projector = SimindProjector()
    assert projector is not None
