import pytest
from sirf_simind_connection import SimindProjector

def test_projector_initialization():
    """Test initialization of SimindProjector."""
    projector = SimindProjector()
    assert projector is not None
