"""Tests for arithmetic operations on backend wrappers."""

import pytest

from sirf_simind_connection.utils.stir_utils import create_stir_acqdata


# Mark as requiring backend
pytestmark = pytest.mark.requires_sirf


def test_scalar_multiplication(acquisition_template):
    """Test scalar multiplication on wrapped acquisition data."""
    a = acquisition_template.clone()
    a.fill(10.0)

    # Test scalar multiplication (what failed in 03_multi_window.py)
    c = a * 2.0
    expected_sum = a.sum() * 2.0
    assert abs(c.sum() - expected_sum) < 1e-6, "Scalar multiplication failed"


def test_right_hand_scalar_multiplication(acquisition_template):
    """Test right-hand scalar multiplication."""
    b = acquisition_template.clone()
    b.fill(5.0)

    d = 3.0 * b
    expected_sum = b.sum() * 3.0
    assert abs(d.sum() - expected_sum) < 1e-6, "Right-hand scalar multiplication failed"


def test_addition(acquisition_template):
    """Test addition of two acquisition data objects."""
    a = acquisition_template.clone()
    b = acquisition_template.clone()
    a.fill(10.0)
    b.fill(5.0)

    e = a + b
    expected_sum = a.sum() + b.sum()
    assert abs(e.sum() - expected_sum) < 1e-6, "Addition failed"


def test_subtraction(acquisition_template):
    """Test subtraction of two acquisition data objects."""
    a = acquisition_template.clone()
    b = acquisition_template.clone()
    a.fill(10.0)
    b.fill(5.0)

    f = a - b
    expected_sum = a.sum() - b.sum()
    assert abs(f.sum() - expected_sum) < 1e-6, "Subtraction failed"


def test_division_by_scalar(acquisition_template):
    """Test division by scalar."""
    a = acquisition_template.clone()
    a.fill(20.0)

    c = a / 2.0
    expected_sum = a.sum() / 2.0
    assert abs(c.sum() - expected_sum) < 1e-6, "Division by scalar failed"


def test_negation(acquisition_template):
    """Test negation operator."""
    a = acquisition_template.clone()
    a.fill(10.0)

    b = -a
    expected_sum = -a.sum()
    assert abs(b.sum() - expected_sum) < 1e-6, "Negation failed"


def test_complex_expression(acquisition_template):
    """Test the complex expression from 03_multi_window.py example."""
    # This is the operation that originally failed:
    # scatter_estimate = lower_scatter * (window_widths[1] / (2 * window_widths[0]))
    #                    + upper_scatter * (window_widths[1] / (2 * window_widths[2]))

    window_widths = [6, 28, 6]  # keV
    lower_scatter = acquisition_template.clone()
    lower_scatter.fill(100.0)
    upper_scatter = acquisition_template.clone()
    upper_scatter.fill(80.0)

    # Calculate expected value
    factor1 = window_widths[1] / (2 * window_widths[0])  # 28 / 12 = 2.333...
    factor2 = window_widths[1] / (2 * window_widths[2])  # 28 / 12 = 2.333...
    expected_sum = lower_scatter.sum() * factor1 + upper_scatter.sum() * factor2

    # Perform the operation
    scatter_estimate = lower_scatter * factor1 + upper_scatter * factor2

    assert abs(scatter_estimate.sum() - expected_sum) < 1e-3, (
        "Complex expression failed"
    )


@pytest.fixture
def acquisition_template():
    """Provide acquisition data template for testing."""
    # Create template with 60 projections, 64x64 matrix
    return create_stir_acqdata([64, 64], 60, [4.42, 4.42])
