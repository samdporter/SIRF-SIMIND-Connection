import os
import subprocess

import pytest


# Integration tests may require both SIRF and SIMIND
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_sirf,
    pytest.mark.requires_simind,
]


@pytest.mark.integration
def test_run_all_examples():
    """Test running all examples as an integration test."""
    script_path = os.path.join(
        os.path.dirname(__file__), "../scripts/run_all_examples.py"
    )

    # Run the example script with 'y' input to proceed
    result = subprocess.run(
        ["python", script_path],
        input="y\n",  # Provide 'y' as input to proceed
        capture_output=True,
        text=True,
        timeout=3600,
    )

    # Check that script completes successfully
    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
