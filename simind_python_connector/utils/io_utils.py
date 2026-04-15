import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_directory():
    """Context manager for creating and cleaning up a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
