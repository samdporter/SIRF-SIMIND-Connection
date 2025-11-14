import pytest

from sirf_simind_connection.utils.io_utils import temporary_directory
from sirf_simind_connection.utils.simind_utils import create_window_file
from sirf_simind_connection.utils.stir_utils import parse_sinogram


@pytest.mark.unit
def test_create_window_file_writes_expected_lines(tmp_path):
    win_stem = tmp_path / "energy"
    create_window_file(140.0, 160.0, 0, output_filename=str(win_stem))

    win_file = win_stem.with_suffix(".win")
    assert win_file.exists()

    lines = win_file.read_text().splitlines()
    assert lines[0] == "140.0,160.0,0"
    # Additional scatter-only line should be appended to encourage SIMIND output
    assert lines[-1].endswith(",1")


@pytest.mark.unit
def test_create_window_file_overwrites_existing(tmp_path):
    win_file = tmp_path / "overwrite.win"
    win_file.write_text("old contents")

    create_window_file([120], [140], [1], output_filename=str(win_file))

    contents = win_file.read_text()
    assert contents != "old contents"
    assert "120.0,140.0,1" in contents


@pytest.mark.unit
def test_temporary_directory_context_manager():
    with temporary_directory() as tmpdir:
        assert tmpdir.exists()
        marker = tmpdir / "marker.txt"
        marker.write_text("ok")
        assert marker.exists()

    # Context manager should clean up the directory tree
    assert not tmpdir.exists()


@pytest.mark.unit
def test_parse_sinogram_from_path(tmp_path):
    header = tmp_path / "template.hs"
    header.write_text(
        "!INTERFILE :=\n"
        "!matrix size [1] := 64\n"
        "start angle := 180\n"
    )

    values = parse_sinogram(header)
    assert values["!matrix size [1]"] == "64"
    assert values["start angle"] == "180"
