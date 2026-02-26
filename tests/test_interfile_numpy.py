from pathlib import Path

import numpy as np
import pytest

from sirf_simind_connection.utils.interfile_numpy import load_interfile_array


@pytest.mark.unit
def test_load_interfile_array_round_trip(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)

    assert loaded.header_path == header_path.resolve()
    assert loaded.data_path == data_path.resolve()
    assert loaded.array.shape == (2, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_requires_matrix_sizes(tmp_path: Path):
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    np.zeros(8, dtype=np.float32).tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    with pytest.raises(ValueError, match="matrix size"):
        load_interfile_array(header_path)


@pytest.mark.unit
def test_load_interfile_array_infers_projection_axis_from_header(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!number of projections := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (2, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_prefers_header_projection_count(tmp_path: Path):
    data = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!number of projections := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (2, 3, 4)
    assert np.array_equal(loaded.array, data[:2, :, :])


@pytest.mark.unit
def test_load_interfile_array_infers_leading_axis_when_header_omits_count(tmp_path: Path):
    data = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (3, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_promotes_single_plane_to_3d(tmp_path: Path):
    data = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (1, 3, 4)
    assert np.array_equal(loaded.array[0], data)
