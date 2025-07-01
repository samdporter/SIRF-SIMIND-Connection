# test_dicom_converter.py - Unit tests for DICOM conversion
"""
test_dicom_converter.py - Unit tests for DICOM conversion functionality
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from sirf_simind_connection.dicom_converter import DicomConverter
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not DICOM_AVAILABLE, reason="DICOM functionality not available")
class TestDicomConverter:
    """Unit tests for DICOM conversion functionality."""
    
    def test_converter_initialization(self):
        """Test DicomConverter initialization."""
        converter = DicomConverter()
        assert converter is not None
    
    @patch('pydicom.dcmread')
    def test_read_dicom_file(self, mock_dcmread, temp_dir):
        """Test reading DICOM file."""
        # Create mock DICOM dataset
        mock_dataset = Mock()
        mock_dataset.pixel_array = np.random.randint(0, 4096, (512, 512))
        mock_dataset.RescaleSlope = 1.0
        mock_dataset.RescaleIntercept = 0.0
        mock_dcmread.return_value = mock_dataset
        
        converter = DicomConverter()
        
        # Create dummy DICOM file
        dicom_file = temp_dir / "test.dcm"
        dicom_file.write_bytes(b"dummy dicom data")
        
        dataset = converter.read_dicom(str(dicom_file))
        assert dataset is not None
        mock_dcmread.assert_called_once()
    
    def test_invalid_dicom_file(self):
        """Test handling of invalid DICOM file."""
        converter = DicomConverter()
        
        with pytest.raises(FileNotFoundError):
            converter.read_dicom("nonexistent.dcm")
    
    @patch('pydicom.dcmread')
    def test_convert_to_stir_format(self, mock_dcmread, temp_dir):
        """Test conversion to STIR format."""
        # Create mock DICOM dataset with SPECT data
        mock_dataset = Mock()
        mock_dataset.pixel_array = np.random.randint(0, 4096, (60, 128, 128))  # projections
        mock_dataset.NumberOfFrames = 60
        mock_dataset.Rows = 128
        mock_dataset.Columns = 128
        mock_dataset.PixelSpacing = [0.4, 0.4]
        mock_dataset.SliceThickness = 0.4
        mock_dcmread.return_value = mock_dataset
        
        converter = DicomConverter()
        
        # Create dummy input file
        input_file = temp_dir / "input.dcm"
        input_file.write_bytes(b"dummy")
        
        output_file = str(temp_dir / "output.hs")
        
        result = converter.convert_to_stir(str(input_file), output_file)
        
        # Should return output filename
        assert result is not None
    
    def test_extract_imaging_parameters(self):
        """Test extraction of imaging parameters from DICOM."""
        # Create mock DICOM dataset
        mock_dataset = Mock()
        mock_dataset.StudyDescription = "SPECT Study"
        mock_dataset.SeriesDescription = "Tc99m MIBI"
        mock_dataset.RadiopharmaceuticalInformationSequence = [Mock()]
        mock_dataset.RadiopharmaceuticalInformationSequence[0].Radionuclide = "Tc-99m"
        mock_dataset.DetectorInformationSequence = [Mock()]
        mock_dataset.EnergyWindowInformationSequence = [Mock()]
        
        converter = DicomConverter()
        params = converter.extract_parameters(mock_dataset)
        
        assert isinstance(params, dict)
        assert 'study_description' in params or 'series_description' in params