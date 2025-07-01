# test_integration.py - Integration tests
"""
test_integration.py - Integration tests for complete workflows
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

try:
    from sirf_simind_connection import SimindSimulator, SimulationConfig
    from sirf.STIR import ImageData, AcquisitionData, OSMAPOSLReconstructor
    INTEGRATION_DEPS_AVAILABLE = True
except ImportError:
    INTEGRATION_DEPS_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration dependencies not available")
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_simulation_workflow(self, temp_dir):
        """Test complete simulation workflow without SIMIND."""
        # This test uses mocked SIMIND to test the workflow structure
        # Create test phantom
        phantom = self._create_test_phantom()
        mu_map = self._create_test_attenuation_map()
        
        # Create configuration
        config_content = """
        TITLE Integration Test
        PHOTONS 10000
        SPECTRUM MONO 140.0
        DETECTOR NaI 0.95
        MATRIX 64 64
        """
        config_file = temp_dir / "config.smc"
        config_file.write_text(config_content)
        
        # Test configuration loading
        config = SimulationConfig(str(config_file))
        saved_config = config.save_file(str(temp_dir / "saved_config.smc"))
        
        assert Path(saved_config).exists()
        
        # Test simulator setup (without actual SIMIND run)
        simulator = SimindSimulator(
            template_smc_file_path=saved_config,
            output_dir=str(temp_dir),
            source=phantom,
            mu_map=mu_map
        )
        
        simulator.set_windows(
            lower_bounds=[126, 100, 154],
            upper_bounds=[154, 126, 180],
            scatter_orders=[0, 1, 1]
        )
        
        assert len(simulator.windows) == 3
        assert simulator.source is not None
        assert simulator.mu_map is not None
    
    def test_scatter_correction_workflow(self, temp_dir):
        """Test scatter correction workflow with synthetic data."""
        # Create synthetic projection data
        proj_data = self._create_synthetic_projections()
        scatter_data = self._create_synthetic_scatter()
        
        # Test scatter correction calculation
        corrected_data = self._apply_scatter_correction(proj_data, scatter_data)
        
        # Verify correction reduces scatter contamination
        original_total = np.sum(proj_data)
        corrected_total = np.sum(corrected_data)
        scatter_total = np.sum(scatter_data)
        
        assert corrected_total < original_total
        assert corrected_total == pytest.approx(original_total - scatter_total, rel=0.1)
    
    def _create_test_phantom(self):
        """Create a simple test phantom."""
        dimensions = (64, 64, 32)
        phantom_array = np.zeros(dimensions)
        
        # Simple cylindrical phantom
        center = np.array(dimensions) // 2
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                for k in range(dimensions[2]):
                    if (i - center[0])**2 + (j - center[1])**2 <= 400:  # radius = 20
                        phantom_array[i, j, k] = 1.0
        
        phantom = ImageData()
        phantom.initialise(dimensions)
        phantom.fill(phantom_array)
        phantom.set_voxel_spacing((0.4, 0.4, 0.4))
        
        return phantom
    
    def _create_test_attenuation_map(self):
        """Create a simple attenuation map."""
        dimensions = (64, 64, 32)
        mu_array = np.zeros(dimensions)
        
        # Simple uniform attenuation
        center = np.array(dimensions) // 2
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                for k in range(dimensions[2]):
                    if (i - center[0])**2 + (j - center[1])**2 <= 400:
                        mu_array[i, j, k] = 0.15  # soft tissue
        
        mu_map = ImageData()
        mu_map.initialise(dimensions)
        mu_map.fill(mu_array)
        mu_map.set_voxel_spacing((0.4, 0.4, 0.4))
        
        return mu_map
    
    def _create_synthetic_projections(self):
        """Create synthetic projection data."""
        # Simple synthetic projections with Poisson noise
        projections = np.random.poisson(1000, (60, 64, 64))
        return projections
    
    def _create_synthetic_scatter(self):
        """Create synthetic scatter data."""
        # Scatter is typically smoother and lower amplitude
        scatter = np.random.poisson(200, (60, 64, 64))
        # Smooth the scatter
        from scipy.ndimage import gaussian_filter
        for i in range(scatter.shape[0]):
            scatter[i] = gaussian_filter(scatter[i], sigma=2.0)
        return scatter
    
    def _apply_scatter_correction(self, projections, scatter):
        """Apply simple scatter correction."""
        corrected = np.maximum(projections - scatter, 0.1 * projections)
        return corrected


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests for the package."""
    
    def test_simulation_setup_performance(self, performance_timer, temp_dir):
        """Test performance of simulation setup."""
        if not INTEGRATION_DEPS_AVAILABLE:
            pytest.skip("Dependencies not available")
        
        # Create large phantom for performance testing
        dimensions = (128, 128, 64)
        phantom_array = np.random.rand(*dimensions)
        
        phantom = ImageData()
        phantom.initialise(dimensions)
        phantom.fill(phantom_array)
        phantom.set_voxel_spacing((0.4, 0.4, 0.4))
        
        mu_array = np.random.rand(*dimensions) * 0.2
        mu_map = ImageData()
        mu_map.initialise(dimensions)
        mu_map.fill(mu_array)
        mu_map.set_voxel_spacing((0.4, 0.4, 0.4))
        
        # Time the setup
        performance_timer.start()
        
        config_content = """
        TITLE Performance Test
        PHOTONS 1000000
        SPECTRUM MONO 140.0
        """
        config_file = temp_dir / "perf_config.smc"
        config_file.write_text(config_content)
        
        config = SimulationConfig(str(config_file))
        saved_config = config.save_file(str(temp_dir / "perf_saved.smc"))
        
        simulator = SimindSimulator(
            template_smc_file_path=saved_config,
            output_dir=str(temp_dir),
            source=phantom,
            mu_map=mu_map
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        performance_timer.stop()
        
        # Performance assertion
        assert performance_timer.elapsed < 10.0  # Should complete in less than 10 seconds
    
    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        if not INTEGRATION_DEPS_AVAILABLE:
            pytest.skip("Dependencies not available")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large data structures
        large_phantom = np.random.rand(256, 256, 128)
        large_projections = np.random.rand(120, 256, 256)
        
        # Convert to SIRF objects
        phantom = ImageData()
        phantom.initialise((256, 256, 128))
        phantom.fill(large_phantom)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 2GB for this test)
        assert memory_increase < 2048  # MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])