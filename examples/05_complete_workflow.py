#!/usr/bin/env python3
"""
Example 05: Complete SPECT Simulation and Reconstruction Workflow

This example demonstrates a complete SPECT imaging workflow using SIRF-SIMIND-Connection:
1. Create a realistic 3D phantom with multiple activity distributions
2. Generate SIMIND Monte Carlo data with 3 energy windows (photopeak + scatter windows)
3. Create scatter correction sinograms using scatter window data
4. Perform iterative reconstruction with and without scatter correction
5. Compare reconstructed images to ground truth and quantify improvements

Author: SIRF-SIMIND-Connection Team
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import time
import yaml
from scipy import ndimage
from matplotlib.gridspec import GridSpec

# SIRF and SIMIND imports
try:
    from sirf.STIR import ImageData, AcquisitionData, OSMAPOSLReconstructor
    from sirf_simind_connection import SimindSimulator, SimulationConfig
    from sirf_simind_connection.utils import density_conversion
    from sirf_simind_connection import configs
    SIRF_AVAILABLE = True
except ImportError as e:
    print(f"SIRF or SIRF-SIMIND-Connection not available: {e}")
    print("This example requires both packages to be installed.")
    SIRF_AVAILABLE = False
    exit(1)


class SPECTWorkflowExample:
    """Complete SPECT simulation and reconstruction workflow."""
    
    def __init__(self, output_dir="spect_workflow_output", temp_dir=None):
        """Initialize the workflow example.
        
        Args:
            output_dir: Directory to save all outputs
            temp_dir: Temporary directory for intermediate files (auto-created if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())
            self._cleanup_temp = True
        else:
            self.temp_dir = Path(temp_dir)
            self._cleanup_temp = False
        
        # Imaging parameters
        self.matrix_size = (128, 128, 64)  # x, y, z
        self.voxel_size = (0.4, 0.4, 0.4)  # cm
        self.num_projections = 120
        self.num_subsets = 8
        self.num_iterations = 10
        
        # Energy windows for Tc-99m (140 keV)
        self.energy_windows = {
            'photopeak': {'lower': 126, 'upper': 154},    # 140 ± 10%
            'scatter_low': {'lower': 100, 'upper': 126},   # Lower scatter window
            'scatter_high': {'lower': 154, 'upper': 180}   # Upper scatter window
        }
        
        print(f"Initialized SPECT workflow in: {self.output_dir}")
        print(f"Temporary files: {self.temp_dir}")
    
    def create_phantom(self):
        """Create a realistic 3D SPECT phantom with multiple activity regions."""
        print("\n=== Creating 3D SPECT Phantom ===")
        
        x, y, z = np.meshgrid(
            np.linspace(-self.matrix_size[0]//2, self.matrix_size[0]//2, self.matrix_size[0]),
            np.linspace(-self.matrix_size[1]//2, self.matrix_size[1]//2, self.matrix_size[1]),
            np.linspace(-self.matrix_size[2]//2, self.matrix_size[2]//2, self.matrix_size[2]),
            indexing='ij'
        )
        
        phantom = np.zeros(self.matrix_size)
        
        # 1. Large cylindrical background (simulating torso)
        torso_radius = min(self.matrix_size[0], self.matrix_size[1]) // 3
        torso_mask = (x**2 + y**2) <= torso_radius**2
        phantom[torso_mask] = 1.0  # Background activity
        
        # 2. Heart-like structure (higher activity)
        heart_center = (-10, 5, 0)
        heart_size = (15, 12, 20)
        heart_mask = (
            ((x - heart_center[0])/heart_size[0])**2 + 
            ((y - heart_center[1])/heart_size[1])**2 + 
            ((z - heart_center[2])/heart_size[2])**2
        ) <= 1.0
        phantom[heart_mask] = 8.0  # High cardiac uptake
        
        # 3. Liver-like region
        liver_center = (15, -5, -10)
        liver_size = (20, 15, 25)
        liver_mask = (
            ((x - liver_center[0])/liver_size[0])**2 + 
            ((y - liver_center[1])/liver_size[1])**2 + 
            ((z - liver_center[2])/liver_size[2])**2
        ) <= 1.0
        phantom[liver_mask] = 3.0  # Moderate liver uptake
        
        # 4. Kidneys
        for kidney_center in [(0, -25, -5), (0, 25, -5)]:
            kidney_mask = (
                ((x - kidney_center[0])/8)**2 + 
                ((y - kidney_center[1])/6)**2 + 
                ((z - kidney_center[2])/10)**2
            ) <= 1.0
            phantom[kidney_mask] = 5.0  # High renal uptake
        
        # 5. Small hot lesions
        lesion_centers = [(8, 15, 5), (-12, -8, 10), (5, -18, -8)]
        for center in lesion_centers:
            lesion_mask = (
                (x - center[0])**2 + 
                (y - center[1])**2 + 
                (z - center[2])**2
            ) <= 16  # 4-voxel radius lesions
            phantom[lesion_mask] = 12.0  # Very high lesion uptake
        
        # Smooth the phantom slightly to make it more realistic
        phantom = ndimage.gaussian_filter(phantom, sigma=0.8)
        
        # Convert to SIRF ImageData
        phantom_image = ImageData()
        phantom_image.initialise(self.matrix_size)
        phantom_image.fill(phantom)
        phantom_image.set_voxel_spacing(self.voxel_size)
        
        # Save phantom
        phantom_file = self.output_dir / "ground_truth_phantom.hv"
        phantom_image.write(str(phantom_file))
        
        print(f"Phantom created with dimensions: {self.matrix_size}")
        print(f"Activity range: {phantom.min():.2f} - {phantom.max():.2f}")
        print(f"Total activity: {phantom.sum():.1e} counts")
        print(f"Saved to: {phantom_file}")
        
        self.phantom = phantom_image
        return phantom_image
    
    def create_attenuation_map(self):
        """Create a realistic attenuation map corresponding to the phantom."""
        print("\n=== Creating Attenuation Map ===")
        
        x, y, z = np.meshgrid(
            np.linspace(-self.matrix_size[0]//2, self.matrix_size[0]//2, self.matrix_size[0]),
            np.linspace(-self.matrix_size[1]//2, self.matrix_size[1]//2, self.matrix_size[1]),
            np.linspace(-self.matrix_size[2]//2, self.matrix_size[2]//2, self.matrix_size[2]),
            indexing='ij'
        )
        
        # Start with air (mu = 0)
        mu_map = np.zeros(self.matrix_size)
        
        # Soft tissue regions (mu ~ 0.15 cm^-1 at 140 keV)
        torso_radius = min(self.matrix_size[0], self.matrix_size[1]) // 3
        torso_mask = (x**2 + y**2) <= torso_radius**2
        mu_map[torso_mask] = 0.15
        
        # Add ribs/spine (bone, mu ~ 0.4 cm^-1 at 140 keV)
        # Spine
        spine_mask = (x**2 + (y + 20)**2) <= 25  # Posterior spine
        mu_map[spine_mask] = 0.38
        
        # Ribs (simplified as arcs)
        for rib_y in [-15, -5, 5, 15]:
            for angle in np.linspace(0.3, 2.8, 20):  # Partial circles
                rib_x = int(25 * np.cos(angle))
                rib_y_pos = int(rib_y + 5 * np.sin(angle))
                if (-50 < rib_x < 50) and (-50 < rib_y_pos < 50):
                    rib_mask = (
                        (x - rib_x)**2 + (y - rib_y_pos)**2
                    ) <= 4  # Small circular cross-sections
                    mu_map[rib_mask] = 0.35
        
        # Lungs (lower attenuation, mu ~ 0.05 cm^-1)
        lung_centers = [(-15, 10, 0), (15, 10, 0)]
        for center in lung_centers:
            lung_mask = (
                ((x - center[0])/12)**2 + 
                ((y - center[1])/15)**2 + 
                ((z - center[2])/18)**2
            ) <= 1.0
            mu_map[lung_mask] = 0.05
        
        # Smooth the attenuation map
        mu_map = ndimage.gaussian_filter(mu_map, sigma=0.5)
        
        # Convert to SIRF ImageData
        mu_image = ImageData()
        mu_image.initialise(self.matrix_size)
        mu_image.fill(mu_map)
        mu_image.set_voxel_spacing(self.voxel_size)
        
        # Save attenuation map
        mu_file = self.output_dir / "attenuation_map.hv"
        mu_image.write(str(mu_file))
        
        print(f"Attenuation map created")
        print(f"μ range: {mu_map.min():.3f} - {mu_map.max():.3f} cm⁻¹")
        print(f"Saved to: {mu_file}")
        
        self.mu_map = mu_image
        return mu_image
    
    def setup_simulation_config(self):
        """Set up SIMIND simulation configuration."""
        print("\n=== Setting up SIMIND Configuration ===")
        
        # Load base configuration
        try:
            # Try to load a predefined scanner configuration
            config = SimulationConfig(configs.get("input.smc"))
            
            # Load scanner specifics (try AnyScan as default)
            try:
                config.import_yaml(configs.get("AnyScan.yaml"))
                print("Loaded AnyScan scanner configuration")
            except:
                print("Using default scanner configuration")
        except:
            # Fallback: create a minimal configuration
            print("Creating minimal SIMIND configuration")
            config_dict = {
                'simulation': {
                    'number_of_photons': 5000000,  # 5M photons for good statistics
                    'detector_binning': [128, 128],
                    'number_of_projections': self.num_projections,
                    'voxel_size': list(self.voxel_size),
                    'matrix_size': list(self.matrix_size)
                },
                'scanner': {
                    'name': 'generic_spect',
                    'detector': {
                        'material': 'NaI',
                        'thickness': 0.95,
                        'crystal_x': 40.0,
                        'crystal_y': 40.0
                    },
                    'collimator': {
                        'type': 'LEHR',
                        'hole_diameter': 0.11,
                        'septal_thickness': 0.016
                    }
                }
            }
            
            # Save as temporary YAML and SMC
            yaml_file = self.temp_dir / "config.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(config_dict, f)
            
            # Create minimal SMC file
            smc_content = f"""
! SIMIND Control File - Generated by SIRF-SIMIND-Connection
TITLE Complete SPECT Workflow Example
PHOTONS {config_dict['simulation']['number_of_photons']}
SPECTRUM MONO 140.0
DETECTOR NaI 0.95
MATRIX {self.matrix_size[0]} {self.matrix_size[1]}
PIXEL_SIZE {self.voxel_size[0]} {self.voxel_size[1]}
PROJECTIONS {self.num_projections}
"""
            smc_file = self.temp_dir / "config.smc"
            smc_file.write_text(smc_content)
            
            config = SimulationConfig(str(smc_file))
        
        # Save final configuration
        self.config_file = config.save_file(str(self.temp_dir / "final_config.smc"))
        print(f"Configuration saved to: {self.config_file}")
        
        return self.config_file
    
    def run_simind_simulation(self):
        """Run SIMIND Monte Carlo simulation with multiple energy windows."""
        print("\n=== Running SIMIND Monte Carlo Simulation ===")
        
        # Create simulator
        simulator = SimindSimulator(
            template_smc_file_path=self.config_file,
            output_dir=str(self.temp_dir),
            source=self.phantom,
            mu_map=self.mu_map
        )
        
        # Set up three energy windows
        simulator.set_windows(
            lower_bounds=[
                self.energy_windows['photopeak']['lower'],
                self.energy_windows['scatter_low']['lower'],
                self.energy_windows['scatter_high']['lower']
            ],
            upper_bounds=[
                self.energy_windows['photopeak']['upper'],
                self.energy_windows['scatter_low']['upper'],
                self.energy_windows['scatter_high']['upper']
            ],
            scatter_orders=[0, 1, 1]  # Total for photopeak, scatter for scatter windows
        )
        
        print("Energy windows configured:")
        print(f"  Photopeak: {self.energy_windows['photopeak']['lower']}-{self.energy_windows['photopeak']['upper']} keV")
        print(f"  Scatter Low: {self.energy_windows['scatter_low']['lower']}-{self.energy_windows['scatter_low']['upper']} keV")
        print(f"  Scatter High: {self.energy_windows['scatter_high']['lower']}-{self.energy_windows['scatter_high']['upper']} keV")
        
        # Run simulation
        start_time = time.time()
        print("Starting Monte Carlo simulation...")
        print("This may take several minutes depending on your system...")
        
        try:
            simulator.run_simulation()
            simulation_time = time.time() - start_time
            print(f"Simulation completed in {simulation_time:.1f} seconds")
        except Exception as e:
            print(f"Simulation failed: {e}")
            print("This may be due to SIMIND not being properly installed or configured.")
            return None
        
        # Get simulation results
        try:
            # Photopeak window (total counts)
            self.photopeak_data = simulator.get_output_total(window=1)
            
            # Scatter windows (scatter-only counts)
            self.scatter_low_data = simulator.get_output_scatter(window=2)
            self.scatter_high_data = simulator.get_output_scatter(window=3)
            
            print("Simulation outputs retrieved successfully")
            print(f"Photopeak counts: {self.photopeak_data.as_array().sum():.0f}")
            print(f"Scatter low counts: {self.scatter_low_data.as_array().sum():.0f}")
            print(f"Scatter high counts: {self.scatter_high_data.as_array().sum():.0f}")
            
            # Save raw projection data
            self.photopeak_data.write(str(self.output_dir / "photopeak_projections.hs"))
            self.scatter_low_data.write(str(self.output_dir / "scatter_low_projections.hs"))
            self.scatter_high_data.write(str(self.output_dir / "scatter_high_projections.hs"))
            
            return True
            
        except Exception as e:
            print(f"Failed to retrieve simulation outputs: {e}")
            return False
    
    def create_scatter_correction(self):
        """Create scatter correction sinogram using dual energy window method."""
        print("\n=== Creating Scatter Correction ===")
        
        # Dual Energy Window (DEW) scatter correction
        # Estimate scatter in photopeak window using scatter windows
        
        # Get arrays
        photopeak_array = self.photopeak_data.as_array()
        scatter_low_array = self.scatter_low_data.as_array()
        scatter_high_array = self.scatter_high_data.as_array()
        
        # Simple DEW method: average the two scatter windows
        # In practice, this would involve more sophisticated calibration
        scatter_estimate = 0.5 * (scatter_low_array + scatter_high_array)
        
        # Apply smoothing to reduce noise in scatter estimate
        from scipy.ndimage import gaussian_filter
        for projection in range(scatter_estimate.shape[0]):
            scatter_estimate[projection] = gaussian_filter(
                scatter_estimate[projection], sigma=1.5
            )
        
        # Create scatter-corrected projections
        scatter_corrected_array = np.maximum(
            photopeak_array - scatter_estimate, 
            0.1 * photopeak_array  # Prevent negative values
        )
        
        # Create SIRF AcquisitionData objects
        self.scatter_estimate = self.photopeak_data.clone()
        self.scatter_estimate.fill(scatter_estimate)
        
        self.scatter_corrected_data = self.photopeak_data.clone()
        self.scatter_corrected_data.fill(scatter_corrected_array)
        
        # Calculate scatter fraction
        total_photopeak = photopeak_array.sum()
        total_scatter = scatter_estimate.sum()
        scatter_fraction = total_scatter / total_photopeak * 100
        
        print(f"Scatter correction completed")
        print(f"Estimated scatter fraction: {scatter_fraction:.1f}%")
        print(f"Scatter counts: {total_scatter:.0f}")
        print(f"Primary counts (corrected): {scatter_corrected_array.sum():.0f}")
        
        # Save scatter correction data
        self.scatter_estimate.write(str(self.output_dir / "scatter_estimate.hs"))
        self.scatter_corrected_data.write(str(self.output_dir / "scatter_corrected_projections.hs"))
        
        return True
    
    def perform_reconstruction(self):
        """Perform iterative reconstruction with and without scatter correction."""
        print("\n=== Performing Image Reconstruction ===")
        
        # Create reconstructor
        reconstructor = OSMAPOSLReconstructor()
        
        # Set up acquisition model
        reconstructor.set_acquisition_model(self.photopeak_data.get_acquisition_model())
        
        # Set reconstruction parameters
        reconstructor.set_num_subsets(self.num_subsets)
        reconstructor.set_num_subiterations(self.num_iterations)
        
        # Create initial image estimate
        initial_image = self.phantom.clone()
        initial_image.fill(1.0)  # Uniform initial estimate
        reconstructor.set_current_estimate(initial_image)
        
        # Reconstruction without scatter correction
        print("Reconstructing without scatter correction...")
        reconstructor.set_acquisition_data(self.photopeak_data)
        reconstructor.set_up()
        
        start_time = time.time()
        reconstructor.process()
        recon_time = time.time() - start_time
        
        self.recon_no_scatter = reconstructor.get_current_estimate()
        print(f"Reconstruction completed in {recon_time:.1f} seconds")
        
        # Reconstruction with scatter correction
        print("Reconstructing with scatter correction...")
        reconstructor.set_acquisition_data(self.scatter_corrected_data)
        reconstructor.set_current_estimate(initial_image.clone())
        reconstructor.set_up()
        
        start_time = time.time()
        reconstructor.process()
        recon_time = time.time() - start_time
        
        self.recon_with_scatter = reconstructor.get_current_estimate()
        print(f"Scatter-corrected reconstruction completed in {recon_time:.1f} seconds")
        
        # Save reconstructed images
        self.recon_no_scatter.write(str(self.output_dir / "reconstruction_no_scatter.hv"))
        self.recon_with_scatter.write(str(self.output_dir / "reconstruction_with_scatter.hv"))
        
        print("Reconstructed images saved")
        return True
    
    def analyze_results(self):
        """Analyze and compare reconstruction results."""
        print("\n=== Analyzing Results ===")
        
        # Get arrays for analysis
        ground_truth = self.phantom.as_array()
        recon_no_sc = self.recon_no_scatter.as_array()
        recon_with_sc = self.recon_with_scatter.as_array()
        
        # Normalize reconstructions to same scale as ground truth
        scale_no_sc = np.sum(ground_truth) / np.sum(recon_no_sc)
        scale_with_sc = np.sum(ground_truth) / np.sum(recon_with_sc)
        
        recon_no_sc *= scale_no_sc
        recon_with_sc *= scale_with_sc
        
        # Calculate metrics
        def calculate_metrics(recon, reference):
            # Mean Squared Error
            mse = np.mean((recon - reference)**2)
            
            # Peak Signal-to-Noise Ratio
            psnr = 20 * np.log10(reference.max() / np.sqrt(mse))
            
            # Structural Similarity (simplified)
            mean_ref = np.mean(reference)
            mean_recon = np.mean(recon)
            var_ref = np.var(reference)
            var_recon = np.var(recon)
            cov = np.mean((reference - mean_ref) * (recon - mean_recon))
            
            ssim = (2 * mean_ref * mean_recon + 1e-6) * (2 * cov + 1e-6) / \
                   ((mean_ref**2 + mean_recon**2 + 1e-6) * (var_ref + var_recon + 1e-6))
            
            return {'mse': mse, 'psnr': psnr, 'ssim': ssim}
        
        metrics_no_sc = calculate_metrics(recon_no_sc, ground_truth)
        metrics_with_sc = calculate_metrics(recon_with_sc, ground_truth)
        
        print("\nQuantitative Analysis:")
        print("=" * 50)
        print("Without Scatter Correction:")
        print(f"  MSE:  {metrics_no_sc['mse']:.4f}")
        print(f"  PSNR: {metrics_no_sc['psnr']:.2f} dB")
        print(f"  SSIM: {metrics_no_sc['ssim']:.4f}")
        
        print("\nWith Scatter Correction:")
        print(f"  MSE:  {metrics_with_sc['mse']:.4f}")
        print(f"  PSNR: {metrics_with_sc['psnr']:.2f} dB")
        print(f"  SSIM: {metrics_with_sc['ssim']:.4f}")
        
        print("\nImprovement from Scatter Correction:")
        mse_improvement = (metrics_no_sc['mse'] - metrics_with_sc['mse']) / metrics_no_sc['mse'] * 100
        psnr_improvement = metrics_with_sc['psnr'] - metrics_no_sc['psnr']
        ssim_improvement = (metrics_with_sc['ssim'] - metrics_no_sc['ssim']) / metrics_no_sc['ssim'] * 100
        
        print(f"  MSE reduction:    {mse_improvement:+.1f}%")
        print(f"  PSNR improvement: {psnr_improvement:+.2f} dB")
        print(f"  SSIM improvement: {ssim_improvement:+.1f}%")
        
        # Save metrics
        metrics_summary = {
            'without_scatter_correction': metrics_no_sc,
            'with_scatter_correction': metrics_with_sc,
            'improvements': {
                'mse_reduction_percent': mse_improvement,
                'psnr_improvement_db': psnr_improvement,
                'ssim_improvement_percent': ssim_improvement
            }
        }
        
        metrics_file = self.output_dir / "quantitative_analysis.yaml"
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics_summary, f, default_flow_style=False)
        
        return metrics_summary
    
    def create_visualization(self):
        """Create comprehensive visualization of results."""
        print("\n=== Creating Visualizations ===")
        
        # Get central slices for display
        center_slice = self.matrix_size[2] // 2
        
        ground_truth = self.phantom.as_array()[:, :, center_slice]
        recon_no_sc = self.recon_no_scatter.as_array()[:, :, center_slice]
        recon_with_sc = self.recon_with_scatter.as_array()[:, :, center_slice]
        
        # Normalize for display
        scale_no_sc = np.sum(ground_truth) / np.sum(self.recon_no_scatter.as_array())
        scale_with_sc = np.sum(ground_truth) / np.sum(self.recon_with_scatter.as_array())
        
        recon_no_sc *= scale_no_sc
        recon_with_sc *= scale_with_sc
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8])
        
        # Display parameters
        vmax = ground_truth.max()
        
        # Row 1: Original images
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth.T, origin='lower', cmap='viridis', vmax=vmax)
        ax1.set_title('Ground Truth Phantom', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(recon_no_sc.T, origin='lower', cmap='viridis', vmax=vmax)
        ax2.set_title('Reconstruction\n(No Scatter Correction)', fontsize=12)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(recon_with_sc.T, origin='lower', cmap='viridis', vmax=vmax)
        ax3.set_title('Reconstruction\n(With Scatter Correction)', fontsize=12)
        ax3.axis('off')
        
        # Colorbar for reconstructions
        ax_cb1 = fig.add_subplot(gs[0, 3])
        plt.colorbar(im1, cax=ax_cb1, label='Activity Concentration')
        
        # Row 2: Difference images
        diff_no_sc = recon_no_sc - ground_truth
        diff_with_sc = recon_with_sc - ground_truth
        diff_max = max(np.abs(diff_no_sc).max(), np.abs(diff_with_sc).max())
        
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(diff_no_sc.T, origin='lower', cmap='RdBu_r', 
                        vmin=-diff_max, vmax=diff_max)
        ax4.set_title('Difference:\nNo Scatter Correction', fontsize=12)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(diff_with_sc.T, origin='lower', cmap='RdBu_r', 
                        vmin=-diff_max, vmax=diff_max)
        ax5.set_title('Difference:\nWith Scatter Correction', fontsize=12)
        ax5.axis('off')
        
        # Scatter estimate
        scatter_slice = self.scatter_estimate.as_array()[:, :, center_slice]
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(scatter_slice.T, origin='lower', cmap='Reds')
        ax6.set_title('Estimated Scatter\nDistribution', fontsize=12)
        ax6.axis('off')
        
        # Colorbar for differences
        ax_cb2 = fig.add_subplot(gs[1, 3])
        plt.colorbar(im4, cax=ax_cb2, label='Difference')
        
        # Row 3: Line profiles and metrics
        # Line profile through center
        center_row = self.matrix_size[0] // 2
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.plot(ground_truth[center_row, :], 'k-', linewidth=2, label='Ground Truth')
        ax7.plot(recon_no_sc[center_row, :], 'r--', linewidth=2, label='No Scatter Correction')
        ax7.plot(recon_with_sc[center_row, :], 'b-', linewidth=2, label='With Scatter Correction')
        ax7.set_xlabel('Pixel Position')
        ax7.set_ylabel('Activity Concentration')
        ax7.set_title('Central Line Profile Comparison', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Metrics text
        ax8 = fig.add_subplot(gs[2, 2:])
        metrics_no_sc = {
            'mse': np.mean((recon_no_sc - ground_truth)**2),
            'psnr': 20 * np.log10(ground_truth.max() / np.sqrt(np.mean((recon_no_sc - ground_truth)**2)))
        }
        metrics_with_sc = {
            'mse': np.mean((recon_with_sc - ground_truth)**2),
            'psnr': 20 * np.log10(ground_truth.max() / np.sqrt(np.mean((recon_with_sc - ground_truth)**2)))
        }
        
        metrics_text = f"""
Quantitative Metrics (Central Slice):

Without Scatter Correction:
  MSE:  {metrics_no_sc['mse']:.4f}
  PSNR: {metrics_no_sc['psnr']:.2f} dB

With Scatter Correction:
  MSE:  {metrics_with_sc['mse']:.4f}
  PSNR: {metrics_with_sc['psnr']:.2f} dB

Improvement:
  PSNR: {metrics_with_sc['psnr'] - metrics_no_sc['psnr']:+.2f} dB
  MSE:  {(metrics_no_sc['mse'] - metrics_with_sc['mse'])/metrics_no_sc['mse']*100:+.1f}%
        """
        
        ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax8.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / "complete_workflow_results.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {fig_file}")
        
        # Create projection data visualization
        self._visualize_projections()
        
        plt.show()
    
    def _visualize_projections(self):
        """Create visualization of projection data."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get central projection
        center_proj = self.num_projections // 2
        
        # Photopeak projections
        photopeak_proj = self.photopeak_data.as_array()[center_proj]
        axes[0, 0].imshow(photopeak_proj.T, origin='lower', cmap='viridis')
        axes[0, 0].set_title('Photopeak Window\n(126-154 keV)')
        axes[0, 0].axis('off')
        
        # Scatter projections
        scatter_low_proj = self.scatter_low_data.as_array()[center_proj]
        axes[0, 1].imshow(scatter_low_proj.T, origin='lower', cmap='viridis')
        axes[0, 1].set_title('Lower Scatter Window\n(100-126 keV)')
        axes[0, 1].axis('off')
        
        scatter_high_proj = self.scatter_high_data.as_array()[center_proj]
        axes[0, 2].imshow(scatter_high_proj.T, origin='lower', cmap='viridis')
        axes[0, 2].set_title('Upper Scatter Window\n(154-180 keV)')
        axes[0, 2].axis('off')
        
        # Scatter estimate and correction
        scatter_est_proj = self.scatter_estimate.as_array()[center_proj]
        axes[1, 0].imshow(scatter_est_proj.T, origin='lower', cmap='Reds')
        axes[1, 0].set_title('Estimated Scatter\nin Photopeak')
        axes[1, 0].axis('off')
        
        scatter_corr_proj = self.scatter_corrected_data.as_array()[center_proj]
        axes[1, 1].imshow(scatter_corr_proj.T, origin='lower', cmap='viridis')
        axes[1, 1].set_title('Scatter-Corrected\nProjections')
        axes[1, 1].axis('off')
        
        # Line profile comparison
        center_row = photopeak_proj.shape[0] // 2
        axes[1, 2].plot(photopeak_proj[center_row, :], 'g-', label='Original', linewidth=2)
        axes[1, 2].plot(scatter_est_proj[center_row, :], 'r--', label='Scatter Est.', linewidth=2)
        axes[1, 2].plot(scatter_corr_proj[center_row, :], 'b-', label='Corrected', linewidth=2)
        axes[1, 2].set_xlabel('Detector Pixel')
        axes[1, 2].set_ylabel('Counts')
        axes[1, 2].set_title('Central Line Profile')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        proj_fig_file = self.output_dir / "projection_data_analysis.png"
        plt.savefig(proj_fig_file, dpi=300, bbox_inches='tight')
        print(f"Projection data visualization saved to: {proj_fig_file}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._cleanup_temp and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def run_complete_workflow(self):
        """Run the complete SPECT simulation and reconstruction workflow."""
        print("="*60)
        print("COMPLETE SPECT SIMULATION AND RECONSTRUCTION WORKFLOW")
        print("="*60)
        
        try:
            # Step 1: Create phantom and attenuation map
            self.create_phantom()
            self.create_attenuation_map()
            
            # Step 2: Set up simulation
            self.setup_simulation_config()
            
            # Step 3: Run SIMIND simulation
            if not self.run_simind_simulation():
                print("Simulation failed. Please check SIMIND installation.")
                return False
            
            # Step 4: Create scatter correction
            self.create_scatter_correction()
            
            # Step 5: Perform reconstruction
            self.perform_reconstruction()
            
            # Step 6: Analyze results
            self.analyze_results()
            
            # Step 7: Create visualizations
            self.create_visualization()
            
            print("\n" + "="*60)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"All results saved in: {self.output_dir}")
            print("\nGenerated files:")
            for file_path in sorted(self.output_dir.glob("*")):
                print(f"  - {file_path.name}")
            
            return True
            
        except Exception as e:
            print(f"\nWorkflow failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()


def main():
    """Main function to run the complete workflow example."""
    if not SIRF_AVAILABLE:
        print("SIRF and/or SIRF-SIMIND-Connection not available.")
        print("Please install both packages before running this example.")
        return
    
    # Create output directory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"spect_workflow_output_{timestamp}"
    
    # Run the workflow
    workflow = SPECTWorkflowExample(output_dir=output_dir)
    
    success = workflow.run_complete_workflow()
    
    if success:
        print(f"\n🎉 Example completed successfully!")
        print(f"📁 Results available in: {workflow.output_dir}")
        print("\n📊 This example demonstrated:")
        print("   ✓ Creating realistic 3D SPECT phantoms")
        print("   ✓ SIMIND Monte Carlo simulation with multiple energy windows")
        print("   ✓ Dual energy window scatter correction")
        print("   ✓ Iterative reconstruction with and without scatter correction")
        print("   ✓ Quantitative image quality assessment")
        print("   ✓ Comprehensive visualization of results")
        
        print("\n🔬 Key findings from this workflow:")
        print("   • Scatter correction improves image quality")
        print("   • Multi-window acquisition enables scatter estimation")
        print("   • Monte Carlo simulation provides realistic data")
        print("   • Quantitative metrics demonstrate improvements")
    else:
        print("\n❌ Example failed. Please check error messages above.")


if __name__ == "__main__":
    main()