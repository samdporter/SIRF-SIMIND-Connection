#!/usr/bin/env python3
"""
generate_test_data.py - Generate test data for SIRF-SIMIND-Connection

This script generates various test phantoms, attenuation maps, and synthetic data
for testing the SIRF-SIMIND-Connection package.

Usage:
    python generate_test_data.py --output-dir test_data --format hv
    python generate_test_data.py --help

Author: SIRF-SIMIND-Connection Team
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import time
from scipy import ndimage
from typing import Tuple, List, Dict, Optional

try:
    from sirf.STIR import ImageData, AcquisitionData
    SIRF_AVAILABLE = True
except ImportError:
    print("Warning: SIRF not available. Will generate numpy arrays only.")
    SIRF_AVAILABLE = False


class TestDataGenerator:
    """Generator for various test datasets used in SPECT imaging."""
    
    def __init__(self, output_dir: str = "test_data"):
        """Initialize the test data generator.
        
        Args:
            output_dir: Directory to save generated test data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Test data will be saved to: {self.output_dir}")
    
    def generate_all_phantoms(self) -> Dict[str, str]:
        """Generate all standard test phantoms.
        
        Returns:
            Dictionary mapping phantom names to file paths
        """
        phantom_files = {}
        
        print("\n=== Generating Test Phantoms ===")
        
        # 1. Simple geometric phantoms
        phantom_files['sphere'] = self.generate_sphere_phantom()
        phantom_files['cylinder'] = self.generate_cylinder_phantom()
        phantom_files['hot_rods'] = self.generate_hot_rod_phantom()
        
        # 2. Anthropomorphic phantoms
        phantom_files['cardiac'] = self.generate_cardiac_phantom()
        phantom_files['torso'] = self.generate_torso_phantom()
        phantom_files['brain'] = self.generate_brain_phantom()
        
        # 3. Quality assurance phantoms
        phantom_files['uniform'] = self.generate_uniform_phantom()
        phantom_files['line_sources'] = self.generate_line_source_phantom()
        phantom_files['resolution'] = self.generate_resolution_phantom()
        
        return phantom_files
    
    def generate_sphere_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                               voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate a simple sphere phantom.
        
        Args:
            dimensions: Image dimensions (x, y, z)
            voxel_size: Voxel spacing in cm
            
        Returns:
            Path to saved phantom file
        """
        print("Generating sphere phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Central sphere
        sphere_radius = min(dimensions[0], dimensions[1]) // 4
        sphere_mask = x**2 + y**2 + z**2 <= sphere_radius**2
        phantom[sphere_mask] = 10.0
        
        # Background
        background_radius = min(dimensions[0], dimensions[1]) // 2.5
        background_mask = x**2 + y**2 <= background_radius**2
        phantom[background_mask] = np.maximum(phantom[background_mask], 1.0)
        
        filename = "sphere_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_cylinder_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                 voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate a cylindrical phantom with hot spots."""
        print("Generating cylinder phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Main cylinder
        cylinder_radius = min(dimensions[0], dimensions[1]) // 3
        cylinder_mask = x**2 + y**2 <= cylinder_radius**2
        phantom[cylinder_mask] = 2.0
        
        # Hot spots
        hot_spot_centers = [
            (cylinder_radius//2, 0, 0),
            (-cylinder_radius//2, 0, 0),
            (0, cylinder_radius//2, 0),
            (0, -cylinder_radius//2, 0),
            (0, 0, dimensions[2]//4),
            (0, 0, -dimensions[2]//4)
        ]
        
        for center in hot_spot_centers:
            hot_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 36
            phantom[hot_mask] = 8.0
        
        filename = "cylinder_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_hot_rod_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate hot rod phantom for resolution testing."""
        print("Generating hot rod phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Background cylinder
        bg_radius = min(dimensions[0], dimensions[1]) // 2.5
        bg_mask = x**2 + y**2 <= bg_radius**2
        phantom[bg_mask] = 1.0
        
        # Hot rods with different diameters
        rod_diameters = [2, 3, 4, 6, 8, 10]  # in voxels
        rod_centers_y = np.linspace(-bg_radius*0.7, bg_radius*0.7, len(rod_diameters))
        
        for i, (diameter, center_y) in enumerate(zip(rod_diameters, rod_centers_y)):
            # Multiple rods of same diameter
            for j in range(3):
                center_x = -bg_radius*0.5 + j * bg_radius*0.5
                rod_mask = ((x - center_x)**2 + (y - center_y)**2) <= (diameter/2)**2
                phantom[rod_mask] = 5.0
        
        filename = "hot_rod_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_cardiac_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate a realistic cardiac phantom."""
        print("Generating cardiac phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Torso outline
        torso_radius_x = dimensions[0] // 2.5
        torso_radius_y = dimensions[1] // 2.8
        torso_mask = (x/torso_radius_x)**2 + (y/torso_radius_y)**2 <= 1
        phantom[torso_mask] = 0.5  # Background activity
        
        # Heart (left ventricle)
        heart_center = (-dimensions[0]//6, dimensions[1]//8, 0)
        heart_radius_x, heart_radius_y, heart_radius_z = 12, 10, 15
        
        # LV wall (thick)
        lv_outer = ((x - heart_center[0])/heart_radius_x)**2 + \
                   ((y - heart_center[1])/heart_radius_y)**2 + \
                   ((z - heart_center[2])/heart_radius_z)**2 <= 1
        
        lv_inner = ((x - heart_center[0])/(heart_radius_x-3))**2 + \
                   ((y - heart_center[1])/(heart_radius_y-3))**2 + \
                   ((z - heart_center[2])/(heart_radius_z-3))**2 <= 1
        
        lv_wall = lv_outer & (~lv_inner)
        phantom[lv_wall] = 8.0  # High cardiac uptake
        
        # Defect (perfusion defect)
        defect_center = (heart_center[0] + 5, heart_center[1], heart_center[2])
        defect_mask = ((x - defect_center[0])**2 + (y - defect_center[1])**2 + 
                      (z - defect_center[2])**2) <= 25
        phantom[defect_mask & lv_wall] = 2.0  # Reduced uptake
        
        # Liver
        liver_center = (dimensions[0]//4, -dimensions[1]//6, -dimensions[2]//4)
        liver_radius_x, liver_radius_y, liver_radius_z = 18, 12, 20
        liver_mask = ((x - liver_center[0])/liver_radius_x)**2 + \
                     ((y - liver_center[1])/liver_radius_y)**2 + \
                     ((z - liver_center[2])/liver_radius_z)**2 <= 1
        phantom[liver_mask] = 3.0
        
        # Smooth the phantom
        phantom = ndimage.gaussian_filter(phantom, sigma=0.8)
        
        filename = "cardiac_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_torso_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                              voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate a full torso phantom."""
        print("Generating torso phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Torso outline (elliptical)
        torso_a, torso_b = dimensions[0]//2.2, dimensions[1]//2.5
        torso_mask = (x/torso_a)**2 + (y/torso_b)**2 <= 1
        phantom[torso_mask] = 1.0
        
        # Lungs (lower activity regions)
        lung_centers = [(-dimensions[0]//4, dimensions[1]//4, 0), 
                       (dimensions[0]//4, dimensions[1]//4, 0)]
        for center in lung_centers:
            lung_mask = ((x - center[0])/15)**2 + ((y - center[1])/12)**2 + (z/18)**2 <= 1
            phantom[lung_mask] = 0.3
        
        # Spine
        spine_mask = (x**2 + (y + dimensions[1]//2.5)**2) <= 16
        phantom[spine_mask] = 1.5
        
        # Kidneys
        kidney_centers = [(0, -dimensions[1]//3, -dimensions[2]//4),
                         (0, -dimensions[1]//3, dimensions[2]//4)]
        for center in kidney_centers:
            kidney_mask = ((x - center[0])/6)**2 + ((y - center[1])/8)**2 + ((z - center[2])/10)**2 <= 1
            phantom[kidney_mask] = 4.0
        
        # Liver
        liver_center = (dimensions[0]//3, -dimensions[1]//8, 0)
        liver_mask = ((x - liver_center[0])/20)**2 + ((y - liver_center[1])/15)**2 + (z/20)**2 <= 1
        phantom[liver_mask] = 2.5
        
        # Heart
        heart_center = (-dimensions[0]//6, dimensions[1]//8, 0)
        heart_mask = ((x - heart_center[0])/10)**2 + ((y - heart_center[1])/8)**2 + (z/12)**2 <= 1
        phantom[heart_mask] = 6.0
        
        # Smooth
        phantom = ndimage.gaussian_filter(phantom, sigma=1.0)
        
        filename = "torso_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_brain_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                              voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate a simplified brain phantom."""
        print("Generating brain phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Brain outline
        brain_radius = min(dimensions[0], dimensions[1]) // 2.3
        brain_mask = x**2 + y**2 + (z*1.2)**2 <= brain_radius**2
        phantom[brain_mask] = 2.0  # Gray matter
        
        # White matter (inner region)
        wm_radius = brain_radius * 0.7
        wm_mask = x**2 + y**2 + (z*1.2)**2 <= wm_radius**2
        phantom[wm_mask] = 1.5
        
        # Ventricles (low activity)
        ventricle_centers = [(-10, 0, 0), (10, 0, 0)]
        for center in ventricle_centers:
            vent_mask = ((x - center[0])**2 + y**2 + z**2) <= 64
            phantom[vent_mask] = 0.2
        
        # High activity regions (basal ganglia)
        bg_centers = [(-8, -8, 0), (8, -8, 0)]
        for center in bg_centers:
            bg_mask = ((x - center[0])**2 + (y - center[1])**2 + z**2) <= 25
            phantom[bg_mask] = 4.0
        
        filename = "brain_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_uniform_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                                activity: float = 5.0) -> str:
        """Generate uniform phantom for calibration."""
        print("Generating uniform phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Uniform cylinder
        radius = min(dimensions[0], dimensions[1]) // 2.5
        uniform_mask = x**2 + y**2 <= radius**2
        phantom[uniform_mask] = activity
        
        filename = "uniform_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_line_source_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate line source phantom for system resolution."""
        print("Generating line source phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Multiple line sources at different positions
        line_positions = [-30, -15, 0, 15, 30]
        
        for pos in line_positions:
            # Vertical line
            line_mask = (x**2 + (y - pos)**2) <= 1
            phantom[line_mask] = 10.0
            
            # Horizontal line
            line_mask = ((x - pos)**2 + y**2) <= 1
            phantom[line_mask] = 10.0
        
        filename = "line_source_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_resolution_phantom(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                   voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate resolution phantom with various sized features."""
        print("Generating resolution phantom...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        phantom = np.zeros(dimensions)
        
        # Background
        bg_radius = min(dimensions[0], dimensions[1]) // 2.5
        bg_mask = x**2 + y**2 <= bg_radius**2
        phantom[bg_mask] = 1.0
        
        # Small spheres of different sizes
        sphere_radii = [1, 2, 3, 4, 5, 6, 8, 10]
        angles = np.linspace(0, 2*np.pi, len(sphere_radii), endpoint=False)
        circle_radius = bg_radius * 0.6
        
        for radius, angle in zip(sphere_radii, angles):
            center_x = circle_radius * np.cos(angle)
            center_y = circle_radius * np.sin(angle)
            sphere_mask = ((x - center_x)**2 + (y - center_y)**2 + z**2) <= radius**2
            phantom[sphere_mask] = 8.0
        
        filename = "resolution_phantom"
        filepath = self._save_image_data(phantom, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_attenuation_maps(self) -> Dict[str, str]:
        """Generate attenuation maps corresponding to phantoms."""
        print("\n=== Generating Attenuation Maps ===")
        
        attenuation_files = {}
        
        # Standard attenuation maps
        attenuation_files['uniform_water'] = self.generate_uniform_attenuation()
        attenuation_files['soft_tissue'] = self.generate_soft_tissue_attenuation()
        attenuation_files['anthropomorphic'] = self.generate_anthropomorphic_attenuation()
        
        return attenuation_files
    
    def generate_uniform_attenuation(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                   voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                                   mu_value: float = 0.15) -> str:
        """Generate uniform attenuation map."""
        print("Generating uniform attenuation map...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        mu_map = np.zeros(dimensions)
        
        # Uniform cylinder
        radius = min(dimensions[0], dimensions[1]) // 2.5
        uniform_mask = x**2 + y**2 <= radius**2
        mu_map[uniform_mask] = mu_value  # cm^-1 for soft tissue at 140 keV
        
        filename = "uniform_attenuation_map"
        filepath = self._save_image_data(mu_map, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_soft_tissue_attenuation(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                        voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate soft tissue attenuation map with bone inserts."""
        print("Generating soft tissue attenuation map...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        mu_map = np.zeros(dimensions)
        
        # Soft tissue background
        tissue_radius = min(dimensions[0], dimensions[1]) // 2.5
        tissue_mask = x**2 + y**2 <= tissue_radius**2
        mu_map[tissue_mask] = 0.15  # cm^-1 for soft tissue
        
        # Bone inserts
        bone_centers = [(0, -tissue_radius*0.7, 0), (tissue_radius*0.5, 0, 0)]
        for center in bone_centers:
            bone_mask = ((x - center[0])**2 + (y - center[1])**2 + z**2) <= 64
            mu_map[bone_mask] = 0.4  # cm^-1 for bone
        
        # Lung regions (lower attenuation)
        lung_centers = [(-tissue_radius*0.3, tissue_radius*0.3, 0),
                       (tissue_radius*0.3, tissue_radius*0.3, 0)]
        for center in lung_centers:
            lung_mask = ((x - center[0])/8)**2 + ((y - center[1])/10)**2 + (z/12)**2 <= 1
            mu_map[lung_mask] = 0.05  # cm^-1 for lung
        
        filename = "soft_tissue_attenuation_map"
        filepath = self._save_image_data(mu_map, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_anthropomorphic_attenuation(self, dimensions: Tuple[int, int, int] = (128, 128, 64),
                                            voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> str:
        """Generate anthropomorphic attenuation map."""
        print("Generating anthropomorphic attenuation map...")
        
        x, y, z = self._create_coordinate_grids(dimensions)
        mu_map = np.zeros(dimensions)
        
        # Torso outline
        torso_a, torso_b = dimensions[0]//2.2, dimensions[1]//2.5
        torso_mask = (x/torso_a)**2 + (y/torso_b)**2 <= 1
        mu_map[torso_mask] = 0.15  # Soft tissue
        
        # Lungs
        lung_centers = [(-dimensions[0]//4, dimensions[1]//4, 0),
                       (dimensions[0]//4, dimensions[1]//4, 0)]
        for center in lung_centers:
            lung_mask = ((x - center[0])/15)**2 + ((y - center[1])/12)**2 + (z/18)**2 <= 1
            mu_map[lung_mask] = 0.05
        
        # Spine and ribs
        spine_mask = (x**2 + (y + dimensions[1]//2.5)**2) <= 16
        mu_map[spine_mask] = 0.4
        
        # Ribs (simplified)
        for rib_y in [-15, -5, 5, 15]:
            for angle in np.linspace(0.3, 2.8, 10):
                rib_x = int(20 * np.cos(angle))
                rib_y_pos = int(rib_y + 3 * np.sin(angle))
                if abs(rib_x) < dimensions[0]//2 and abs(rib_y_pos) < dimensions[1]//2:
                    rib_mask = ((x - rib_x)**2 + (y - rib_y_pos)**2) <= 4
                    mu_map[rib_mask] = 0.35
        
        # Smooth
        mu_map = ndimage.gaussian_filter(mu_map, sigma=0.5)
        
        filename = "anthropomorphic_attenuation_map"
        filepath = self._save_image_data(mu_map, dimensions, voxel_size, filename)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def generate_synthetic_projections(self, num_projections: int = 60,
                                     detector_size: Tuple[int, int] = (128, 128)) -> str:
        """Generate synthetic projection data."""
        print("\n=== Generating Synthetic Projection Data ===")
        print("Generating synthetic projections...")
        
        # Create realistic projection data with noise
        projections = np.zeros((num_projections, detector_size[0], detector_size[1]))
        
        for proj_idx in range(num_projections):
            angle = proj_idx * 2 * np.pi / num_projections
            
            # Simple geometric projection
            for i in range(detector_size[0]):
                for j in range(detector_size[1]):
                    # Distance from center
                    dist_from_center = np.sqrt((i - detector_size[0]//2)**2 + 
                                             (j - detector_size[1]//2)**2)
                    
                    # Basic projection pattern
                    base_counts = 1000 * np.exp(-0.01 * dist_from_center**2)
                    
                    # Add angular variation
                    angular_variation = 1 + 0.3 * np.sin(angle + 0.1 * dist_from_center)
                    
                    # Add Poisson noise
                    expected_counts = base_counts * angular_variation
                    projections[proj_idx, i, j] = np.random.poisson(max(0, expected_counts))
        
        # Save as numpy array
        proj_file = self.output_dir / "synthetic_projections.npy"
        np.save(proj_file, projections)
        print(f"  Saved: {proj_file}")
        
        # Also save metadata
        metadata = {
            'num_projections': num_projections,
            'detector_size': detector_size,
            'angular_range': 360.0,
            'total_counts': int(projections.sum()),
            'description': 'Synthetic SPECT projection data with Poisson noise'
        }
        
        metadata_file = self.output_dir / "synthetic_projections_metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        
        return str(proj_file)
    
    def generate_test_configurations(self) -> Dict[str, str]:
        """Generate test configuration files."""
        print("\n=== Generating Test Configurations ===")
        
        config_files = {}
        
        # Basic SIMIND configuration
        config_files['basic_simind'] = self._create_basic_smc_config()
        config_files['multi_window'] = self._create_multi_window_config()
        config_files['scanner_config'] = self._create_scanner_config()
        
        return config_files
    
    def _create_basic_smc_config(self) -> str:
        """Create basic SIMIND .smc configuration."""
        print("Generating basic SIMIND configuration...")
        
        smc_content = """
! SIMIND Control File - Basic Test Configuration
! Generated by SIRF-SIMIND-Connection test data generator
TITLE Basic Test Simulation
PHOTONS 1000000
SPECTRUM MONO 140.0
DETECTOR NaI 0.95
MATRIX 128 128
PIXEL_SIZE 0.4 0.4
PROJECTIONS 60
ORBIT CIRCULAR
START_ANGLE 0.0
STOP_ANGLE 360.0
ENERGY_WINDOW 1 126.0 154.0
OUTPUT_FORMAT STIR
SAVE_TOTAL 1
SAVE_SCATTER 0
"""
        
        config_file = self.output_dir / "basic_test_config.smc"
        config_file.write_text(smc_content.strip())
        print(f"  Saved: {config_file}")
        return str(config_file)
    
    def _create_multi_window_config(self) -> str:
        """Create multi-window configuration."""
        print("Generating multi-window configuration...")
        
        smc_content = """
! SIMIND Control File - Multi-Window Test Configuration
TITLE Multi-Window Test Simulation
PHOTONS 2000000
SPECTRUM MONO 140.0
DETECTOR NaI 0.95
MATRIX 128 128
PIXEL_SIZE 0.4 0.4
PROJECTIONS 120
ORBIT CIRCULAR
START_ANGLE 0.0
STOP_ANGLE 360.0
! Energy windows for scatter correction
ENERGY_WINDOW 1 126.0 154.0  ! Photopeak
ENERGY_WINDOW 2 100.0 126.0  ! Lower scatter
ENERGY_WINDOW 3 154.0 180.0  ! Upper scatter
OUTPUT_FORMAT STIR
SAVE_TOTAL 1
SAVE_SCATTER 1
"""
        
        config_file = self.output_dir / "multi_window_config.smc"
        config_file.write_text(smc_content.strip())
        print(f"  Saved: {config_file}")
        return str(config_file)
    
    def _create_scanner_config(self) -> str:
        """Create YAML scanner configuration."""
        print("Generating scanner configuration...")
        
        scanner_config = {
            'scanner': {
                'name': 'test_scanner',
                'type': 'dual_head_spect',
                'manufacturer': 'Generic',
                'model': 'Test Scanner v1.0',
                'detector': {
                    'material': 'NaI',
                    'thickness': 0.95,
                    'crystal_x': 40.0,
                    'crystal_y': 30.0,
                    'pixel_size': [0.4, 0.4],
                    'matrix_size': [128, 128],
                    'intrinsic_resolution': 3.5
                },
                'collimator': {
                    'type': 'LEHR',
                    'hole_diameter': 0.11,
                    'septal_thickness': 0.016,
                    'hole_length': 2.4,
                    'sensitivity': 120.0
                },
                'gantry': {
                    'radius': 15.0,
                    'angular_range': 360.0,
                    'num_projections': 120,
                    'rotation_speed': 3.0
                }
            },
            'simulation': {
                'number_of_photons': 2000000,
                'random_seed': 42,
                'detector_binning': [128, 128],
                'voxel_size': [0.4, 0.4, 0.4],
                'output_format': 'STIR',
                'save_intermediate': False
            },
            'physics': {
                'scatter_modeling': True,
                'attenuation_correction': True,
                'detector_response': True,
                'collimator_response': True
            },
            'energy_windows': {
                'tc99m_photopeak': {'lower': 126, 'upper': 154},
                'tc99m_scatter_low': {'lower': 100, 'upper': 126},
                'tc99m_scatter_high': {'lower': 154, 'upper': 180}
            }
        }
        
        config_file = self.output_dir / "test_scanner_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(scanner_config, f, default_flow_style=False, indent=2)
        
        print(f"  Saved: {config_file}")
        return str(config_file)
    
    def create_visualization_summary(self, phantom_files: Dict[str, str],
                                   attenuation_files: Dict[str, str]) -> str:
        """Create visualization summary of all generated data."""
        print("\n=== Creating Visualization Summary ===")
        
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))
        fig.suptitle('SIRF-SIMIND-Connection Test Data Summary', fontsize=16, fontweight='bold')
        
        # Plot phantoms
        phantom_idx = 0
        for name, filepath in phantom_files.items():
            if phantom_idx >= 12:  # Limit to available subplot space
                break
            
            row = phantom_idx // 6
            col = phantom_idx % 6
            
            try:
                if SIRF_AVAILABLE and filepath.endswith('.hv'):
                    # Load SIRF image
                    img = ImageData()
                    img.read(filepath)
                    data = img.as_array()
                else:
                    # Load numpy array
                    data = np.load(filepath.replace('.hv', '.npy'))
                
                # Show central slice
                center_slice = data.shape[2] // 2
                axes[row, col].imshow(data[:, :, center_slice].T, origin='lower', cmap='viridis')
                axes[row, col].set_title(f'{name.replace("_", " ").title()}', fontsize=10)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading\n{name}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
            
            phantom_idx += 1
        
        # Plot attenuation maps
        for name, filepath in attenuation_files.items():
            if phantom_idx >= 24:
                break
            
            row = phantom_idx // 6
            col = phantom_idx % 6
            
            try:
                if SIRF_AVAILABLE and filepath.endswith('.hv'):
                    img = ImageData()
                    img.read(filepath)
                    data = img.as_array()
                else:
                    data = np.load(filepath.replace('.hv', '.npy'))
                
                center_slice = data.shape[2] // 2
                axes[row, col].imshow(data[:, :, center_slice].T, origin='lower', cmap='Reds')
                axes[row, col].set_title(f'{name.replace("_", " ").title()}\n(Attenuation)', fontsize=10)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading\n{name}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
            
            phantom_idx += 1
        
        # Hide unused subplots
        for idx in range(phantom_idx, 24):
            row = idx // 6
            col = idx % 6
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        summary_file = self.output_dir / "test_data_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        print(f"Visualization summary saved: {summary_file}")
        
        plt.close()
        return str(summary_file)
    
    def _create_coordinate_grids(self, dimensions: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create coordinate grids centered at origin."""
        x, y, z = np.meshgrid(
            np.linspace(-dimensions[0]//2, dimensions[0]//2, dimensions[0]),
            np.linspace(-dimensions[1]//2, dimensions[1]//2, dimensions[1]),
            np.linspace(-dimensions[2]//2, dimensions[2]//2, dimensions[2]),
            indexing='ij'
        )
        return x, y, z
    
    def _save_image_data(self, data_array: np.ndarray, dimensions: Tuple[int, int, int],
                        voxel_size: Tuple[float, float, float], filename: str) -> Path:
        """Save image data in appropriate format."""
        if SIRF_AVAILABLE:
            # Save as SIRF ImageData
            img = ImageData()
            img.initialise(dimensions)
            img.fill(data_array)
            img.set_voxel_spacing(voxel_size)
            
            filepath = self.output_dir / f"{filename}.hv"
            img.write(str(filepath))
        else:
            # Save as numpy array
            filepath = self.output_dir / f"{filename}.npy"
            np.save(filepath, data_array)
            
            # Save metadata
            metadata = {
                'dimensions': dimensions,
                'voxel_size': voxel_size,
                'data_type': str(data_array.dtype),
                'min_value': float(data_array.min()),
                'max_value': float(data_array.max()),
                'total_activity': float(data_array.sum())
            }
            
            metadata_file = self.output_dir / f"{filename}_metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
        
        return filepath


def main():
    """Main function to generate all test data."""
    parser = argparse.ArgumentParser(description="Generate test data for SIRF-SIMIND-Connection")
    parser.add_argument("--output-dir", default="test_data", help="Output directory for test data")
    parser.add_argument("--phantoms", action="store_true", help="Generate phantoms only")
    parser.add_argument("--attenuation", action="store_true", help="Generate attenuation maps only")
    parser.add_argument("--projections", action="store_true", help="Generate synthetic projections only")
    parser.add_argument("--configs", action="store_true", help="Generate configurations only")
    parser.add_argument("--all", action="store_true", default=True, help="Generate all data (default)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization summary")
    
    args = parser.parse_args()
    
    # Override default if specific options are given
    if any([args.phantoms, args.attenuation, args.projections, args.configs]):
        args.all = False
    
    print("SIRF-SIMIND-Connection Test Data Generator")
    print("=" * 50)
    
    generator = TestDataGenerator(args.output_dir)
    
    phantom_files = {}
    attenuation_files = {}
    
    start_time = time.time()
    
    if args.all or args.phantoms:
        phantom_files = generator.generate_all_phantoms()
    
    if args.all or args.attenuation:
        attenuation_files = generator.generate_attenuation_maps()
    
    if args.all or args.projections:
        generator.generate_synthetic_projections()
    
    if args.all or args.configs:
        generator.generate_test_configurations()
    
    if not args.no_viz and (phantom_files or attenuation_files):
        generator.create_visualization_summary(phantom_files, attenuation_files)
    
    generation_time = time.time() - start_time
    
    # Create summary report
    summary = {
        'generation_time': f"{generation_time:.2f} seconds",
        'output_directory': str(generator.output_dir),
        'phantoms_generated': len(phantom_files),
        'attenuation_maps_generated': len(attenuation_files),
        'sirf_available': SIRF_AVAILABLE,
        'generated_files': sorted([f.name for f in generator.output_dir.glob("*")])
    }
    
    summary_file = generator.output_dir / "generation_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print("\n" + "=" * 50)
    print("TEST DATA GENERATION COMPLETED")
    print("=" * 50)
    print(f"Total time: {generation_time:.2f} seconds")
    print(f"Output directory: {generator.output_dir}")
    print(f"Generated {len(phantom_files)} phantoms")
    print(f"Generated {len(attenuation_files)} attenuation maps")
    print(f"Summary saved to: {summary_file}")
    
    # List all generated files
    print("\nGenerated files:")
    for file_path in sorted(generator.output_dir.glob("*")):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()