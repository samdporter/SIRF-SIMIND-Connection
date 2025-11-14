"""
Output processing for SIMIND simulation results.

This module handles post-processing of SIMIND outputs for both scattwin
and penetrate scoring routines, including conversion to STIR format and
validation of geometry parameters.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from sirf_simind_connection.utils.import_helpers import get_sirf_types

_, AcquisitionData, SIRF_AVAILABLE = get_sirf_types()

# Import backend interfaces and factories using centralized access
from sirf_simind_connection.utils.backend_access import BACKEND_AVAILABLE, BACKENDS
from sirf_simind_connection.utils.stir_utils import extract_attributes_from_stir

# Unpack interfaces needed by output processor
create_acquisition_data = BACKENDS.factories.create_acquisition_data
ensure_acquisition_interface = BACKENDS.wrappers.ensure_acquisition_interface
AcquisitionDataInterface = BACKENDS.types.AcquisitionDataInterface

# Import types
from .types import OutputError, PenetrateOutputType, ScoringRoutine


class OutputProcessor:
    """Enhanced output processor that handles both scattwin and penetrate outputs."""

    def __init__(self, converter, output_dir: Path):
        """
        Initialize the output processor.

        Args:
            converter: SIMIND to STIR converter instance
            output_dir: Directory containing simulation outputs
        """
        self.converter = converter
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def process_outputs(
        self,
        output_prefix: str,
        template_sinogram_path: Optional[str] = None,
        source=None,
        scoring_routine: ScoringRoutine = ScoringRoutine.SCATTWIN,
        preferred_backend: Optional[str] = None,
    ) -> Dict:
        """
        Process outputs based on the scoring routine used.

        Args:
            output_prefix: Prefix used for output files
            template_sinogram_path: Path to template sinogram header file (.hs)
            source: Source image for geometry reference (backend-agnostic)
            scoring_routine: Scoring routine that was used
            preferred_backend: Backend hint when wrapping acquired data

        Returns:
            Dictionary of output name -> AcquisitionDataInterface
        """

        if scoring_routine == ScoringRoutine.SCATTWIN:
            return self._process_scattwin_outputs(
                output_prefix, template_sinogram_path, source, preferred_backend
            )
        elif scoring_routine == ScoringRoutine.PENETRATE:
            return self._process_penetrate_outputs(
                output_prefix, template_sinogram_path, source, preferred_backend
            )
        else:
            raise ValueError(
                f"Unsupported scoring routine for output processing: {scoring_routine}"
            )

    def _process_scattwin_outputs(
        self,
        output_prefix: str,
        template_sinogram_path: Optional[str],
        source,
        preferred_backend: Optional[str],
    ) -> Dict:
        """Process scattwin routine outputs (existing functionality).

        Args:
            output_prefix: Prefix used for output files
            template_sinogram_path: Path to template sinogram header file
            source: Source image object (backend-agnostic)

        Returns:
            Dictionary of output name -> AcquisitionDataInterface
        """

        h00_files = self._find_scattwin_output_files(output_prefix)

        if not h00_files:
            raise OutputError("No SIMIND scattwin output files found")

        # Process each file
        for h00_file in h00_files:
            self._process_single_scattwin_file(h00_file, template_sinogram_path, source)

        # Load and organize converted files
        return self._load_converted_scattwin_files(output_prefix, preferred_backend)

    def _process_penetrate_outputs(
        self,
        output_prefix,
        template_sinogram,
        source,
        preferred_backend: Optional[str],
    ):
        # Find the single .h00 file from penetrate routine
        h00_file = self.converter.find_penetrate_h00_file(
            output_prefix, str(self.output_dir)
        )

        if not h00_file:
            raise OutputError("No penetrate .h00 file found")

        # Create multiple .hs files, one for each .bXX file
        outputs = self.converter.create_penetrate_headers_from_template(
            h00_file, output_prefix, str(self.output_dir)
        )

        if not outputs:
            raise OutputError("No penetrate output files found")

        if preferred_backend and ensure_acquisition_interface is not None:
            return {
                name: ensure_acquisition_interface(
                    data, preferred_backend=preferred_backend
                )
                for name, data in outputs.items()
            }
        return outputs

    def _find_scattwin_output_files(self, output_prefix: str) -> List[Path]:
        """Find SIMIND scattwin output files."""
        scatter_types = ["_air_w", "_sca_w", "_tot_w", "_pri_w"]
        return [
            f
            for f in self.output_dir.glob("*.h00")
            if any(s in f.name for s in scatter_types) and output_prefix in f.name
        ]

    def _process_single_scattwin_file(
        self,
        h00_file: Path,
        template_sinogram_path: Optional[str],
        source,
    ) -> None:
        """Process a single scattwin output file with corrections.

        Args:
            h00_file: Path to SIMIND output file
            template_sinogram_path: Path to template sinogram header
            source: Source image object (backend-agnostic)
        """
        try:
            # Apply template-based corrections
            if template_sinogram_path:
                self._apply_template_corrections(h00_file, template_sinogram_path)

            if source:
                self._validate_scaling_factors(h00_file, source)

            # Convert to STIR format
            self.converter.convert_file(str(h00_file))
            self.logger.info(f"Processed {h00_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to process {h00_file}: {e}")
            raise OutputError(f"Failed to process {h00_file}: {e}")

    def _apply_template_corrections(
        self, h00_file: Path, template_sinogram_path: str
    ) -> None:
        """Apply corrections based on template sinogram header file.

        Args:
            h00_file: SIMIND output file to correct
            template_sinogram_path: Path to template sinogram header file (.hs)
        """
        try:
            # Extract attributes from template sinogram (backend-agnostic!)
            attributes = extract_attributes_from_stir(template_sinogram_path)

            # Template correction 1: Set acquisition time (projections × time per
            # projection)
            if "number_of_projections" in attributes and "image_duration" in attributes:
                time_per_projection = (
                    attributes["image_duration"] / attributes["number_of_projections"]
                )
                total_duration = (
                    attributes["number_of_projections"] * time_per_projection
                )
                self.converter.edit_parameter(
                    str(h00_file), "!image duration (sec)[1]", total_duration
                )
                self.logger.debug(f"Set image duration: {total_duration} s")

            # Template correction 2: Check and correct radius from template sinogram
            if "height_to_detector_surface" in attributes:
                expected_radius = attributes[
                    "height_to_detector_surface"
                ]  # Already in mm from STIR
                current_radius = self.converter.read_parameter(
                    str(h00_file), ";# Radius"
                )

                if (
                    current_radius is None
                    or abs(float(current_radius) - expected_radius) > 0.1
                ):
                    self.logger.info(
                        f"Correcting radius from template: {expected_radius:.4f} mm"
                    )
                    self.converter.edit_parameter(
                        str(h00_file), ";#Radius", expected_radius
                    )

            # Template correction 3: Handle non-circular orbits if present
            if attributes.get("orbit") == "non-circular" and "radii" in attributes:
                radii = attributes["radii"]  # Already in mm from STIR
                orbits_string = "{" + ",".join([f"{r:.1f}" for r in radii]) + "}"
                self.converter.add_parameter(
                    str(h00_file),
                    "Radii",
                    orbits_string,
                    59,  # line number to insert at
                )
                self.logger.debug("Added non-circular orbit radii from template")

        except Exception as e:
            self.logger.warning(
                f"Failed to apply template corrections to {h00_file}: {e}"
            )

    def _validate_scaling_factors(self, h00_file: Path, source) -> None:
        """Validate and fix scaling factors against source image.

        Args:
            h00_file: Path to SIMIND output file
            source: Source image object (backend-agnostic)
        """
        try:
            # Get voxel size from source image (in mm)
            voxel_size = source.voxel_sizes()[2]  # Get voxel size in z-direction
            self.logger.debug(f"Source voxel size: {voxel_size:.3f} mm")

            # Validate and fix scaling factors using the converter
            scaling_ok = self.converter.validate_and_fix_scaling_factors(
                str(h00_file), source, tolerance=0.00001
            )

            if not scaling_ok:
                self.logger.info(f"Corrected scaling factors in {h00_file.name}")
            else:
                self.logger.debug(f"Scaling factors validated for {h00_file.name}")

        except Exception as e:
            self.logger.warning(
                f"Failed to validate scaling factors for {h00_file}: {e}"
            )

    def _load_converted_scattwin_files(
        self, output_prefix: str, preferred_backend: Optional[str]
    ) -> Dict:
        """Load all converted scattwin .hs files.

        Args:
            output_prefix: Prefix used for output files

        Returns:
            Dictionary of output name -> AcquisitionDataInterface
        """
        output = {}
        hs_files = list(self.output_dir.glob(f"*{output_prefix}*.hs"))

        for hs_file in hs_files:
            try:
                # Extract scatter type and window from filename
                key = self._extract_output_key(hs_file.name)
                if BACKEND_AVAILABLE:
                    logging.info(f"Loading {hs_file} using backend factory")
                    if ensure_acquisition_interface is not None:
                        output[key] = ensure_acquisition_interface(
                            str(hs_file), preferred_backend=preferred_backend
                        )
                    else:
                        output[key] = create_acquisition_data(str(hs_file))
                else:
                    logging.info(f"Loading {hs_file} using SIRF")
                    output[key] = AcquisitionData(str(hs_file))
            except Exception as e:
                self.logger.error(f"Failed to load {hs_file}: {e}")
                continue

        if not output:
            raise OutputError(
                f"No valid scattwin output files found with prefix {output_prefix} "
                f"found in {self.output_dir}"
            )

        self.logger.info(f"Loaded {len(output)} scattwin output files")
        return output

    def _extract_output_key(self, filename: str) -> str:
        """Extract scatter type and window from filename."""
        # Parse filename to extract scatter type and window number
        parts = filename.split("_")
        if len(parts) >= 2:
            scatter_type = parts[-2]
            window = parts[-1].split(".")[0]
            return f"{scatter_type}_{window}"
        return filename

    def get_penetrate_component_description(
        self, component: PenetrateOutputType
    ) -> str:
        """Get detailed description for penetrate output component."""
        return component.description

    def list_expected_files(
        self, output_prefix: str, scoring_routine: ScoringRoutine
    ) -> List[str]:
        """List expected output files for a given scoring routine."""
        if scoring_routine == ScoringRoutine.SCATTWIN:
            # Scattwin files for window 1 (most common case)
            return [
                f"{output_prefix}_tot_w1.a00",
                f"{output_prefix}_sca_w1.a00",
                f"{output_prefix}_pri_w1.a00",
                f"{output_prefix}_air_w1.a00",
            ]
        elif scoring_routine == ScoringRoutine.PENETRATE:
            # All possible penetrate files
            return [f"{output_prefix}.b{i:02d}" for i in range(1, 20)]
        else:
            return []

    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during processing."""
        # This could be extended to clean up converter temporary files
        # For now, just log that cleanup was called
        self.logger.debug("Output processor cleanup completed")


__all__ = ["OutputProcessor"]
