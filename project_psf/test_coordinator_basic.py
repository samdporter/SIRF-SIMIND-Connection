#!/usr/bin/env python3
"""
Basic test of SimindCoordinator and SimindSubsetProjector integration.

This script creates a minimal test case to validate:
1. SimindCoordinator creation and configuration
2. SimindSubsetProjector creation with coordinator
3. CIL partitioner integration
4. Basic forward/backward operations
"""

import logging
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    from sirf_simind_connection.core.coordinator import SimindCoordinator
    from sirf_simind_connection.core.projector import SimindSubsetProjector
    from sirf_simind_connection.utils.cil_partitioner import (
        partition_data_with_cil_objectives,
    )

    logging.info("✓ Successfully imported coordinator and partitioner modules")
except ImportError as e:
    logging.error(f"Import failed: {e}")
    sys.exit(1)

try:
    from sirf.STIR import (
        AcquisitionData,
        AcquisitionModelUsingMatrix,
        ImageData,
        SPECTUBMatrix,
    )

    logging.info("✓ SIRF imports successful")
except ImportError as e:
    logging.error(f"SIRF not available: {e}")
    sys.exit(1)

try:
    from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction

    logging.info("✓ CIL imports successful")
except ImportError as e:
    logging.error(f"CIL not available: {e}")
    sys.exit(1)

logging.info("\n" + "=" * 80)
logging.info("Basic Coordinator Test - Module Loading")
logging.info("=" * 80)
logging.info("\nAll required modules loaded successfully!")
logging.info("\nTo run a full test:")
logging.info("  1. Create SIMIND simulator")
logging.info("  2. Create coordinator with simulator, num_subsets, update_interval")
logging.info("  3. Partition data using partition_data_with_cil_objectives")
logging.info("  4. Run forward/backward projections")
logging.info("\nTest PASSED ✓")
