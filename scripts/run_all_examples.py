#!/usr/bin/env python
"""
Run All Examples Script

This script runs all SIRF-SIMIND-Connection examples in sequence,
providing a comprehensive demonstration of the package capabilities.

Run from the scripts/ directory:
    cd scripts/
    python run_all_examples.py
"""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


# Add the parent directory to Python path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Change to parent directory so relative paths work correctly
os.chdir(parent_dir)

# List of examples to run (now in ../examples from scripts/)
EXAMPLES = [
    {
        "name": "01_basic_simulation.py",
        "description": "Basic SIMIND simulation with simple phantom",
        "module": "examples.01_basic_simulation",
        "estimated_time": "1 minute",
    },
    {
        "name": "02_runtime_switch_comparison.py",
        "description": "Runtime-switch comparison using pure Python connector",
        "module": "examples.02_runtime_switch_comparison",
        "estimated_time": "1 minute",
    },
    {
        "name": "03_multi_window.py",
        "description": "Multi-energy window simulation (TEW method)",
        "module": "examples.03_multi_window",
        "estimated_time": "1 minute",
    },
    {
        "name": "04_custom_config.py",
        "description": "Custom configuration creation and YAML workflow",
        "module": "examples.04_custom_config",
        "estimated_time": "< 1 minute",
    },
    {
        "name": "05_scattwin_vs_penetrate_comparison.py",
        "description": "Comparison of SCATTWIN vs PENETRATE scoring routines",
        "module": "examples.05_scattwin_vs_penetrate_comparison",
        "estimated_time": "3 minutes",
    },
    {
        "name": "06_schneider_density_conversion.py",
        "description": "Schneider2000 density conversion demonstration",
        "module": "examples.06_schneider_density_conversion",
        "estimated_time": "< 1 minute",
    },
]


def print_header():
    """Print script header."""
    print("=" * 80)
    print("SIRF-SIMIND-Connection - Run All Examples")
    print("=" * 80)
    print(f"Running from: {os.getcwd()}")
    print(f"Total examples: {len(EXAMPLES)}")
    print("Estimated total runtime: 10-25 minutes (depending on system)")
    print("\nThis script will run all examples in sequence.")
    print("Each example creates output in its own directory under 'output/'")
    print("=" * 80)


def check_dependencies():
    """Check if required packages are available."""
    print("\nChecking dependencies...")

    required_modules = ["sirf_simind_connection", "matplotlib", "numpy"]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing.append(module)

    if missing:
        print(f"\nERROR: Missing required modules: {missing}")
        print("Please install missing dependencies and try again.")
        return False

    print("✓ All dependencies found")
    return True


def run_example_subprocess(example_file):
    """Run an example file as a subprocess."""
    try:
        print(f"Running {example_file} as subprocess...")
        result = subprocess.run(
            [
                sys.executable,
                f"examples/{example_file}",  # examples are in parent directory
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )  # 30 min timeout

        if result.returncode == 0:
            return True, "Completed successfully"
        else:
            return False, f"Exit code {result.returncode}: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Timed out after 30 minutes"
    except Exception as e:
        return False, f"Subprocess error: {e}"


def run_single_example(example_info, example_num):
    """Run a single example and return results."""
    name = example_info["name"]
    description = example_info["description"]
    estimated_time = example_info["estimated_time"]

    print(f"\n{'=' * 60}")
    print(f"EXAMPLE {example_num}: {name}")
    print(f"Description: {description}")
    print(f"Estimated time: {estimated_time}")
    print("=" * 60)

    start_time = time.time()

    success, message = run_example_subprocess(name)

    elapsed_time = time.time() - start_time

    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"\n{status}: {message}")
    print(f"Runtime: {elapsed_time:.1f} seconds")

    return {
        "name": name,
        "success": success,
        "message": message,
        "runtime": elapsed_time,
    }


def print_summary(results):
    """Print final summary of all examples."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_runtime = sum(r["runtime"] for r in results)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"Total examples run: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(
        f"Total runtime: {total_runtime:.1f} seconds ({total_runtime / 60:.1f} minutes)"
    )

    print("\nDetailed Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "✓" if result["success"] else "✗"
        print(f"{i:2d}. {status} {result['name']:<35} ({result['runtime']:5.1f}s)")
        if not result["success"]:
            print(f"    Error: {result['message']}")

    print("\nOutput Directories:")
    output_dir = Path("output")
    if output_dir.exists():
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            file_count = len(list(subdir.glob("*")))
            print(f"  - {subdir.name}/ ({file_count} files)")
    else:
        print("  No output directory found")

    if failed > 0:
        print(f"\n⚠️  {failed} example(s) failed. Check error messages above.")
        print("Common issues:")
        print("  - SIMIND not installed or not in PATH")
        print("  - SIRF not properly installed")
        print("  - Missing dependencies")
        print("  - Insufficient disk space")
    else:
        print("\n🎉 All examples completed successfully!")
        print("Check the output/ directory for results and visualizations.")


def main():
    """Main execution function."""
    print_header()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Ask user for confirmation
    response = input("\nProceed with running all examples? [y/N]: ").lower().strip()
    if response not in ["y", "yes"]:
        print("Cancelled by user.")
        sys.exit(0)

    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")

    # Create main output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Run all examples
    results = []
    start_time = time.time()

    for i, example in enumerate(EXAMPLES, 1):
        try:
            result = run_single_example(example, i)
            results.append(result)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user during example {i}")
            break
        except Exception as e:
            print(f"\n\n✗ Unexpected error in example {i}: {e}")
            traceback.print_exc()
            results.append(
                {
                    "name": example["name"],
                    "success": False,
                    "message": f"Unexpected error: {e}",
                    "runtime": 0,
                }
            )

    # Print final summary
    print_summary(results)

    total_time = time.time() - start_time
    print(
        f"\nTotal execution time: {total_time:.1f} seconds "
        f"({total_time / 60:.1f} minutes)"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
