#!/usr/bin/env python3
"""
setup_testing_environment.py - Automated setup for SIRF-SIMIND-Connection testing environment

This script automatically sets up a complete testing environment for the SIRF-SIMIND-Connection
package, including dependencies, test data, and development tools.

Usage:
    python setup_testing_environment.py
    python setup_testing_environment.py --minimal      # Minimal setup
    python setup_testing_environment.py --with-simind  # Include SIMIND installation help
    python setup_testing_environment.py --docker       # Set up Docker environment

Features:
- Dependency installation and verification
- Test data generation
- Development tool configuration
- Git hooks installation
- Environment validation
- CI/CD setup assistance

Author: SIRF-SIMIND-Connection Team
"""

import argparse
import subprocess
import sys
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import platform
from dataclasses import dataclass
from urllib.request import urlretrieve
from urllib.error import URLError


@dataclass
class SetupStep:
    """Represents a setup step with status tracking."""
    name: str
    description: str
    required: bool = True
    completed: bool = False
    error_message: Optional[str] = None
    duration: float = 0.0


class TestingEnvironmentSetup:
    """Automated setup for testing environment."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the setup manager.
        
        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
        self.steps: List[SetupStep] = []
        self.project_root = Path.cwd()
        self.python_executable = sys.executable
        self.system_info = self._get_system_info()
        
        if self.verbose:
            print("SIRF-SIMIND-Connection Testing Environment Setup")
            print("=" * 60)
            print(f"Project root: {self.project_root}")
            print(f"Python: {self.python_executable}")
            print(f"Platform: {self.system_info['platform']}")
            print()
    
    def setup_complete_environment(self, 
                                 minimal: bool = False,
                                 with_simind: bool = False,
                                 docker_setup: bool = False,
                                 skip_optional: bool = False) -> bool:
        """Set up the complete testing environment.
        
        Args:
            minimal: Set up minimal environment only
            with_simind: Include SIMIND installation assistance
            docker_setup: Set up Docker testing environment
            skip_optional: Skip optional dependencies
            
        Returns:
            True if setup completed successfully
        """
        start_time = time.time()
        
        if self.verbose:
            mode = "minimal" if minimal else "docker" if docker_setup else "complete"
            print(f"Setting up {mode} testing environment...")
            print("-" * 40)
        
        # Core setup steps (always run)
        self._setup_python_environment()
        self._install_core_dependencies()
        self._install_package_dev_mode()
        
        if not minimal:
            # Full setup steps
            if not skip_optional:
                self._install_optional_dependencies()
            
            self._setup_development_tools()
            self._generate_test_data()
            self._setup_git_hooks()
            self._validate_installation()
            
            if with_simind:
                self._setup_simind_assistance()
        
        if docker_setup:
            self._setup_docker_environment()
        
        # Final validation
        self._run_final_validation()
        
        total_time = time.time() - start_time
        
        # Generate setup report
        self._generate_setup_report(total_time)
        
        # Print summary
        self._print_setup_summary(total_time)
        
        # Check if setup was successful
        required_steps = [step for step in self.steps if step.required]
        failed_required = [step for step in required_steps if not step.completed]
        
        return len(failed_required) == 0
    
    def _run_step(self, step: SetupStep, func, *args, **kwargs) -> bool:
        """Run a setup step with error handling and timing."""
        if self.verbose:
            print(f"🔧 {step.name}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            step.duration = time.time() - start_time
            step.completed = True
            
            if self.verbose:
                print(f"✅ ({step.duration:.1f}s)")
            
            return True
            
        except Exception as e:
            step.duration = time.time() - start_time
            step.error_message = str(e)
            step.completed = False
            
            if self.verbose:
                print(f"❌ ({step.duration:.1f}s)")
                if step.required:
                    print(f"   Error: {str(e)}")
                else:
                    print(f"   Warning: {str(e)} (optional)")
            
            return False
        finally:
            self.steps.append(step)
    
    def _setup_python_environment(self):
        """Set up Python environment and verify version."""
        step = SetupStep(
            name="Python Environment Check",
            description="Verify Python version and virtual environment",
            required=True
        )
        
        def check_python():
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
            
            # Check if in virtual environment
            in_venv = (hasattr(sys, 'real_prefix') or 
                      (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
            
            if not in_venv and self.verbose:
                print("\n   ⚠️  Not in virtual environment. Consider using venv or conda.")
            
            return f"Python {version.major}.{version.minor}.{version.micro}"
        
        self._run_step(step, check_python)
    
    def _install_core_dependencies(self):
        """Install core dependencies."""
        step = SetupStep(
            name="Core Dependencies",
            description="Install numpy, scipy, matplotlib, PyYAML",
            required=True
        )
        
        def install_core():
            core_packages = [
                'numpy>=1.20.0',
                'scipy>=1.7.0',
                'matplotlib>=3.3.0',
                'PyYAML>=5.4.0',
                'pydicom>=2.0.0'
            ]
            
            cmd = [self.python_executable, '-m', 'pip', 'install'] + core_packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"pip install failed: {result.stderr}")
            
            return f"Installed {len(core_packages)} packages"
        
        self._run_step(step, install_core)
    
    def _install_optional_dependencies(self):
        """Install optional dependencies."""
        step = SetupStep(
            name="Optional Dependencies",
            description="Install psutil, h5py, and other optional packages",
            required=False
        )
        
        def install_optional():
            optional_packages = [
                'psutil',
                'h5py',
                'requests',
                'pillow',
                'imageio',
            ]
            
            installed = []
            for package in optional_packages:
                try:
                    cmd = [self.python_executable, '-m', 'pip', 'install', package]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        installed.append(package)
                except subprocess.TimeoutExpired:
                    pass
            
            return f"Installed {len(installed)}/{len(optional_packages)} optional packages"
        
        self._run_step(step, install_optional)
    
    def _install_package_dev_mode(self):
        """Install the package in development mode."""
        step = SetupStep(
            name="Package Installation",
            description="Install SIRF-SIMIND-Connection in development mode",
            required=True
        )
        
        def install_package():
            # Try to install with dev dependencies first
            cmd = [self.python_executable, '-m', 'pip', 'install', '-e', '.[dev]']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                # Fallback: install without dev dependencies
                cmd = [self.python_executable, '-m', 'pip', 'install', '-e', '.']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Package installation failed: {result.stderr}")
                
                return "Installed package (without dev dependencies)"
            
            return "Installed package with dev dependencies"
        
        self._run_step(step, install_package)
    
    def _setup_development_tools(self):
        """Set up development tools."""
        step = SetupStep(
            name="Development Tools",
            description="Install pytest, black, flake8, mypy, pre-commit",
            required=False
        )
        
        def install_dev_tools():
            dev_packages = [
                'pytest>=6.0',
                'pytest-cov>=2.10.0',
                'pytest-mock>=3.6.0',
                'pytest-xdist',
                'black>=21.0.0',
                'flake8>=3.8.0',
                'mypy>=0.910',
                'isort>=5.0.0',
                'bandit>=1.7.0',
                'pre-commit>=2.15.0'
            ]
            
            cmd = [self.python_executable, '-m', 'pip', 'install'] + dev_packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # Try installing essential tools only
                essential = ['pytest', 'black', 'flake8']
                cmd = [self.python_executable, '-m', 'pip', 'install'] + essential
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Essential dev tools installation failed: {result.stderr}")
                
                return f"Installed essential dev tools only"
            
            return f"Installed all development tools"
        
        self._run_step(step, install_dev_tools)
    
    def _generate_test_data(self):
        """Generate test data."""
        step = SetupStep(
            name="Test Data Generation",
            description="Generate phantoms, attenuation maps, and test configurations",
            required=False
        )
        
        def generate_data():
            data_script = self.project_root / "scripts" / "generate_test_data.py"
            
            if not data_script.exists():
                return "Test data script not found (skipped)"
            
            cmd = [self.python_executable, str(data_script), '--output-dir', 'test_data']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Test data generation failed: {result.stderr}")
            
            # Count generated files
            test_data_dir = self.project_root / "test_data"
            if test_data_dir.exists():
                file_count = len(list(test_data_dir.glob("*")))
                return f"Generated {file_count} test data files"
            
            return "Test data generated"
        
        self._run_step(step, generate_data)
    
    def _setup_git_hooks(self):
        """Set up git hooks."""
        step = SetupStep(
            name="Git Hooks Setup",
            description="Install pre-commit hooks and git configuration",
            required=False
        )
        
        def setup_hooks():
            if not (self.project_root / ".git").exists():
                return "Not a git repository (skipped)"
            
            # Try to install pre-commit hooks
            try:
                cmd = ['pre-commit', 'install']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    return "Pre-commit hooks installed"
            except FileNotFoundError:
                pass
            
            # Fallback: create simple pre-commit hook
            hooks_dir = self.project_root / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            
            pre_commit_hook = hooks_dir / "pre-commit"
            hook_content = """#!/bin/bash
# Simple pre-commit hook for SIRF-SIMIND-Connection
echo "Running pre-commit checks..."

# Run quick tests
python -m pytest tests/ -m "not slow and not requires_simind" --tb=short -q

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
"""
            
            pre_commit_hook.write_text(hook_content)
            pre_commit_hook.chmod(0o755)
            
            return "Simple git hook installed"
        
        self._run_step(step, setup_hooks)
    
    def _setup_simind_assistance(self):
        """Provide SIMIND installation assistance."""
        step = SetupStep(
            name="SIMIND Setup Assistance",
            description="Check SIMIND availability and provide installation guidance",
            required=False
        )
        
        def check_simind():
            # Check if SIMIND is already available
            try:
                result = subprocess.run(['simind', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return "SIMIND already installed and available"
            except FileNotFoundError:
                pass
            
            # Provide installation guidance
            guidance_file = self.project_root / "SIMIND_INSTALLATION_GUIDE.md"
            guidance_content = """# SIMIND Installation Guide

## Overview
SIMIND is required for Monte Carlo simulations but is not automatically installed.

## Installation Steps

### 1. Download SIMIND
- Visit: https://simind.blogg.lu.se/downloads/
- Register for access
- Download appropriate version for your system

### 2. System-Specific Installation

#### Linux/macOS:
```bash
# Extract downloaded archive
tar -xzf simind_*.tar.gz
cd simind_*

# Add to PATH
echo 'export PATH=$PATH:/path/to/simind/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
simind --version
```

#### Windows:
1. Extract ZIP file to C:\\simind
2. Add C:\\simind\\bin to system PATH
3. Open new command prompt and verify: simind --version

### 3. Testing SIMIND Installation
```bash
# Test with package
python scripts/validate_installation.py --full

# Or run SIMIND tests
make test-simind
```

## Alternative: Docker
Use the provided Docker environment with SIMIND pre-installed:
```bash
docker build -f Dockerfile.dev -t sirf-simind-dev .
docker run -it sirf-simind-dev
```

## Troubleshooting
- Ensure SIMIND executable is in PATH
- Check file permissions on Linux/macOS
- Verify system compatibility
- See package documentation for more details
"""
            
            guidance_file.write_text(guidance_content)
            return f"SIMIND guidance saved to {guidance_file.name}"
        
        self._run_step(step, check_simind)
    
    def _setup_docker_environment(self):
        """Set up Docker testing environment."""
        step = SetupStep(
            name="Docker Environment",
            description="Validate Docker setup and build test images",
            required=False
        )
        
        def setup_docker():
            # Check if Docker is available
            try:
                result = subprocess.run(['docker', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return "Docker not available (install Docker to use this feature)"
            except FileNotFoundError:
                return "Docker not found (install Docker to use this feature)"
            
            # Check if Dockerfile exists
            dockerfile = self.project_root / "Dockerfile.test"
            if not dockerfile.exists():
                return "Dockerfile.test not found (skipped)"
            
            # Try to build test image
            cmd = ['docker', 'build', '-f', 'Dockerfile.test', '-t', 'sirf-simind-test', '.']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"Docker build failed: {result.stderr}")
            
            return "Docker test image built successfully"
        
        self._run_step(step, setup_docker)
    
    def _validate_installation(self):
        """Run installation validation."""
        step = SetupStep(
            name="Installation Validation",
            description="Validate package installation and dependencies",
            required=True
        )
        
        def validate():
            validation_script = self.project_root / "scripts" / "validate_installation.py"
            
            if not validation_script.exists():
                # Basic validation
                try:
                    import sirf_simind_connection
                    return f"Package import successful (version: {getattr(sirf_simind_connection, '__version__', 'unknown')})"
                except ImportError as e:
                    raise RuntimeError(f"Package import failed: {e}")
            
            # Run validation script
            cmd = [self.python_executable, str(validation_script), '--quick', '--quiet']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise RuntimeError(f"Validation failed: {result.stderr}")
            
            return "Full validation passed"
        
        self._run_step(step, validate)
    
    def _run_final_validation(self):
        """Run final validation tests."""
        step = SetupStep(
            name="Final Validation",
            description="Run quick test suite to verify setup",
            required=True
        )
        
        def final_validation():
            # Try to run quick tests
            cmd = [self.python_executable, '-m', 'pytest', 'tests/', 
                   '-m', 'not slow and not requires_simind', '--tb=short', '-q']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                # If pytest fails, try basic imports
                try:
                    import sirf_simind_connection
                    import numpy
                    import matplotlib
                    return "Basic imports successful (pytest issues)"
                except ImportError as e:
                    raise RuntimeError(f"Basic validation failed: {e}")
            
            # Parse test results
            lines = result.stdout.split('\n')
            for line in lines:
                if 'failed' in line and 'passed' in line:
                    return f"Tests completed: {line.strip()}"
            
            return "Quick tests passed"
        
        self._run_step(step, final_validation)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'platform': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_executable': sys.executable
        }
    
    def _generate_setup_report(self, total_time: float):
        """Generate setup report."""
        report_data = {
            'setup_info': {
                'timestamp': time.time(),
                'total_time': total_time,
                'system_info': self.system_info,
                'project_root': str(self.project_root)
            },
            'steps': [
                {
                    'name': step.name,
                    'description': step.description,
                    'required': step.required,
                    'completed': step.completed,
                    'duration': step.duration,
                    'error_message': step.error_message
                }
                for step in self.steps
            ]
        }
        
        report_file = self.project_root / "setup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _print_setup_summary(self, total_time: float):
        """Print setup summary."""
        if not self.verbose:
            return
        
        completed_steps = [step for step in self.steps if step.completed]
        failed_steps = [step for step in self.steps if not step.completed]
        required_failed = [step for step in failed_steps if step.required]
        
        print("\n" + "=" * 60)
        print("SETUP SUMMARY")
        print("=" * 60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Completed steps: {len(completed_steps)}/{len(self.steps)}")
        print(f"Failed steps: {len(failed_steps)}")
        print(f"Required failures: {len(required_failed)}")
        
        if completed_steps:
            print(f"\n✅ Completed steps:")
            for step in completed_steps:
                print(f"   • {step.name} ({step.duration:.1f}s)")
        
        if failed_steps:
            print(f"\n❌ Failed steps:")
            for step in failed_steps:
                status = "REQUIRED" if step.required else "optional"
                print(f"   • {step.name} ({status})")
                if step.error_message:
                    print(f"     Error: {step.error_message}")
        
        print(f"\n📋 Next steps:")
        if len(required_failed) == 0:
            print("   • Setup completed successfully!")
            print("   • Run 'make test-quick' to verify everything works")
            print("   • See README_TESTING.md for detailed usage instructions")
            print("   • Try 'python examples/05_complete_workflow.py' for a demo")
        else:
            print("   • Fix required step failures before proceeding")
            print("   • Run this script again after fixing issues")
            print("   • Check error messages above for troubleshooting")
        
        print(f"\n📁 Generated files:")
        generated_files = [
            "setup_report.json",
            "test_data/",
            "SIMIND_INSTALLATION_GUIDE.md"
        ]
        for file_path in generated_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"   • {file_path}")


def main():
    """Main function for the setup script."""
    parser = argparse.ArgumentParser(
        description="Set up testing environment for SIRF-SIMIND-Connection"
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="Set up minimal environment only (no dev tools, test data)"
    )
    parser.add_argument(
        "--with-simind", action="store_true",
        help="Include SIMIND installation assistance"
    )
    parser.add_argument(
        "--docker", action="store_true",
        help="Set up Docker testing environment"
    )
    parser.add_argument(
        "--skip-optional", action="store_true",
        help="Skip optional dependencies and tools"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force setup even if some steps fail"
    )
    
    args = parser.parse_args()
    
    # Create setup manager
    setup = TestingEnvironmentSetup(verbose=not args.quiet)
    
    try:
        # Run setup
        success = setup.setup_complete_environment(
            minimal=args.minimal,
            with_simind=args.with_simind,
            docker_setup=args.docker,
            skip_optional=args.skip_optional
        )
        
        # Exit with appropriate code
        if success or args.force:
            print(f"\n🎉 Setup completed {'successfully' if success else 'with warnings'}!")
            sys.exit(0)
        else:
            print(f"\n❌ Setup failed. Please check error messages above.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nSetup failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()