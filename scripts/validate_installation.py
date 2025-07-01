#!/usr/bin/env python3
"""
validate_installation.py - Comprehensive validation script for SIRF-SIMIND-Connection

This script performs comprehensive validation of the SIRF-SIMIND-Connection installation,
including dependency checking, functionality testing, and performance benchmarking.

Usage:
    python validate_installation.py
    python validate_installation.py --quick
    python validate_installation.py --full --report-dir validation_results

Author: SIRF-SIMIND-Connection Team
"""

import argparse
import sys
import os
import time
import subprocess
import importlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import yaml

# Try to import packages for validation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ValidationResult:
    """Container for validation test results."""
    
    def __init__(self, name: str, passed: bool, details: str = "", 
                 duration: float = 0.0, error: Optional[Exception] = None):
        self.name = name
        self.passed = passed
        self.details = details
        self.duration = duration
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'details': self.details,
            'duration': self.duration,
            'error': str(self.error) if self.error else None,
            'timestamp': self.timestamp
        }


class InstallationValidator:
    """Comprehensive validator for SIRF-SIMIND-Connection installation."""
    
    def __init__(self, report_dir: Optional[str] = None, verbose: bool = True):
        """Initialize the validator.
        
        Args:
            report_dir: Directory to save validation reports
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        
        if report_dir:
            self.report_dir = Path(report_dir)
            self.report_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.report_dir = Path(tempfile.mkdtemp(prefix="validation_"))
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="validation_temp_"))
        
        if self.verbose:
            print("SIRF-SIMIND-Connection Installation Validator")
            print("=" * 60)
            print(f"Report directory: {self.report_dir}")
            print(f"Temporary directory: {self.temp_dir}")
    
    def run_validation(self, quick: bool = False, full: bool = False) -> bool:
        """Run the complete validation suite.
        
        Args:
            quick: Run only essential validation tests
            full: Run comprehensive validation including performance tests
            
        Returns:
            True if all validations pass, False otherwise
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\nStarting validation (mode: {'quick' if quick else 'full' if full else 'standard'})")
            print("-" * 60)
        
        # Core validation tests (always run)
        self._validate_python_version()
        self._validate_core_dependencies()
        self._validate_package_import()
        
        if not quick:
            self._validate_optional_dependencies()
            self._validate_sirf_availability()
            self._validate_simind_availability()
            self._validate_basic_functionality()
            
        if full:
            self._validate_examples()
            self._validate_test_suite()
            self._validate_performance()
            self._validate_documentation()
        
        total_time = time.time() - start_time
        
        # Generate report
        self._generate_report(total_time)
        
        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Total tests: {total_count}")
            print(f"Passed: {passed_count}")
            print(f"Failed: {total_count - passed_count}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Report saved to: {self.report_dir}")
            
            if passed_count < total_count:
                print("\nFailed tests:")
                for result in self.results:
                    if not result.passed:
                        print(f"  ❌ {result.name}: {result.details}")
            else:
                print("\n🎉 All validations passed!")
        
        return passed_count == total_count
    
    def _run_test(self, name: str, test_func, *args, **kwargs) -> ValidationResult:
        """Run a single validation test with error handling."""
        if self.verbose:
            print(f"Running: {name}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, tuple):
                passed, details = result
            elif isinstance(result, bool):
                passed, details = result, "OK" if result else "Failed"
            else:
                passed, details = True, str(result) if result else "OK"
            
            validation_result = ValidationResult(name, passed, details, duration)
            
            if self.verbose:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} ({duration:.2f}s)")
                if not passed and details:
                    print(f"    Details: {details}")
            
        except Exception as e:
            duration = time.time() - start_time
            validation_result = ValidationResult(name, False, str(e), duration, e)
            
            if self.verbose:
                print(f"❌ ERROR ({duration:.2f}s)")
                print(f"    Error: {str(e)}")
        
        self.results.append(validation_result)
        return validation_result
    
    def _validate_python_version(self):
        """Validate Python version requirements."""
        def check_python_version():
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                return False, f"Python {version.major}.{version.minor} < 3.8 (required)"
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        
        self._run_test("Python Version Check", check_python_version)
    
    def _validate_core_dependencies(self):
        """Validate core dependency availability."""
        core_deps = ['numpy', 'scipy', 'matplotlib', 'yaml']
        
        for dep in core_deps:
            self._run_test(f"Core Dependency: {dep}", self._check_import, dep)
    
    def _validate_optional_dependencies(self):
        """Validate optional dependency availability."""
        optional_deps = ['pydicom', 'pytest', 'h5py']
        
        for dep in optional_deps:
            self._run_test(f"Optional Dependency: {dep}", self._check_import, dep, optional=True)
    
    def _validate_package_import(self):
        """Validate main package import."""
        def check_package_import():
            try:
                import sirf_simind_connection
                version = getattr(sirf_simind_connection, '__version__', 'unknown')
                return True, f"Version {version}"
            except ImportError as e:
                return False, f"Import failed: {str(e)}"
        
        self._run_test("Package Import", check_package_import)
    
    def _validate_sirf_availability(self):
        """Validate SIRF availability."""
        def check_sirf():
            try:
                from sirf.STIR import ImageData, AcquisitionData
                # Try creating a simple object
                img = ImageData()
                return True, "SIRF available and functional"
            except ImportError:
                return False, "SIRF not available (optional for some functionality)"
            except Exception as e:
                return False, f"SIRF import error: {str(e)}"
        
        self._run_test("SIRF Availability", check_sirf)
    
    def _validate_simind_availability(self):
        """Validate SIMIND availability."""
        def check_simind():
            try:
                result = subprocess.run(['simind', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return True, "SIMIND executable found"
                else:
                    return False, "SIMIND executable not responding correctly"
            except FileNotFoundError:
                return False, "SIMIND executable not found in PATH (required for simulations)"
            except subprocess.TimeoutExpired:
                return False, "SIMIND executable timeout"
            except Exception as e:
                return False, f"SIMIND check error: {str(e)}"
        
        self._run_test("SIMIND Availability", check_simind)
    
    def _validate_basic_functionality(self):
        """Validate basic package functionality."""
        def test_config_creation():
            try:
                from sirf_simind_connection import SimulationConfig
                
                # Create a minimal SMC file
                smc_content = """
                TITLE Test Configuration
                PHOTONS 1000
                SPECTRUM MONO 140.0
                """
                smc_file = self.temp_dir / "test.smc"
                smc_file.write_text(smc_content)
                
                config = SimulationConfig(str(smc_file))
                return True, "SimulationConfig creation successful"
            except Exception as e:
                return False, f"Configuration creation failed: {str(e)}"
        
        def test_utility_functions():
            try:
                from sirf_simind_connection.utils import density_conversion
                
                # Test basic conversion
                hu_values = np.array([0, 1000, -1000])
                densities = density_conversion.hounsfield_to_density(hu_values)
                
                if len(densities) == 3 and np.all(densities >= 0):
                    return True, "Utility functions working"
                else:
                    return False, "Utility function output invalid"
            except Exception as e:
                return False, f"Utility function error: {str(e)}"
        
        self._run_test("Configuration Creation", test_config_creation)
        self._run_test("Utility Functions", test_utility_functions)
    
    def _validate_examples(self):
        """Validate example scripts."""
        def test_example_syntax(example_file):
            """Test that example file has valid syntax."""
            try:
                with open(example_file, 'r') as f:
                    content = f.read()
                compile(content, example_file, 'exec')
                return True, "Syntax valid"
            except SyntaxError as e:
                return False, f"Syntax error: {str(e)}"
            except Exception as e:
                return False, f"Error: {str(e)}"
        
        # Find example files
        example_dir = Path(__file__).parent.parent / "examples"
        if example_dir.exists():
            for example_file in example_dir.glob("*.py"):
                test_name = f"Example Syntax: {example_file.name}"
                self._run_test(test_name, test_example_syntax, example_file)
        else:
            self._run_test("Examples Directory", lambda: (False, "Examples directory not found"))
    
    def _validate_test_suite(self):
        """Validate test suite functionality."""
        def run_unit_tests():
            try:
                # Try to run a subset of unit tests
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', 
                    'tests/', '-m', 'unit and not requires_simind',
                    '--tb=short', '-q'
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    return True, "Unit tests passed"
                else:
                    return False, f"Unit tests failed: {result.stdout}"
            except subprocess.TimeoutExpired:
                return False, "Unit tests timed out"
            except FileNotFoundError:
                return False, "pytest not available"
            except Exception as e:
                return False, f"Test execution error: {str(e)}"
        
        self._run_test("Unit Test Suite", run_unit_tests)
    
    def _validate_performance(self):
        """Run basic performance validation."""
        def test_import_time():
            """Test package import time."""
            start_time = time.time()
            try:
                import sirf_simind_connection
                import_time = time.time() - start_time
                
                if import_time < 5.0:  # Should import quickly
                    return True, f"Import time: {import_time:.2f}s"
                else:
                    return False, f"Slow import: {import_time:.2f}s"
            except Exception as e:
                return False, f"Import failed: {str(e)}"
        
        def test_memory_usage():
            """Test basic memory usage."""
            try:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Do some basic operations
                import sirf_simind_connection
                import numpy as np
                
                # Create some test data
                test_array = np.random.rand(100, 100, 50)
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                if memory_increase < 500:  # Less than 500 MB increase
                    return True, f"Memory usage: {memory_increase:.1f} MB"
                else:
                    return False, f"High memory usage: {memory_increase:.1f} MB"
            except ImportError:
                return True, "psutil not available (skipped)"
            except Exception as e:
                return False, f"Memory test error: {str(e)}"
        
        self._run_test("Import Performance", test_import_time)
        self._run_test("Memory Usage", test_memory_usage)
    
    def _validate_documentation(self):
        """Validate documentation availability."""
        def check_readme():
            readme_file = Path(__file__).parent.parent / "README.md"
            if readme_file.exists():
                content = readme_file.read_text()
                if len(content) > 100:  # Basic content check
                    return True, "README.md found and non-empty"
                else:
                    return False, "README.md too short"
            else:
                return False, "README.md not found"
        
        def check_docstrings():
            try:
                from sirf_simind_connection import SimindSimulator
                if SimindSimulator.__doc__:
                    return True, "Docstrings present"
                else:
                    return False, "Missing docstrings"
            except Exception as e:
                return False, f"Docstring check error: {str(e)}"
        
        self._run_test("README Documentation", check_readme)
        self._run_test("Code Documentation", check_docstrings)
    
    def _check_import(self, module_name: str, optional: bool = False) -> Tuple[bool, str]:
        """Check if a module can be imported."""
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            return True, f"Available (version: {version})"
        except ImportError:
            if optional:
                return True, "Not available (optional)"
            else:
                return False, "Not available (required)"
        except Exception as e:
            return False, f"Import error: {str(e)}"
    
    def _generate_report(self, total_time: float):
        """Generate comprehensive validation report."""
        # Summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Create detailed report
        report = {
            'validation_summary': {
                'timestamp': time.time(),
                'total_time': total_time,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'executable': sys.executable
            },
            'test_results': [result.to_dict() for result in self.results]
        }
        
        # Save JSON report
        json_report_file = self.report_dir / "validation_report.json"
        with open(json_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save YAML report (if available)
        if YAML_AVAILABLE:
            yaml_report_file = self.report_dir / "validation_report.yaml"
            with open(yaml_report_file, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
        
        # Save human-readable report
        text_report_file = self.report_dir / "validation_report.txt"
        with open(text_report_file, 'w') as f:
            f.write("SIRF-SIMIND-Connection Installation Validation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Validation Date: {time.ctime()}\n")
            f.write(f"Total Time: {total_time:.2f} seconds\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n\n")
            
            f.write("Summary:\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Success Rate: {(passed_tests / total_tests * 100):.1f}%\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 40 + "\n")
            
            for result in self.results:
                status = "PASS" if result.passed else "FAIL"
                f.write(f"{result.name:.<40} {status:>8} ({result.duration:.2f}s)\n")
                if result.details:
                    f.write(f"    {result.details}\n")
                if result.error:
                    f.write(f"    Error: {result.error}\n")
                f.write("\n")
        
        # Generate HTML report if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            self._generate_html_report(report)
    
    def _generate_html_report(self, report_data: Dict):
        """Generate HTML visualization report."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Wedge
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('SIRF-SIMIND-Connection Validation Report', fontsize=16, fontweight='bold')
            
            # 1. Success/Failure pie chart
            passed = report_data['validation_summary']['passed_tests']
            failed = report_data['validation_summary']['failed_tests']
            
            if passed + failed > 0:
                ax1.pie([passed, failed], labels=['Passed', 'Failed'], 
                       colors=['green', 'red'], autopct='%1.1f%%')
                ax1.set_title('Test Results Overview')
            
            # 2. Test duration bar chart
            test_names = [r['name'] for r in report_data['test_results']]
            durations = [r['duration'] for r in report_data['test_results']]
            colors = ['green' if r['passed'] else 'red' for r in report_data['test_results']]
            
            ax2.barh(range(len(test_names)), durations, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(test_names)))
            ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in test_names])
            ax2.set_xlabel('Duration (seconds)')
            ax2.set_title('Test Execution Times')
            
            # 3. Success rate by category
            categories = {}
            for result in report_data['test_results']:
                category = result['name'].split(':')[0] if ':' in result['name'] else 'General'
                if category not in categories:
                    categories[category] = {'passed': 0, 'total': 0}
                categories[category]['total'] += 1
                if result['passed']:
                    categories[category]['passed'] += 1
            
            if categories:
                cat_names = list(categories.keys())
                success_rates = [categories[cat]['passed'] / categories[cat]['total'] * 100 
                               for cat in cat_names]
                
                bars = ax3.bar(range(len(cat_names)), success_rates, color='lightblue', alpha=0.7)
                ax3.set_xticks(range(len(cat_names)))
                ax3.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in cat_names], 
                                   rotation=45)
                ax3.set_ylabel('Success Rate (%)')
                ax3.set_title('Success Rate by Category')
                ax3.set_ylim(0, 100)
                
                # Color bars based on success rate
                for bar, rate in zip(bars, success_rates):
                    if rate >= 90:
                        bar.set_color('green')
                    elif rate >= 70:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            # 4. System information text
            ax4.axis('off')
            system_info = f"""
System Information:
• Python: {report_data['system_info']['python_version']}
• Platform: {report_data['system_info']['platform']}
• Total Tests: {report_data['validation_summary']['total_tests']}
• Success Rate: {report_data['validation_summary']['success_rate']:.1f}%
• Total Time: {report_data['validation_summary']['total_time']:.2f}s

Validation completed successfully!
            """
            ax4.text(0.1, 0.9, system_info, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save HTML report
            html_file = self.report_dir / "validation_report.html"
            plt.savefig(html_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
            
            # Create simple HTML wrapper
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SIRF-SIMIND-Connection Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; color: #333; }}
        .summary {{ background-color: #f0f0f0; padding: 20px; margin: 20px 0; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SIRF-SIMIND-Connection Validation Report</h1>
        <p>Generated on {time.ctime()}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {report_data['validation_summary']['total_tests']}</p>
        <p><strong>Passed:</strong> <span class="success">{report_data['validation_summary']['passed_tests']}</span></p>
        <p><strong>Failed:</strong> <span class="failure">{report_data['validation_summary']['failed_tests']}</span></p>
        <p><strong>Success Rate:</strong> {report_data['validation_summary']['success_rate']:.1f}%</p>
        <p><strong>Total Time:</strong> {report_data['validation_summary']['total_time']:.2f} seconds</p>
    </div>
    
    <img src="{html_file.with_suffix('.png').name}" alt="Validation Charts" style="width: 100%; max-width: 1200px;">
    
    <h2>Detailed Results</h2>
    <table border="1" style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #f0f0f0;">
            <th style="padding: 10px;">Test Name</th>
            <th style="padding: 10px;">Status</th>
            <th style="padding: 10px;">Duration</th>
            <th style="padding: 10px;">Details</th>
        </tr>
"""
            
            for result in report_data['test_results']:
                status_class = "success" if result['passed'] else "failure"
                status_text = "PASS" if result['passed'] else "FAIL"
                html_content += f"""
        <tr>
            <td style="padding: 10px;">{result['name']}</td>
            <td style="padding: 10px;" class="{status_class}">{status_text}</td>
            <td style="padding: 10px;">{result['duration']:.2f}s</td>
            <td style="padding: 10px;">{result['details']}</td>
        </tr>
"""
            
            html_content += """
    </table>
</body>
</html>
"""
            
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            plt.close()
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate HTML report: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main function for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate SIRF-SIMIND-Connection installation and functionality"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only essential validation tests"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run comprehensive validation including performance tests"
    )
    parser.add_argument(
        "--report-dir", default=None,
        help="Directory to save validation reports (default: temporary directory)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = InstallationValidator(
        report_dir=args.report_dir,
        verbose=not args.quiet
    )
    
    try:
        # Run validation
        success = validator.run_validation(quick=args.quick, full=args.full)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nValidation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()