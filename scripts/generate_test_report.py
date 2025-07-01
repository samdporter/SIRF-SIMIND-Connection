#!/usr/bin/env python3
"""
generate_test_report.py - Comprehensive test report generator for SIRF-SIMIND-Connection

This script generates comprehensive test reports by collecting and analyzing results from:
- Unit tests
- Integration tests
- Performance benchmarks
- Code coverage
- Example validation
- Installation validation

Usage:
    python generate_test_report.py
    python generate_test_report.py --output-dir test_reports --format html
    python generate_test_report.py --include-benchmarks --include-coverage

Author: SIRF-SIMIND-Connection Team
"""

import argparse
import subprocess
import sys
import os
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import tempfile
import shutil

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


@dataclass
class TestResult:
    """Container for individual test results."""
    name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    message: str = ""
    category: str = ""
    file_path: str = ""


@dataclass
class TestSuite:
    """Container for test suite results."""
    name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    tests: List[TestResult]


@dataclass
class CoverageData:
    """Container for code coverage data."""
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    missing_lines: List[int]
    file_coverage: Dict[str, float]


@dataclass
class BenchmarkData:
    """Container for benchmark results."""
    total_benchmarks: int
    successful: int
    failed: int
    total_time: float
    average_execution_time: float
    memory_usage_stats: Dict[str, float]


class TestReportGenerator:
    """Comprehensive test report generator."""
    
    def __init__(self, output_dir: str = "test_reports", verbose: bool = True):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
            verbose: Whether to print detailed output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_report_"))
        
        # Data containers
        self.test_suites: List[TestSuite] = []
        self.coverage_data: Optional[CoverageData] = None
        self.benchmark_data: Optional[BenchmarkData] = None
        self.validation_results: Dict[str, Any] = {}
        self.system_info: Dict[str, Any] = {}
        
        if self.verbose:
            print("SIRF-SIMIND-Connection Test Report Generator")
            print("=" * 50)
            print(f"Output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    run_tests: bool = True,
                                    include_coverage: bool = True,
                                    include_benchmarks: bool = False,
                                    include_examples: bool = True,
                                    include_validation: bool = True,
                                    formats: List[str] = ['html', 'json']) -> Dict[str, str]:
        """Generate comprehensive test report.
        
        Args:
            run_tests: Whether to run tests or use existing results
            include_coverage: Whether to include coverage analysis
            include_benchmarks: Whether to include performance benchmarks
            include_examples: Whether to validate example scripts
            include_validation: Whether to run installation validation
            formats: Output formats ('html', 'json', 'xml', 'txt')
            
        Returns:
            Dictionary mapping format to output file path
        """
        start_time = time.time()
        
        if self.verbose:
            print("\nGenerating comprehensive test report...")
            print("-" * 40)
        
        # Collect system information
        self._collect_system_info()
        
        # Run or collect test results
        if run_tests:
            self._run_test_suites()
        else:
            self._collect_existing_results()
        
        # Collect coverage data
        if include_coverage:
            self._collect_coverage_data()
        
        # Run benchmarks
        if include_benchmarks:
            self._run_benchmarks()
        
        # Validate examples
        if include_examples:
            self._validate_examples()
        
        # Run validation
        if include_validation:
            self._run_validation()
        
        # Generate reports in requested formats
        generated_files = {}
        for format_type in formats:
            output_file = self._generate_report_format(format_type)
            if output_file:
                generated_files[format_type] = str(output_file)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nReport generation completed in {total_time:.2f} seconds")
            print("Generated files:")
            for format_type, file_path in generated_files.items():
                print(f"  {format_type.upper()}: {file_path}")
        
        return generated_files
    
    def _collect_system_info(self):
        """Collect system and environment information."""
        import platform
        import sys
        
        self.system_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'path': sys.path[:5]  # First 5 entries
            },
            'environment': {
                'cwd': os.getcwd(),
                'user': os.getenv('USER', 'unknown'),
                'path': os.getenv('PATH', '')[:200] + '...'  # Truncated
            }
        }
        
        # Try to get git information
        try:
            git_branch = subprocess.run(['git', 'branch', '--show-current'], 
                                      capture_output=True, text=True, timeout=5)
            git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      capture_output=True, text=True, timeout=5)
            
            if git_branch.returncode == 0 and git_commit.returncode == 0:
                self.system_info['git'] = {
                    'branch': git_branch.stdout.strip(),
                    'commit': git_commit.stdout.strip()[:8]  # Short hash
                }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    def _run_test_suites(self):
        """Run comprehensive test suites."""
        test_commands = [
            ('Unit Tests', ['pytest', 'tests/', '-m', 'unit', '--tb=short', '-v', '--junit-xml=unit_results.xml']),
            ('Integration Tests', ['pytest', 'tests/', '-m', 'integration and not requires_simind', '--tb=short', '-v', '--junit-xml=integration_results.xml']),
            ('SIMIND Tests', ['pytest', 'tests/', '-m', 'requires_simind', '--tb=short', '-v', '--junit-xml=simind_results.xml']),
        ]
        
        for suite_name, command in test_commands:
            if self.verbose:
                print(f"Running {suite_name}...")
            
            try:
                # Change to temp directory for test outputs
                result = subprocess.run(
                    command,
                    cwd=self.temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                # Parse results
                xml_file = self.temp_dir / f"{command[-1].split('=')[1]}"
                if xml_file.exists():
                    test_suite = self._parse_junit_xml(xml_file, suite_name)
                    self.test_suites.append(test_suite)
                else:
                    # Fallback: create suite from return code
                    test_suite = TestSuite(
                        name=suite_name,
                        total_tests=1,
                        passed=1 if result.returncode == 0 else 0,
                        failed=0 if result.returncode == 0 else 1,
                        skipped=0,
                        errors=0,
                        duration=0.0,
                        tests=[]
                    )
                    self.test_suites.append(test_suite)
                
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print(f"  {suite_name} timed out")
                
                test_suite = TestSuite(
                    name=suite_name,
                    total_tests=1,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    duration=600.0,
                    tests=[TestResult(
                        name=f"{suite_name} (timeout)",
                        status="error",
                        duration=600.0,
                        message="Test suite timed out"
                    )]
                )
                self.test_suites.append(test_suite)
                
            except Exception as e:
                if self.verbose:
                    print(f"  {suite_name} failed: {e}")
                
                test_suite = TestSuite(
                    name=suite_name,
                    total_tests=1,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    duration=0.0,
                    tests=[TestResult(
                        name=f"{suite_name} (error)",
                        status="error",
                        duration=0.0,
                        message=str(e)
                    )]
                )
                self.test_suites.append(test_suite)
    
    def _collect_existing_results(self):
        """Collect existing test results from standard locations."""
        result_files = [
            ('Unit Tests', 'unit_results.xml'),
            ('Integration Tests', 'integration_results.xml'),
            ('SIMIND Tests', 'simind_results.xml'),
        ]
        
        for suite_name, filename in result_files:
            xml_file = Path(filename)
            if xml_file.exists():
                test_suite = self._parse_junit_xml(xml_file, suite_name)
                self.test_suites.append(test_suite)
    
    def _parse_junit_xml(self, xml_file: Path, suite_name: str) -> TestSuite:
        """Parse JUnit XML test results."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Handle different XML structures
            if root.tag == 'testsuites':
                testsuite = root.find('testsuite')
            else:
                testsuite = root
            
            if testsuite is None:
                raise ValueError("No testsuite element found")
            
            total_tests = int(testsuite.get('tests', 0))
            failures = int(testsuite.get('failures', 0))
            errors = int(testsuite.get('errors', 0))
            skipped = int(testsuite.get('skipped', 0))
            duration = float(testsuite.get('time', 0))
            passed = total_tests - failures - errors - skipped
            
            tests = []
            for testcase in testsuite.findall('testcase'):
                name = testcase.get('name', 'Unknown')
                test_duration = float(testcase.get('time', 0))
                
                # Determine status
                if testcase.find('failure') is not None:
                    status = 'failed'
                    message = testcase.find('failure').text or ''
                elif testcase.find('error') is not None:
                    status = 'error'
                    message = testcase.find('error').text or ''
                elif testcase.find('skipped') is not None:
                    status = 'skipped'
                    message = testcase.find('skipped').text or ''
                else:
                    status = 'passed'
                    message = ''
                
                tests.append(TestResult(
                    name=name,
                    status=status,
                    duration=test_duration,
                    message=message,
                    file_path=testcase.get('file', '')
                ))
            
            return TestSuite(
                name=suite_name,
                total_tests=total_tests,
                passed=passed,
                failed=failures,
                skipped=skipped,
                errors=errors,
                duration=duration,
                tests=tests
            )
            
        except Exception as e:
            if self.verbose:
                print(f"  Failed to parse {xml_file}: {e}")
            
            return TestSuite(
                name=suite_name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=0.0,
                tests=[TestResult(
                    name="XML Parsing Error",
                    status="error",
                    duration=0.0,
                    message=str(e)
                )]
            )
    
    def _collect_coverage_data(self):
        """Collect code coverage data."""
        if self.verbose:
            print("Collecting coverage data...")
        
        try:
            # Run coverage
            subprocess.run([
                'pytest', 'tests/', 
                '--cov=sirf_simind_connection',
                '--cov-report=xml:coverage.xml',
                '--cov-report=json:coverage.json',
                '-q'
            ], cwd=self.temp_dir, capture_output=True, timeout=300)
            
            # Parse coverage XML
            coverage_xml = self.temp_dir / 'coverage.xml'
            if coverage_xml.exists():
                self.coverage_data = self._parse_coverage_xml(coverage_xml)
            
        except Exception as e:
            if self.verbose:
                print(f"  Coverage collection failed: {e}")
    
    def _parse_coverage_xml(self, xml_file: Path) -> CoverageData:
        """Parse coverage XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get overall coverage
            coverage_elem = root.find('.//coverage')
            if coverage_elem is not None:
                lines_covered = int(coverage_elem.get('lines-covered', 0))
                lines_valid = int(coverage_elem.get('lines-valid', 0))
                coverage_percentage = (lines_covered / lines_valid * 100) if lines_valid > 0 else 0
            else:
                lines_covered = lines_valid = coverage_percentage = 0
            
            # Get file-level coverage
            file_coverage = {}
            for package in root.findall('.//package'):
                for cls in package.findall('classes/class'):
                    filename = cls.get('filename', '')
                    lines = cls.find('lines')
                    if lines is not None:
                        covered = sum(1 for line in lines.findall('line') 
                                    if line.get('hits', '0') != '0')
                        total = len(lines.findall('line'))
                        if total > 0:
                            file_coverage[filename] = covered / total * 100
            
            return CoverageData(
                total_lines=lines_valid,
                covered_lines=lines_covered,
                coverage_percentage=coverage_percentage,
                missing_lines=[],  # Would need more detailed parsing
                file_coverage=file_coverage
            )
            
        except Exception as e:
            if self.verbose:
                print(f"  Failed to parse coverage XML: {e}")
            
            return CoverageData(
                total_lines=0,
                covered_lines=0,
                coverage_percentage=0.0,
                missing_lines=[],
                file_coverage={}
            )
    
    def _run_benchmarks(self):
        """Run performance benchmarks."""
        if self.verbose:
            print("Running performance benchmarks...")
        
        try:
            # Run benchmark script
            result = subprocess.run([
                sys.executable, 'scripts/benchmark_performance.py',
                '--quick', '--output-dir', str(self.temp_dir), '--quiet'
            ], capture_output=True, text=True, timeout=300)
            
            # Load benchmark results
            benchmark_file = self.temp_dir / 'benchmark_results.json'
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                successful = sum(1 for r in results if r.get('success', False))
                failed = len(results) - successful
                
                if results:
                    exec_times = [r.get('execution_time', 0) for r in results if r.get('success')]
                    avg_time = sum(exec_times) / len(exec_times) if exec_times else 0
                    
                    memory_usages = [r.get('memory_usage_mb', 0) for r in results if r.get('success')]
                    memory_stats = {
                        'mean': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                        'max': max(memory_usages) if memory_usages else 0,
                        'min': min(memory_usages) if memory_usages else 0
                    }
                else:
                    avg_time = 0
                    memory_stats = {'mean': 0, 'max': 0, 'min': 0}
                
                self.benchmark_data = BenchmarkData(
                    total_benchmarks=len(results),
                    successful=successful,
                    failed=failed,
                    total_time=sum(exec_times),
                    average_execution_time=avg_time,
                    memory_usage_stats=memory_stats
                )
            
        except Exception as e:
            if self.verbose:
                print(f"  Benchmark execution failed: {e}")
    
    def _validate_examples(self):
        """Validate example scripts."""
        if self.verbose:
            print("Validating example scripts...")
        
        examples_dir = Path('examples')
        if not examples_dir.exists():
            return
        
        validated_examples = []
        for example_file in examples_dir.glob('*.py'):
            try:
                # Syntax check
                with open(example_file) as f:
                    compile(f.read(), str(example_file), 'exec')
                
                validated_examples.append({
                    'name': example_file.name,
                    'status': 'valid',
                    'message': 'Syntax check passed'
                })
                
            except SyntaxError as e:
                validated_examples.append({
                    'name': example_file.name,
                    'status': 'invalid',
                    'message': f'Syntax error: {e}'
                })
            except Exception as e:
                validated_examples.append({
                    'name': example_file.name,
                    'status': 'error',
                    'message': f'Error: {e}'
                })
        
        self.validation_results['examples'] = validated_examples
    
    def _run_validation(self):
        """Run installation validation."""
        if self.verbose:
            print("Running installation validation...")
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/validate_installation.py',
                '--quick', '--quiet', '--report-dir', str(self.temp_dir)
            ], capture_output=True, text=True, timeout=120)
            
            # Try to load validation results
            validation_file = self.temp_dir / 'validation_report.json'
            if validation_file.exists():
                with open(validation_file) as f:
                    self.validation_results['installation'] = json.load(f)
            else:
                self.validation_results['installation'] = {
                    'validation_summary': {
                        'success_rate': 100 if result.returncode == 0 else 0,
                        'total_tests': 1,
                        'passed_tests': 1 if result.returncode == 0 else 0
                    }
                }
                
        except Exception as e:
            if self.verbose:
                print(f"  Validation failed: {e}")
            
            self.validation_results['installation'] = {
                'validation_summary': {
                    'success_rate': 0,
                    'total_tests': 1,
                    'passed_tests': 0,
                    'error': str(e)
                }
            }
    
    def _generate_report_format(self, format_type: str) -> Optional[Path]:
        """Generate report in specified format."""
        if format_type == 'html':
            return self._generate_html_report()
        elif format_type == 'json':
            return self._generate_json_report()
        elif format_type == 'xml':
            return self._generate_xml_report()
        elif format_type == 'txt':
            return self._generate_text_report()
        else:
            if self.verbose:
                print(f"Unknown format: {format_type}")
            return None
    
    def _generate_html_report(self) -> Path:
        """Generate HTML report."""
        html_content = self._create_html_content()
        
        html_file = self.output_dir / "test_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Generate accompanying CSS
        css_file = self.output_dir / "report_styles.css"
        with open(css_file, 'w') as f:
            f.write(self._get_css_styles())
        
        return html_file
    
    def _create_html_content(self) -> str:
        """Create HTML content for the report."""
        # Calculate overall statistics
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        total_passed = sum(suite.passed for suite in self.test_suites)
        total_failed = sum(suite.failed for suite in self.test_suites)
        total_skipped = sum(suite.skipped for suite in self.test_suites)
        total_errors = sum(suite.errors for suite in self.test_suites)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIRF-SIMIND-Connection Test Report</title>
    <link rel="stylesheet" href="report_styles.css">
</head>
<body>
    <header>
        <h1>SIRF-SIMIND-Connection Test Report</h1>
        <p class="timestamp">Generated on {self.system_info['timestamp']}</p>
    </header>
    
    <nav>
        <ul>
            <li><a href="#summary">Summary</a></li>
            <li><a href="#test-suites">Test Suites</a></li>
            <li><a href="#coverage">Coverage</a></li>
            <li><a href="#benchmarks">Benchmarks</a></li>
            <li><a href="#validation">Validation</a></li>
            <li><a href="#system">System Info</a></li>
        </ul>
    </nav>
    
    <main>
        <section id="summary" class="card">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="metric">
                    <h3>Overall Success Rate</h3>
                    <div class="metric-value {'success' if overall_success_rate >= 80 else 'warning' if overall_success_rate >= 60 else 'failure'}">
                        {overall_success_rate:.1f}%
                    </div>
                </div>
                <div class="metric">
                    <h3>Total Tests</h3>
                    <div class="metric-value">{total_tests}</div>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <div class="metric-value success">{total_passed}</div>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <div class="metric-value {'failure' if total_failed > 0 else ''}">{total_failed}</div>
                </div>
            </div>
        </section>
        
        <section id="test-suites" class="card">
            <h2>Test Suite Results</h2>
"""
        
        # Add test suite details
        for suite in self.test_suites:
            suite_success_rate = (suite.passed / suite.total_tests * 100) if suite.total_tests > 0 else 0
            html += f"""
            <div class="test-suite">
                <h3>{suite.name}</h3>
                <div class="suite-stats">
                    <span class="stat">Total: {suite.total_tests}</span>
                    <span class="stat success">Passed: {suite.passed}</span>
                    <span class="stat {'failure' if suite.failed > 0 else ''}">Failed: {suite.failed}</span>
                    <span class="stat">Skipped: {suite.skipped}</span>
                    <span class="stat">Duration: {suite.duration:.2f}s</span>
                    <span class="stat">Success Rate: {suite_success_rate:.1f}%</span>
                </div>
            </div>
"""
        
        html += """
        </section>
"""
        
        # Add coverage section
        if self.coverage_data:
            html += f"""
        <section id="coverage" class="card">
            <h2>Code Coverage</h2>
            <div class="coverage-summary">
                <div class="metric">
                    <h3>Overall Coverage</h3>
                    <div class="metric-value {'success' if self.coverage_data.coverage_percentage >= 80 else 'warning' if self.coverage_data.coverage_percentage >= 60 else 'failure'}">
                        {self.coverage_data.coverage_percentage:.1f}%
                    </div>
                </div>
                <div class="metric">
                    <h3>Lines Covered</h3>
                    <div class="metric-value">{self.coverage_data.covered_lines}</div>
                </div>
                <div class="metric">
                    <h3>Total Lines</h3>
                    <div class="metric-value">{self.coverage_data.total_lines}</div>
                </div>
            </div>
        </section>
"""
        
        # Add benchmark section
        if self.benchmark_data:
            html += f"""
        <section id="benchmarks" class="card">
            <h2>Performance Benchmarks</h2>
            <div class="benchmark-summary">
                <div class="metric">
                    <h3>Total Benchmarks</h3>
                    <div class="metric-value">{self.benchmark_data.total_benchmarks}</div>
                </div>
                <div class="metric">
                    <h3>Successful</h3>
                    <div class="metric-value success">{self.benchmark_data.successful}</div>
                </div>
                <div class="metric">
                    <h3>Average Time</h3>
                    <div class="metric-value">{self.benchmark_data.average_execution_time:.3f}s</div>
                </div>
                <div class="metric">
                    <h3>Memory Usage</h3>
                    <div class="metric-value">{self.benchmark_data.memory_usage_stats.get('mean', 0):.1f}MB</div>
                </div>
            </div>
        </section>
"""
        
        # Add system info
        html += f"""
        <section id="system" class="card">
            <h2>System Information</h2>
            <div class="system-info">
                <p><strong>Platform:</strong> {self.system_info['platform']['system']} {self.system_info['platform']['release']}</p>
                <p><strong>Python:</strong> {self.system_info['python']['version'].split()[0]}</p>
                <p><strong>Machine:</strong> {self.system_info['platform']['machine']}</p>
"""
        
        if 'git' in self.system_info:
            html += f"""
                <p><strong>Git Branch:</strong> {self.system_info['git']['branch']}</p>
                <p><strong>Git Commit:</strong> {self.system_info['git']['commit']}</p>
"""
        
        html += """
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generated by SIRF-SIMIND-Connection Test Report Generator</p>
    </footer>
</body>
</html>
"""
        
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
/* SIRF-SIMIND-Connection Test Report Styles */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.timestamp {
    opacity: 0.8;
    font-size: 0.9rem;
}

nav {
    background: #fff;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    position: sticky;
    top: 0;
    z-index: 100;
}

nav ul {
    list-style: none;
    display: flex;
    justify-content: center;
    gap: 2rem;
}

nav a {
    text-decoration: none;
    color: #667eea;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.2s;
}

nav a:hover {
    background-color: #f0f0f0;
}

main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 1rem;
}

.card {
    background: white;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.card h2 {
    color: #667eea;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
}

.summary-grid, .coverage-summary, .benchmark-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

.metric {
    text-align: center;
    padding: 1rem;
    border-radius: 6px;
    background: #f8f9fa;
}

.metric h3 {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #333;
}

.metric-value.success {
    color: #28a745;
}

.metric-value.warning {
    color: #ffc107;
}

.metric-value.failure {
    color: #dc3545;
}

.test-suite {
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.test-suite h3 {
    color: #495057;
    margin-bottom: 1rem;
}

.suite-stats {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.stat {
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    background: #e9ecef;
    font-size: 0.85rem;
    font-weight: 500;
}

.stat.success {
    background: #d4edda;
    color: #155724;
}

.stat.failure {
    background: #f8d7da;
    color: #721c24;
}

.system-info p {
    margin-bottom: 0.5rem;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 4px;
}

footer {
    text-align: center;
    padding: 2rem;
    color: #666;
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    header h1 {
        font-size: 2rem;
    }
    
    nav ul {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .summary-grid, .coverage-summary, .benchmark-summary {
        grid-template-columns: 1fr;
    }
    
    .suite-stats {
        flex-direction: column;
        gap: 0.5rem;
    }
}
"""
    
    def _generate_json_report(self) -> Path:
        """Generate JSON report."""
        report_data = {
            'metadata': {
                'generator': 'SIRF-SIMIND-Connection Test Report Generator',
                'timestamp': self.system_info['timestamp'],
                'system_info': self.system_info
            },
            'test_suites': [asdict(suite) for suite in self.test_suites],
            'coverage': asdict(self.coverage_data) if self.coverage_data else None,
            'benchmarks': asdict(self.benchmark_data) if self.benchmark_data else None,
            'validation': self.validation_results
        }
        
        json_file = self.output_dir / "test_report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_file
    
    def _generate_xml_report(self) -> Path:
        """Generate XML report."""
        # Simple XML structure
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<test_report timestamp="{self.system_info['timestamp']}">
    <summary>
        <total_tests>{sum(suite.total_tests for suite in self.test_suites)}</total_tests>
        <passed>{sum(suite.passed for suite in self.test_suites)}</passed>
        <failed>{sum(suite.failed for suite in self.test_suites)}</failed>
        <skipped>{sum(suite.skipped for suite in self.test_suites)}</skipped>
    </summary>
    <test_suites>
"""
        
        for suite in self.test_suites:
            xml_content += f"""
        <test_suite name="{suite.name}" tests="{suite.total_tests}" 
                   passed="{suite.passed}" failed="{suite.failed}" 
                   skipped="{suite.skipped}" duration="{suite.duration}">
"""
            for test in suite.tests:
                xml_content += f"""
            <test name="{test.name}" status="{test.status}" duration="{test.duration}">
                <message>{test.message}</message>
            </test>
"""
            xml_content += "        </test_suite>\n"
        
        xml_content += """
    </test_suites>
</test_report>
"""
        
        xml_file = self.output_dir / "test_report.xml"
        with open(xml_file, 'w') as f:
            f.write(xml_content)
        
        return xml_file
    
    def _generate_text_report(self) -> Path:
        """Generate plain text report."""
        report = []
        report.append("SIRF-SIMIND-Connection Test Report")
        report.append("=" * 50)
        report.append(f"Generated: {self.system_info['timestamp']}")
        report.append("")
        
        # Summary
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        total_passed = sum(suite.passed for suite in self.test_suites)
        total_failed = sum(suite.failed for suite in self.test_suites)
        
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {total_passed}")
        report.append(f"Failed: {total_failed}")
        report.append(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        report.append("")
        
        # Test suites
        report.append("TEST SUITES")
        report.append("-" * 20)
        for suite in self.test_suites:
            report.append(f"{suite.name}:")
            report.append(f"  Tests: {suite.total_tests}")
            report.append(f"  Passed: {suite.passed}")
            report.append(f"  Failed: {suite.failed}")
            report.append(f"  Duration: {suite.duration:.2f}s")
            report.append("")
        
        # Coverage
        if self.coverage_data:
            report.append("COVERAGE")
            report.append("-" * 20)
            report.append(f"Overall: {self.coverage_data.coverage_percentage:.1f}%")
            report.append(f"Lines: {self.coverage_data.covered_lines}/{self.coverage_data.total_lines}")
            report.append("")
        
        # System info
        report.append("SYSTEM INFORMATION")
        report.append("-" * 20)
        report.append(f"Platform: {self.system_info['platform']['system']}")
        report.append(f"Python: {self.system_info['python']['version'].split()[0]}")
        
        text_file = self.output_dir / "test_report.txt"
        with open(text_file, 'w') as f:
            f.write("\n".join(report))
        
        return text_file
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main function for the test report generator."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test reports for SIRF-SIMIND-Connection"
    )
    parser.add_argument(
        "--output-dir", default="test_reports",
        help="Directory to save generated reports"
    )
    parser.add_argument(
        "--format", action="append", default=[],
        choices=['html', 'json', 'xml', 'txt'],
        help="Output formats (can specify multiple)"
    )
    parser.add_argument(
        "--no-tests", action="store_true",
        help="Skip running tests, use existing results"
    )
    parser.add_argument(
        "--include-coverage", action="store_true", default=True,
        help="Include code coverage analysis"
    )
    parser.add_argument(
        "--include-benchmarks", action="store_true",
        help="Include performance benchmarks"
    )
    parser.add_argument(
        "--no-examples", action="store_true",
        help="Skip example validation"
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Skip installation validation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Set default formats if none specified
    if not args.format:
        args.format = ['html', 'json']
    
    # Create report generator
    generator = TestReportGenerator(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    try:
        # Generate comprehensive report
        generated_files = generator.generate_comprehensive_report(
            run_tests=not args.no_tests,
            include_coverage=args.include_coverage,
            include_benchmarks=args.include_benchmarks,
            include_examples=not args.no_examples,
            include_validation=not args.no_validation,
            formats=args.format
        )
        
        print(f"\n✅ Report generation completed!")
        print(f"📁 Output directory: {generator.output_dir}")
        print("\n📋 Generated reports:")
        for format_type, file_path in generated_files.items():
            print(f"  {format_type.upper()}: {file_path}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nReport generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nReport generation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()