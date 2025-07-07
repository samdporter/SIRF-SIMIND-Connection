# Docker Setup - Quick Start Guide

## 🚀 Quick Start (5 minutes)

```bash
# 1. Build and start development environment
make docker-dev

# 2. Check everything works
make docker-validate

# 3. Run tests
make docker-test

# 4. Access development container
make docker-shell
```

## 📋 Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM available for containers
- For GUI apps: X11 server (Linux) or XQuartz (macOS)

## 🎯 Common Workflows

### Development Workflow
```bash
# Start development environment (Jupyter + tools)
make docker-dev           # Background services
make docker-logs          # See logs

# Access development shell
make docker-shell         # Interactive shell

# Access Jupyter Lab
make docker-jupyter       # → http://localhost:8888
```

### Testing Workflow
```bash
# Quick validation (30 seconds)
make docker-validate

# All tests without SIMIND (2-5 minutes)
make docker-test

# Integration tests (requires SIMIND)
make docker-test-integration

# Complete test suite
make docker-test-full     # All tests + linting + benchmarks
```

### Code Quality Workflow
```bash
# Run linting and formatting checks
make docker-lint

# Fix formatting issues (in container)
make docker-shell
# Then inside container: black src/ tests/
```

## 🛠️ Available Services

| Service | Purpose | Command | Access |
|---------|---------|---------|--------|
| **Development** | Full dev environment | `make docker-dev` | Shell + Jupyter |
| **Testing** | Run test suites | `make docker-test-full` | Test reports |
| **Validation** | Quick health check | `make docker-validate` | Status output |
| **Linting** | Code quality | `make docker-lint` | Quality reports |
| **Jupyter** | Interactive notebooks | `make docker-jupyter` | http://localhost:8888 |
| **Documentation** | Build/serve docs | `make docker-docs` | http://localhost:8000 |
| **Benchmarks** | Performance tests | `make docker-benchmark` | Performance data |

## 🧪 Testing Strategies

### 1. **Quick Validation** (30 seconds)
```bash
make docker-validate
# ✓ Checks imports, syntax, basic functionality
```

### 2. **Unit Tests** (2-5 minutes)
```bash
make docker-test
# ✓ Tests that don't require SIMIND
# ✓ Code coverage reports
# ✓ Fast feedback loop
```

### 3. **Integration Tests** (5-15 minutes)
```bash
# Requires SIMIND installation first
make docker-install-simind    # Interactive setup
make docker-test-integration
# ✓ Tests with SIMIND
# ✓ Full workflow validation
```

### 4. **Full Test Suite** (10-20 minutes)
```bash
make docker-test-full
# ✓ All tests + linting + benchmarks
# ✓ Complete validation
# ✓ CI/CD ready
```

## 🔧 SIMIND Installation

SIMIND requires separate download and installation:

### Method 1: Interactive (Recommended)
```bash
# 1. Download SIMIND from https://simind.blogg.lu.se/downloads/
# 2. Start development container
make docker-dev

# 3. Install SIMIND interactively
make docker-install-simind
# Enter path when prompted: /path/to/simind_linux.tar.gz
```

### Method 2: Manual
```bash
# Copy archive to container
docker cp simind_linux.tar.gz sirf-simind-dev:/tmp/

# Install in container
docker-compose exec sirf-simind-dev install_simind.sh /tmp/simind_linux.tar.gz

# Validate installation
make docker-validate-simind
```

## 🐛 Debugging & Troubleshooting

### Check Service Status
```bash
make docker-status         # Show all services
make docker-logs           # Development logs
make docker-logs-follow    # Follow logs in real-time
```

### Common Issues

#### **"Container won't start"**
```bash
# Check logs
make docker-logs

# Clean rebuild
make docker-clean
make docker-build
```

#### **"Tests failing"**
```bash
# Run with verbose output
make docker-shell
# Then inside: pytest tests/test_specific.py -v -s
```

#### **"X11/GUI not working"**
```bash
# Linux: Allow X11 connections
xhost +local:docker

# macOS: Start XQuartz
# Check DISPLAY variable
echo $DISPLAY
```

#### **"Permission errors"**
```bash
# Access as root to fix permissions
make docker-shell
# Inside container: sudo chown -R sirfuser:sirfuser /home/sirfuser/workspace
```

#### **"Out of disk space"**
```bash
# Clean up Docker
make docker-clean-all       # Remove everything
docker system prune -a      # Deep clean
```

### Interactive Debugging
```bash
# Access running container
make docker-shell

# Run specific commands
make docker-shell
# Then: python -c "import sirf; print(sirf.__version__)"
```

## 🔍 Inspecting Results

### Test Results
```bash
# Test artifacts saved to:
./test-results/
├── junit.xml              # Test results
├── coverage.xml            # Coverage data
└── htmlcov/               # Coverage report
    └── index.html         # View in browser
```

### Performance Data
```bash
# Benchmark results in:
./benchmark-results/
└── benchmark.json         # Performance metrics
```

### Code Quality Reports
```bash
# Linting results in:
./lint-results/
├── black.txt              # Formatting issues
├── flake8.txt             # Style issues
├── mypy/                  # Type checking
└── bandit.txt             # Security issues
```

## ⚡ Performance Tips

### Faster Builds
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
make docker-build

# Skip tests during development
make docker-dev             # Start dev environment only
```

### Persistent Development
```bash
# Keep containers running
make docker-dev             # Starts in background
# Code changes sync automatically via volumes
```

### Resource Management
```bash
# Monitor resource usage
docker stats

# Check Docker status
make docker-status
```

## 📊 CI/CD Integration

### GitHub Actions
```bash
# Use CI profile for automated testing
make docker-ci              # Complete CI pipeline
```

### Local CI Simulation
```bash
# Run complete CI pipeline locally
make docker-ci              # Full Docker CI suite
make ci                     # Native Python CI suite
```

## 🆘 Getting Help

### Service Information
```bash
make help                   # Show all available commands
make docker-status          # Show Docker service status
docker-compose ps           # Show running services
```

### Container Information
```bash
# Enter container and explore
make docker-shell
ls -la                      # See project structure
pytest --help              # See testing options
install_simind.sh --help   # SIMIND installation help
```

### Log Analysis
```bash
# Real-time logs
make docker-logs-follow

# Service-specific logs
docker-compose logs test-unit
docker-compose logs lint
```

## 📝 Quick Reference

| Goal | Command | Time |
|------|---------|------|
| Start coding | `make docker-dev` | 2 min |
| Check if working | `make docker-validate` | 30 sec |
| Run basic tests | `make docker-test` | 3 min |
| Full validation | `make docker-test-full` | 15 min |
| Code formatting | `make docker-lint` | 1 min |
| Install SIMIND | `make docker-install-simind` | 5 min |
| Debug issues | `make docker-shell` | instant |
| Clean everything | `make docker-clean-all` | 2 min |

## 🔄 Native Python vs Docker

Your Makefile supports both workflows:

### Native Python Development
```bash
make install               # Install locally
make test                  # Run tests natively
make lint                  # Check code quality
make validate              # Validate installation
```

### Docker Development  
```bash
make docker-build          # Build Docker images
make docker-dev            # Start Docker environment
make docker-test           # Run tests in Docker
make docker-lint           # Check quality in Docker
```

**Choose based on your needs:**
- **Native**: Faster, direct access, easier debugging
- **Docker**: Consistent environment, includes SIRF, easier SIMIND setup

## 🎯 Recommended Workflows

### New Developer Setup
```bash
# Option 1: Docker (recommended for SIRF+SIMIND)
make docker-dev-setup      # Complete Docker setup

# Option 2: Native Python  
make dev-setup             # Complete native setup
```

### Daily Development
```bash
# Quick validation
make docker-validate       # or make validate

# Development loop
make docker-shell          # Start development
# Edit code...
make docker-test           # Test changes
```

### Before Committing
```bash
make pre-commit            # Run all pre-commit checks
# OR
make docker-ci             # Complete Docker CI pipeline
```

---

**💡 Pro Tip**: Start with `make docker-dev` and `make docker-validate` - this gets you a working SIRF environment in under 3 minutes!