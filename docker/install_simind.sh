# docker/install_simind.sh
#!/bin/bash
# SIMIND installation script for Docker container
#
# Usage: ./install_simind.sh [path_to_simind_archive]
#
# This script helps install SIMIND in the Docker container.
# SIMIND must be downloaded separately from https://simind.blogg.lu.se/downloads/

set -e

SIMIND_ARCHIVE="$1"
INSTALL_DIR="/opt/simind"

echo "SIMIND Installation Script"
echo "=========================="

if [ -z "$SIMIND_ARCHIVE" ]; then
    echo "Error: No SIMIND archive provided"
    echo ""
    echo "Usage: $0 <path_to_simind_archive>"
    echo ""
    echo "To install SIMIND:"
    echo "1. Register at https://simind.blogg.lu.se/downloads/"
    echo "2. Download the Linux version of SIMIND"
    echo "3. Copy the archive into the container:"
    echo "   docker cp simind_linux.tar.gz <container_name>:/tmp/"
    echo "4. Run this script:"
    echo "   docker exec <container_name> /tmp/install_simind.sh /tmp/simind_linux.tar.gz"
    echo ""
    exit 1
fi

if [ ! -f "$SIMIND_ARCHIVE" ]; then
    echo "Error: SIMIND archive not found: $SIMIND_ARCHIVE"
    exit 1
fi

echo "Installing SIMIND from: $SIMIND_ARCHIVE"
echo "Installation directory: $INSTALL_DIR"

# Create installation directory
sudo mkdir -p "$INSTALL_DIR"

# Extract archive
echo "Extracting SIMIND archive..."
if [[ "$SIMIND_ARCHIVE" == *.tar.gz ]] || [[ "$SIMIND_ARCHIVE" == *.tgz ]]; then
    sudo tar -xzf "$SIMIND_ARCHIVE" -C "$INSTALL_DIR" --strip-components=1
elif [[ "$SIMIND_ARCHIVE" == *.zip ]]; then
    sudo unzip -q "$SIMIND_ARCHIVE" -d "$INSTALL_DIR"
    # Move contents up one level if extracted to subdirectory
    if [ $(sudo find "$INSTALL_DIR" -maxdepth 1 -type d | wc -l) -eq 2 ]; then
        SUBDIR=$(sudo find "$INSTALL_DIR" -maxdepth 1 -type d | tail -1)
        sudo mv "$SUBDIR"/* "$INSTALL_DIR/"
        sudo rmdir "$SUBDIR"
    fi
else
    echo "Error: Unsupported archive format. Expected .tar.gz, .tgz, or .zip"
    exit 1
fi

# Set permissions
echo "Setting permissions..."
sudo chmod -R 755 "$INSTALL_DIR"
if [ -d "$INSTALL_DIR/bin" ]; then
    sudo chmod +x "$INSTALL_DIR/bin"/*
fi

# Verify installation
echo "Verifying installation..."
if [ -d "$INSTALL_DIR/bin" ] && [ -n "$(ls -A $INSTALL_DIR/bin 2>/dev/null)" ]; then
    echo "✅ SIMIND installed successfully!"
    echo "   Installation directory: $INSTALL_DIR"
    echo "   Executables found in: $INSTALL_DIR/bin"
    echo "   PATH already configured in container"
    
    # Try to run SIMIND
    if command -v simind >/dev/null 2>&1; then
        echo "✅ SIMIND executable is accessible"
        simind --version 2>/dev/null || echo "ℹ️  SIMIND version command returned error (may be normal)"
    else
        echo "⚠️  SIMIND not found in PATH. You may need to restart the container."
    fi
else
    echo "❌ SIMIND installation may have failed"
    echo "   Expected executables in $INSTALL_DIR/bin but directory is empty or missing"
    exit 1
fi

echo ""
echo "Installation complete! You can now run SIMIND-dependent tests:"
echo "  pytest tests/ -m requires_simind -v"
echo "  python examples/05_complete_workflow.py"