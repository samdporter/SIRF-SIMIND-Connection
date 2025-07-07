#!/bin/bash
# docker/install_simind.sh - Improved SIMIND installation script
#
# Usage: ./install_simind.sh [path_to_simind_archive] [--validate-only] [--force]
#
# This script helps install SIMIND in the Docker container.
# SIMIND must be downloaded separately from https://simind.blogg.lu.se/downloads/

set -euo pipefail  # Stricter error handling

# Configuration
SIMIND_ARCHIVE="${1:-}"
INSTALL_DIR="/opt/simind"
BACKUP_DIR="/opt/simind.backup.$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/simind_install.log"
VALIDATE_ONLY=false
FORCE_INSTALL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option $1" >&2
            exit 1
            ;;
        *)
            if [ -z "$SIMIND_ARCHIVE" ]; then
                SIMIND_ARCHIVE="$1"
            fi
            shift
            ;;
    esac
done

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1" >&2
    cleanup_on_error
    exit 1
}

# Cleanup function
cleanup_on_error() {
    if [ -d "$BACKUP_DIR" ] && [ -d "$INSTALL_DIR" ]; then
        log "Restoring from backup due to error..."
        run_privileged "rm -rf '$INSTALL_DIR' && mv '$BACKUP_DIR' '$INSTALL_DIR'" || true
    fi
}

# Help function
show_help() {
    cat << EOF
SIMIND Installation Script for Docker Container

Usage: $0 [path_to_simind_archive] [options]

Options:
    --validate-only    Only check if SIMIND is properly installed
    --force           Force reinstallation even if SIMIND exists
    --help, -h        Show this help message

Examples:
    $0 /tmp/simind_linux.tar.gz
    $0 --validate-only
    $0 /tmp/simind.zip --force

To install SIMIND:
1. Register at https://simind.blogg.lu.se/downloads/
2. Download the Linux version of SIMIND
3. Copy the archive into the container:
   docker cp simind_linux.tar.gz <container_name>:/tmp/
4. Run this script:
   docker exec <container_name> /tmp/install_simind.sh /tmp/simind_linux.tar.gz

EOF
}

# Check if we can run privileged commands
run_privileged() {
    local cmd="$1"
    if [ "$EUID" -eq 0 ]; then
        # Already root
        eval "$cmd"
    elif command -v sudo >/dev/null 2>&1; then
        # Use sudo
        sudo bash -c "$cmd"
    else
        error_exit "Need root privileges but sudo not available"
    fi
}

# System requirements check
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check architecture
    local arch=$(uname -m)
    case $arch in
        x86_64|amd64)
            log "✅ Architecture: $arch (supported)"
            ;;
        i386|i686)
            log "⚠️  Architecture: $arch (may have compatibility issues)"
            ;;
        *)
            error_exit "Unsupported architecture: $arch"
            ;;
    esac
    
    # Check required libraries/tools
    local missing_deps=()
    for dep in tar unzip gzip; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error_exit "Missing required dependencies: ${missing_deps[*]}"
    fi
    
    log "✅ System requirements check passed"
}

# Validate existing installation
validate_installation() {
    log "Validating SIMIND installation..."
    
    local validation_errors=()
    
    # Check installation directory
    if [ ! -d "$INSTALL_DIR" ]; then
        validation_errors+=("Installation directory missing: $INSTALL_DIR")
    fi
    
    # Check for executables
    if [ -d "$INSTALL_DIR/bin" ]; then
        local exe_count=$(find "$INSTALL_DIR/bin" -type f -executable 2>/dev/null | wc -l)
        if [ "$exe_count" -eq 0 ]; then
            validation_errors+=("No executable files found in $INSTALL_DIR/bin")
        else
            log "✅ Found $exe_count executable(s) in bin directory"
        fi
    else
        validation_errors+=("Bin directory missing: $INSTALL_DIR/bin")
    fi
    
    # Check PATH accessibility
    if command -v simind >/dev/null 2>&1; then
        log "✅ SIMIND executable accessible via PATH"
        local simind_path=$(which simind)
        log "   Located at: $simind_path"
        
        # Try version command (may fail, that's OK)
        if simind --version >/dev/null 2>&1; then
            local version=$(simind --version 2>&1 || echo "Unknown")
            log "✅ SIMIND version: $version"
        else
            log "ℹ️  SIMIND version command failed (may be normal)"
        fi
    else
        validation_errors+=("SIMIND not accessible via PATH")
    fi
    
    # Check library directory if it exists
    if [ -d "$INSTALL_DIR/lib" ]; then
        local lib_count=$(find "$INSTALL_DIR/lib" -name "*.so*" 2>/dev/null | wc -l)
        log "ℹ️  Found $lib_count shared library file(s)"
    fi
    
    # Report results
    if [ ${#validation_errors[@]} -eq 0 ]; then
        log "✅ SIMIND installation validation successful!"
        return 0
    else
        log "❌ SIMIND installation validation failed:"
        for error in "${validation_errors[@]}"; do
            log "   • $error"
        done
        return 1
    fi
}

# Detect archive type and extract
extract_archive() {
    local archive="$1"
    local temp_dir="$2"
    
    log "Detecting archive type for: $(basename "$archive")"
    
    # Detect by file command first, then by extension
    local file_type=""
    if command -v file >/dev/null 2>&1; then
        file_type=$(file -b "$archive")
    fi
    
    case "$file_type" in
        *"gzip compressed"*|*"tar archive"*)
            log "Detected: gzipped tar archive"
            tar -xzf "$archive" -C "$temp_dir"
            ;;
        *"Zip archive"*)
            log "Detected: ZIP archive"
            unzip -q "$archive" -d "$temp_dir"
            ;;
        *)
            # Fall back to extension-based detection
            case "$archive" in
                *.tar.gz|*.tgz)
                    log "Detected by extension: gzipped tar archive"
                    tar -xzf "$archive" -C "$temp_dir"
                    ;;
                *.tar.bz2|*.tbz2)
                    log "Detected by extension: bzip2 tar archive"
                    tar -xjf "$archive" -C "$temp_dir"
                    ;;
                *.tar.xz|*.txz)
                    log "Detected by extension: xz tar archive"
                    tar -xJf "$archive" -C "$temp_dir"
                    ;;
                *.tar)
                    log "Detected by extension: tar archive"
                    tar -xf "$archive" -C "$temp_dir"
                    ;;
                *.zip)
                    log "Detected by extension: ZIP archive"
                    unzip -q "$archive" -d "$temp_dir"
                    ;;
                *)
                    error_exit "Unsupported archive format: $archive"
                    ;;
            esac
            ;;
    esac
}

# Normalize extracted directory structure
normalize_structure() {
    local temp_dir="$1"
    
    # Count top-level items
    local items=($(find "$temp_dir" -maxdepth 1 -mindepth 1))
    
    if [ ${#items[@]} -eq 1 ] && [ -d "${items[0]}" ]; then
        # Single directory - move contents up one level
        local subdir="${items[0]}"
        log "Moving contents from subdirectory: $(basename "$subdir")"
        
        # Move all contents to temp location
        local move_temp="${temp_dir}_move"
        mv "$subdir" "$move_temp"
        mv "$move_temp"/* "$temp_dir"/
        rmdir "$move_temp"
    fi
    
    # Verify we have expected structure
    if [ ! -d "$temp_dir/bin" ] && [ -f "$temp_dir/simind" ]; then
        # Single executable - create bin directory
        log "Creating bin directory for single executable"
        mkdir -p "$temp_dir/bin"
        mv "$temp_dir/simind" "$temp_dir/bin/"
    fi
}

# Main installation function
install_simind() {
    local archive="$1"
    
    log "Starting SIMIND installation from: $archive"
    
    # Verify archive exists and is readable
    if [ ! -f "$archive" ]; then
        error_exit "Archive not found: $archive"
    fi
    
    if [ ! -r "$archive" ]; then
        error_exit "Archive not readable: $archive"
    fi
    
    # Create temporary extraction directory
    local temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" EXIT
    
    # Backup existing installation
    if [ -d "$INSTALL_DIR" ] && [ ! "$FORCE_INSTALL" = true ]; then
        if validate_installation >/dev/null 2>&1; then
            log "SIMIND already installed and working. Use --force to reinstall."
            return 0
        fi
        log "Backing up existing installation..."
        run_privileged "cp -r '$INSTALL_DIR' '$BACKUP_DIR'"
    fi
    
    # Extract archive
    log "Extracting archive..."
    extract_archive "$archive" "$temp_dir"
    
    # Normalize directory structure
    normalize_structure "$temp_dir"
    
    # Install to target directory
    log "Installing to $INSTALL_DIR..."
    run_privileged "mkdir -p '$INSTALL_DIR'"
    run_privileged "cp -r '$temp_dir'/* '$INSTALL_DIR'/"
    
    # Set permissions
    log "Setting permissions..."
    run_privileged "chmod -R 755 '$INSTALL_DIR'"
    if [ -d "$INSTALL_DIR/bin" ]; then
        run_privileged "find '$INSTALL_DIR/bin' -type f -exec chmod +x {} +"
    fi
    
    # Clean up backup if installation successful
    if validate_installation; then
        if [ -d "$BACKUP_DIR" ]; then
            log "Removing backup (installation successful)"
            run_privileged "rm -rf '$BACKUP_DIR'"
        fi
        log "✅ SIMIND installation completed successfully!"
    else
        error_exit "Installation validation failed"
    fi
}

# Main execution
main() {
    echo "SIMIND Installation Script"
    echo "=========================="
    
    # Initialize log
    : > "$LOG_FILE"
    log "Starting SIMIND installation script"
    log "Log file: $LOG_FILE"
    
    # Check system requirements
    check_system_requirements
    
    # Handle validation-only mode
    if [ "$VALIDATE_ONLY" = true ]; then
        if validate_installation; then
            echo "✅ SIMIND validation successful"
            exit 0
        else
            echo "❌ SIMIND validation failed"
            exit 1
        fi
    fi
    
    # Check if archive provided
    if [ -z "$SIMIND_ARCHIVE" ]; then
        echo "Error: No SIMIND archive provided"
        echo ""
        show_help
        exit 1
    fi
    
    # Install SIMIND
    install_simind "$SIMIND_ARCHIVE"
    
    echo ""
    echo "Installation complete! You can now run SIMIND-dependent tests:"
    echo "  pytest tests/ -m requires_simind -v"
    echo "  python examples/05_complete_workflow.py"
    echo ""
    echo "To validate installation later:"
    echo "  $0 --validate-only"
}

# Execute main function
main "$@"