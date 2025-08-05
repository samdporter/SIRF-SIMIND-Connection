.. _installation:

Installation
============

To install the SIRF-SIMIND-Connection package, ensure you have Python 3.8+ and follow the steps below:

1. Clone the Repository:

   ```bash
   git clone https://github.com/samdporter/SIRF-SIMIND-Connection.git
   cd SIRF-SIMIND-Connection
   ```

2. Install Dependencies:

   ```bash
   pip install -e ".[dev]"  # Include development tools
   ```

3. Verify Installation:

   Use the scripts in the `scripts` directory:
   
   ```bash
   cd scripts/
   python verify_installation.py
   ```
