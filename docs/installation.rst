.. _installation:

Installation
============

Prerequisites
-------------

- Python 3.9+
- Optional backend libraries depending on workflow:
  - STIR Python for STIR-based paths
  - SIRF for SIRF-based paths
  - PyTomography for PyTomography adaptor paths

Install the Python Package
--------------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/samdporter/SIRF-SIMIND-Connection.git
      cd SIRF-SIMIND-Connection

2. Install in editable/development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

SIMIND Requirement (External Dependency)
----------------------------------------

SIMIND is **not** distributed in this repository (or on PyPI) and must be
installed separately by the user.

Use the official SIMIND resources for installation and manual/reference
documentation:

- SIMIND site (Medical Radiation Physics, Lund University):
  https://www.msf.lu.se/en/research/simind-monte-carlo-program
- SIMIND manual page:
  https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

Recommended Local Layout
~~~~~~~~~~~~~~~~~~~~~~~~

For this repository's scripts and Docker setup, place SIMIND under:

.. code-block:: text

    ./simind

and ensure these paths exist:

.. code-block:: text

    ./simind/simind
    ./simind/smc_dir/

If needed, make the binary executable:

.. code-block:: bash

    chmod +x ./simind/simind

Local (Non-Docker) Runtime Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For direct local runs, ensure SIMIND is on ``PATH`` and ``SMC_DIR`` points to
the SIMIND data directory:

.. code-block:: bash

    export PATH="$PWD/simind:$PATH"
    export SMC_DIR="$PWD/simind/smc_dir/"

Docker Behavior
~~~~~~~~~~~~~~~

The Docker Compose services and container helper scripts are configured for the
repo-local SIMIND layout above. They automatically wire SIMIND paths inside the
containers when ``./simind/simind`` is present.

Quick Verification
------------------

Verify basic Python package import:

.. code-block:: bash

    python -c "import sirf_simind_connection as s; print(s.__version__)"

Verify SIMIND detection from this repo root:

.. code-block:: bash

    test -x ./simind/simind && echo "SIMIND found" || echo "SIMIND missing"
