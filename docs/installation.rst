.. _installation:

Installation
============

Prerequisites
-------------

- Python 3.9+
- A separate SIMIND installation for running simulations. SIMIND is not
  distributed with this package or installed by pip.
- Optional backend libraries depending on workflow:
  - STIR Python for STIR-based paths
  - SIRF for SIRF-based paths
  - PyTomography for PyTomography adaptor paths

Install the Python Package
--------------------------

Install the released package from PyPI:

.. code-block:: bash

    pip install simind-python-connector

Import it with:

.. code-block:: python

    import simind_python_connector

Install the examples extra if you want to run the plotting examples:

.. code-block:: bash

    pip install "simind-python-connector[examples]"

Development Install
-------------------

For local development from a checkout:

.. code-block:: bash

    git clone https://github.com/samdporter/simind-python-connector.git
    cd simind-python-connector
    pip install -e ".[dev,examples]"

SIMIND Requirement (External Dependency)
----------------------------------------

Disclaimer
~~~~~~~~~~

This project is independent and is **not affiliated with, endorsed by, or
maintained by** the SIMIND project or Lund University.

SIMIND is **not** distributed with this package (or on PyPI) and must be
installed separately.

Use the official SIMIND resources for installation and manual/reference
documentation:

- SIMIND site (Medical Radiation Physics, Lund University):
  https://www.msf.lu.se/en/research/simind-monte-carlo-program
- SIMIND manual page:
  https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

Repo/Docker Layout
~~~~~~~~~~~~~~~~~~

For the repository helper scripts and Docker setup, place SIMIND under:

.. code-block:: text

    ./simind

and ensure these paths exist:

.. code-block:: text

    ./simind/simind
    ./simind/smc_dir/

If needed, make the binary executable:

.. code-block:: bash

    chmod +x ./simind/simind

Local Runtime Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

For direct local runs, ensure the SIMIND executable is available as ``simind``
on ``PATH``. If your SIMIND installation needs ``SMC_DIR``, set it to the
SIMIND data directory:

.. code-block:: bash

    export PATH="/path/to/simind/bin:$PATH"
    export SMC_DIR="/path/to/simind/smc_dir/"

When working from this repository with the layout above, those commands become:

.. code-block:: bash

    export PATH="$PWD/simind:$PATH"
    export SMC_DIR="$PWD/simind/smc_dir/"

Docker Behavior
~~~~~~~~~~~~~~~

The Docker Compose services and container helper scripts use the local SIMIND
layout above. They automatically wire SIMIND paths inside the containers when
``./simind/simind`` is present.

Quick Verification
------------------

Verify basic Python package import:

.. code-block:: bash

    python -c "import simind_python_connector as s; print(s.__version__)"

Verify SIMIND detection from this repo root:

.. code-block:: bash

    test -x ./simind/simind && echo "SIMIND found" || echo "SIMIND missing"
