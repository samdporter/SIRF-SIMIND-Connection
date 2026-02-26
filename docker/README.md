Docker Environments
===================

This directory provides dedicated container setups for:

- Core Python workflows (no SIRF/STIR/PyTomography)
- STIR workflows
- SIRF workflows
- PyTomography workflows

Quick start
-----------

Build and run a specific environment:

```bash
docker compose -f docker/compose.yaml build stir
docker compose -f docker/compose.yaml run --rm stir
```

Run the dedicated connector + OSEM examples directly:

```bash
docker compose -f docker/compose.yaml run --rm stir python examples/07A_stir_adaptor_osem.py
docker compose -f docker/compose.yaml run --rm sirf python examples/07B_sirf_adaptor_osem.py
docker compose -f docker/compose.yaml run --rm pytomography python examples/07C_pytomography_adaptor_osem.py
```

Run the containerized test matrix:

```bash
bash scripts/run_container_validation.sh
```

Include SIMIND-dependent examples/integration tests:

```bash
bash scripts/run_container_validation.sh --with-simind
```

Run selected example groups:

```bash
bash scripts/run_container_examples.sh --only-core
bash scripts/run_container_examples.sh --only-pytomography
bash scripts/run_container_examples.sh --only-osem
```

Set a specific Docker platform when needed:

```bash
bash scripts/run_container_examples.sh --only-core --docker-platform linux/amd64
bash scripts/run_container_validation.sh --with-simind --docker-platform linux/amd64
```

Require SIMIND explicitly (fail fast if missing):

```bash
bash scripts/run_container_examples.sh --only-osem --require-simind
bash scripts/run_container_validation.sh --with-simind --require-simind
```

Services
--------

- `python`: Core Python image for non-backend examples/tests.
- `stir`: STIR-focused image (based on `synerbi/sirf`) with this repository installed.
- `sirf`: SIRF-focused image (based on `synerbi/sirf`) with this repository installed.
- `pytomography`: Python image with PyTomography + this repository installed.
- Each service enables an import guard so only its target library is importable
  during tests/examples (`stir`, `sirf.STIR`, or `pytomography` respectively;
  `python` blocks all three backend libraries).

Notes
-----

- The SIRF container build arg is `SIRF_BASE_IMAGE`.
- The STIR container build arg is `STIR_BASE_IMAGE`.
- Install SIMIND by following the official installation instructions on the SIMIND website.
- For quick local use with these containers, put SIMIND at `./simind` in this repo
  so the executable is available as `./simind/simind`.
- SIMIND-dependent examples require the `simind` executable inside the container.
  Runner scripts now auto-check `./simind/simind` and skip SIMIND-dependent checks
  when missing (or fail fast if `--require-simind` is used).
- For mixed host/runtime architectures, the runner scripts auto-detect SIMIND
  binary architecture (`x86_64`/`aarch64`) and set Docker platform automatically;
  use `--docker-platform` to override.
- `input.smc` is bundled in `sirf_simind_connection/configs`; SIMIND runtime checks
  only gate external executable availability.
- The toy adaptor examples use `sirf_simind_connection/configs/Example.yaml`,
  a reduced projection-space profile derived from `AnyScan.yaml`.
- These images are intended to run examples and interactive exploration.
