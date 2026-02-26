ARG PYTHON_BASE_IMAGE=python:3.12-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SIMIND_ROOT=/workspace/simind
ENV SIMIND_BIN=${SIMIND_ROOT}/simind
ENV SIMIND_MPI_BIN=${SIMIND_ROOT}/simind_mpi
ENV SIMIND_SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIMIND_DATA_DIR=${SIMIND_ROOT}/smc_dir/
ENV SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIRF_SIMIND_BLOCK_IMPORTS=sirf,stir,stirextra,pytomography
ENV PYTHONPATH=/workspace/docker/import_guard:${PYTHONPATH}
ENV PATH=${SIMIND_ROOT}/bin:${SIMIND_ROOT}:${PATH}

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --upgrade pip \
    && pip install -e ".[dev,examples]" \
    && python - <<'PY'
import importlib
for module in ("numpy", "matplotlib", "pytest"):
    importlib.import_module(module)
print("Core Python modules available")
PY

CMD ["python", "examples/06_schneider_density_conversion.py"]
