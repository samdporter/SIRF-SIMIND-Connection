ARG STIR_BASE_IMAGE=synerbi/sirf:latest
FROM ${STIR_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SIRF_INSTALL_PATH=/opt/SIRF-SuperBuild/INSTALL
ENV SIMIND_ROOT=/workspace/simind
ENV SIMIND_BIN=${SIMIND_ROOT}/simind
ENV SIMIND_MPI_BIN=${SIMIND_ROOT}/simind_mpi
ENV SIMIND_SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIMIND_DATA_DIR=${SIMIND_ROOT}/smc_dir/
ENV SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIRF_SIMIND_BLOCK_IMPORTS=sirf,pytomography
ENV PYTHONPATH=/workspace/docker/import_guard:${SIRF_INSTALL_PATH}/python:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib:${SIRF_INSTALL_PATH}/lib:${SIRF_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${SIMIND_ROOT}/bin:${SIMIND_ROOT}:${SIRF_INSTALL_PATH}/bin:${PATH}

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --upgrade pip \
    && pip install -e ".[dev,examples]" \
    && python - <<'PY'
import importlib
for module in ("stir", "stirextra", "pytest"):
    importlib.import_module(module)
print("STIR Python modules available")
PY

CMD ["python", "examples/07A_stir_adaptor_osem.py"]
