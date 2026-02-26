Geometry Considerations
=======================

This page summarizes the geometry conventions used across SIMIND, STIR/SIRF,
and PyTomography in this repository.

At-a-Glance Axis Conventions
----------------------------

- STIR/SIRF image arrays are handled as ``(z, y, x)``.
- STIR/SIRF projection arrays are often 4D with a singleton TOF axis:
  ``(tof, bin, view, axial)``. For these, **view 0** is ``arr[0, :, 0, :]``.
- Raw Interfile projection loading in ``interfile_numpy`` is exposed as
  ``(views, bins, axial)`` when no singleton TOF axis is present.
- ``PyTomographySimindAdaptor`` public object-space tensors are ``(x, y, z)``.
- PyTomography object space is ``(x, y, z)`` (``Lx, Ly, Lz``).
- PyTomography SIMIND projections from ``pytomography.io.SPECT.simind`` are
  ``(theta, r, z)``.

Practical Implication
---------------------

When comparing STIR/SIRF and PyTomography outputs:

- Use PyTomography projections directly for reconstruction (no extra manual flip).
- Keep PyTomography object tensors in ``(x, y, z)`` end-to-end
  (input -> system matrix -> reconstruction).
- Connector internals convert object tensors to SIMIND image file order
  ``(z, y, x)`` when writing ``.smi/.dmi`` inputs.

Units
-----

- SIMIND geometry parameters are configured in **cm**.
- STIR/SIRF image geometry is typically expressed in **mm**.
- Connectors handle conversion internally via voxel-size settings (for example,
  runtime switch ``PX`` is set in cm for SIMIND).

Example Configuration Guardrails
--------------------------------

The OSEM examples intentionally pin key simulation parameters so geometry checks
are reproducible:

- ``NN=1`` (runtime switch) for faster, deterministic iteration.
- ``config[29]=30`` for projection count.
- ``config[53]=0`` to keep collimator modeling geometric-only in these tests.
- ``config[19]=2`` to keep a consistent mapping used by current examples.

Debug Checklist
---------------

If reconstruction geometry looks wrong:

1. Confirm array shape and axis order before plotting/reconstruction.
2. Confirm you are extracting projection **view 0** from the correct axis.
3. Confirm PyTomography object-space tensors are in ``(x, y, z)``.
4. Confirm attenuation-map orientation matches the reconstruction backend.
5. Compare hotspot center-of-mass between source and recon in a common axis convention.
