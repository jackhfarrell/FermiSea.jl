# Changelog

## v0.2.0

We switch the primary angular discretization from harmonics to discrete
ordinates. We sample the departure at angles around one circular Fermi surface
at zero temperature, and each angular value streams along its own velocity.
The state stays unweighted; response weights enter when we take moments.

This is a breaking change to the transport API. We now build a
`BoltzmannEquation` from a `CircularFermiSurface`, put the mesh, boundaries, and
collisions into a `BoltzmannProblem`, and call `solve` for the steady state.
The old harmonic API is no longer supported by this interface.

We include Callaway, tomographic, and custom collisions, magnetic and electric
driving, fixed and floating contacts, and diffuse or mixed specular walls.
For `MaxwellWall`, `p_scatter` is the diffuse scattering probability.

The steady solver checks the full kinetic residual with angular weights and
the DG spatial mass. On unstructured meshes, `MomentSweep` corrects density,
momentum, and stress modes; nearly collisionless problems use `TransportSweep`.

We can sample currents, plot flow, and save the state or just its observables.
State and surface files use HDF5 format version 2. The new readers do not read
the old harmonic state files.

The docs include a square-bells tutorial and explain the model conventions,
solve checks, and saved results. We retain the original diffusive,
hydrodynamic, and ballistic comparison in the README.

## v0.1.1

We fix bugs in the harmonic implementation.

## v0.1.0

Our first release uses angular harmonics on a circular Fermi surface.

- `IsotropicFermiHarmonics2D` equations for 2D isotropic Fermi-surface harmonic
  expansions.
- Collision and magnetic-field source terms for Trixi.jl semidiscretizations.
- Contact and wall boundary conditions for current-driven and probe-style
  simulations.
- Analysis callbacks and HDF5 output helpers for visualization workflows.
- Documentation, reference pages, and worked tutorial examples.
