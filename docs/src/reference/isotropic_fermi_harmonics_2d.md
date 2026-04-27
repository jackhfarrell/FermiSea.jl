# IsotropicFermiHarmonics2D

`IsotropicFermiHarmonics2D` discretizes a two-dimensional linear Boltzmann
equation for a perturbation ``\phi(\mathbf{x}, \theta, t)`` on an isotropic
Fermi surface:

```math
\partial_t \phi
+ v_F \hat{\mathbf{v}}(\theta) \cdot \nabla_{\mathbf{x}} \phi
+ \omega_c \partial_\theta \phi
= W[\phi].
```

Here ``v_F`` is the Fermi velocity, ``\hat{\mathbf{v}}(\theta)`` is the direction
on the Fermi surface, ``\omega_c`` is the cyclotron frequency from a
perpendicular magnetic field, and ``W`` is a linear collision operator.

The implementation expands ``\phi`` in angular harmonics and evolves the
truncated coefficient vector
``\mathbf{u} = (a_0, a_1, b_1, a_2, b_2, \ldots)``. After truncation,
FermiSea.jl solves a linear hyperbolic moment system of the form

```math
\partial_t \mathbf{u}
+ \partial_x(A_x \mathbf{u})
+ \partial_y(A_y \mathbf{u})
= S(\mathbf{u}),
```

where ``A_x`` and ``A_y`` are the harmonic streaming matrices and ``S`` collects
optional collision and magnetic-field source terms. Trixi.jl handles the spatial
discretization of this system; FermiSea.jl provides the harmonic equations,
source terms, and boundary conditions documented below.

```@docs
IsotropicFermiHarmonics2D
FermiSea.build_streaming_matrices
```

## Source Terms

```@docs
LinearCollisionMatrix
MagneticFieldSource
SourceTerms
```

## Boundary Conditions

```@docs
MaxwellWallBC
OhmicContactBC
FloatingProbeBC
CurrentContactBC
```
