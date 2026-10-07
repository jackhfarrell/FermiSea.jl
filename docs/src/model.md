# [Model and conventions](@id model-conventions)

We work with one circular Fermi surface at zero temperature. The state
``\phi(\mathbf{x},\theta)`` is an electrochemical energy departure, sampled at
angles around the circle. We write the change in the distribution as

```math
\delta f = \delta(\epsilon-\mu)\bigl(\phi-q\Phi\bigr).
```

Here ``q`` is the signed carrier charge and ``\Phi`` is the electrostatic
potential. For a device driven by reservoir bias, we solve the steady equation

```math
\mathbf{v}\cdot\nabla\phi + \omega\,\partial_\theta\phi
+ \mathcal{C}\phi = 0,
\qquad \omega=-qB\frac{v_F}{p_F}.
```

Here ``\mathcal{C}`` is the collision operator. We also support bulk driving
forces; see [Collisions and forcing](@ref).

The departure has energy units. Use consistent model units for velocities,
collision rates, magnetic fields, mesh coordinates, and reservoir offsets.
We have already absorbed the scalar electrostatic potential into ``\phi``, so
we must not add it again as a source. We use this convention for DC linear
response; more general settings require a separate treatment of electrostatics.

## Circular Fermi surface

[`circular_fermi_surface`](@ref) samples the circle at midpoint angles
``\theta_i=2\pi(i-1/2)/N``. Momenta and velocities point in the same direction
with magnitudes ``p_F`` and ``v_F``. All node energies equal ``\mu``.
We require an even ``N\geq4`` so opposite velocity pairs balance wall flux.

Each response weight is

```math
w_i=\frac{g_s p_F}{2\pi v_F N},
\qquad \sum_i w_i=\frac{g_s p_F}{2\pi v_F}.
```

These response weights include the spin degeneracy ``g_s`` and phase-space
factor ``(2\pi)^{-2}``. We keep ``\phi`` unweighted and apply the weights when
taking moments or inner products. The local inner product is
``\sum_i w_i a_i b_i``. Spatial norms also include the discontinuous Galerkin
(DG) mass. We store all angular values at one spatial node before moving to the
next spatial node.

## State and streaming

At each spatial node, we keep ``N`` values
``\phi_i=\phi(\mathbf{x},\theta_i)``. Each value streams along its own velocity:

```math
(F_x)_i=v_{ix}\phi_i,\qquad (F_y)_i=v_{iy}\phi_i.
```

[`flux_upwind`](@ref) selects the state on the upwind side separately for each
normal velocity. `nangle` sets angular resolution; `degree` and the mesh set
spatial resolution. We need to resolve both.

## Collisions and forcing

We use [`Callaway`](@ref) for a toy two-rate collision model:

```math
\mathcal{C}=\gamma_{\rm mr}(I-P_n)+\gamma_{\rm mc}(I-P_{n\mathbf{p}}).
```

``P_n`` keeps the constant population mode. ``P_{n\mathbf{p}}`` also keeps
``p_x`` and ``p_y``. Both are orthogonal projectors in the weighted inner
product, so ``\mathcal{C}`` is self-adjoint and nonnegative. Momentum modes decay at
``\gamma_{\rm mr}``, while higher harmonics decay at
``\gamma_{\rm mr}+\gamma_{\rm mc}``. There is no separate energy mode because
every node lies at ``\mu``.

[`TomographicCollision`](@ref) assigns separate rates to odd and even harmonics.
We apply it using FFTs.
[`CustomCollision`](@ref) accepts a matrix or action callback.
With a custom law, we need to check conservation and positivity ourselves.

[`MagneticField`](@ref) differentiates in the counterclockwise angle using FFTs.
Its sign follows ``d\mathbf{p}/dt=qB(v_y,-v_x)``. We set the derivative of the
even angular-grid Nyquist mode to zero to keep the FFT derivative real. Trixi uses the
negative collision and magnetic actions as sources because these terms sit on
the left of the steady equation.

[`ElectricDrive`](@ref) accepts the electric field ``\mathbf{E}`` and applies
``q\mathbf{E}\cdot\mathbf{v}`` once. The channel interface accepts the field along
its axis.

## Boundaries and observables

Normals point out of the device. Positive normal velocity leaves it, and negative
normal velocity enters it. [`FixedReservoir`](@ref) sets the incoming energy
departure. [`FloatingContact`](@ref) finds the departure needed for a prescribed
outward charge current (often zero, to model a voltage probe).

[`DiffuseWall`](@ref) emits a constant incoming departure chosen to give zero net
particle flux. [`MaxwellWall`](@ref) mixes this with specular reflection by
matching flux in tangential-momentum order. This preserves a constant equilibrium
and particle flux on the discretized surface. Increase its angular resolution alongside
the rest of the calculation.

For real devices, we often use a mixture of diffuse and specular reflection.
We use ``p`` for the diffuse scattering probability:
a fraction ``p`` of particles reflects diffusely, losing memory of its incident
direction, and the remaining fraction ``1-p`` reflects specularly, preserving
tangential momentum. This gives us a simple way to represent boundary roughness
without resolving it explicitly. Use `MaxwellWall(p_scatter=p)`, with
``p=0`` for fully specular reflection and ``p=1`` for fully diffuse reflection.
Both contributions conserve particle flux, so their mixture does too.

[`particle_current`](@ref) gives ``\sum_i w_i\mathbf{v}_i\phi_i``.
[`charge_current`](@ref) multiplies it by ``q``.
[`particle_density`](@ref) is an electrochemical density-like moment. Recovering
physical induced density requires separate electrostatic information.
For total contact flow, we integrate the boundary numerical flux with the same
outward sign convention.
