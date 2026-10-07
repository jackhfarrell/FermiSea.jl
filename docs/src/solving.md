# [Solving](@id solving)

To simulate a device, we put the model, mesh, boundaries, and sources into a
[`BoltzmannProblem`](@ref), then call `solve(problem; tolerance=1e-9)`. We use GMRES to solve the steady
equation ``A\phi=b``, with Trixi's transport residual giving the action of ``A``.
Set `degree` at least as high as the mesh's geometry degree.

## Changing parameters

Pass an operator, a tuple of operators, or [`SourceTerms`](@ref) to `source_terms`.
The problem allocates the scratch arrays needed by FFT and custom actions. If
we build a Trixi semidiscretization directly, we need to call
[`prepare_source`](@ref) ourselves.

Build a new problem when changing a rate, field, mesh, or callback parameter.
Changing a source or surface in place can leave the problem using stale data.
Sources and preconditioners also have scratch arrays, so prepare them separately
for each concurrent solve.

## Checking the solve

We can check how well the solution satisfies the discrete kinetic equation, using the
momentum response weights and DG spatial mass:

```math
r=\frac{\lVert A\phi-b\rVert_W}{s}.
```

The scale ``s`` is the weighted norm of ``b``. When ``b`` vanishes, we use the
initial residual norm, or one in solver units if that also vanishes. We check
the kinetic equation after GMRES finishes, so `solution.residual` is the
residual of the full problem.
It must be at most `tolerance`.

A failed solve throws an error. To inspect an unfinished solve, set
`throw_on_failure=false` and look at `converged`, `status`, and `residual`.
[`GMRES`](@ref) controls restart length, iteration limit, and preconditioning.

## Warm starts

We often solve the same device several times while varying a scattering rate.
For a rate sweep, the previous solution is a useful starting guess. Let's change
the collision rate in the
[README quickstart](https://github.com/jackhfarrell/FermiSea.jl#quickstart), keeping
its surface, mesh, and boundaries:

```julia
collision = Callaway(model; gamma_mr = 0.01, gamma_mc = 25.0)
next_problem = BoltzmannProblem(model, mesh, boundaries;
                                source_terms = collision, degree = 2)
next_solution = solve(next_problem; initial_guess = solution)
```

The momentum nodes, velocities, weights, spatial coordinates, and degree must
match. Rates and fields can change. You can also supply a flat state vector.

## Floating contacts

[`FloatingContact`](@ref) finds the reservoir offset needed for its prescribed
outward charge current. The default zero current gives a voltage probe.
`current_tolerance` checks the contact currents separately from the kinetic
residual. If every contact floats, name a `reference` contact to fix the energy
gauge.

## Preconditioners

[`AutomaticPreconditioner`](@ref) selects [`MomentSweep`](@ref) for supported
Callaway problems on conforming unstructured meshes. The correction retains the
population, two momentum modes, and the stress modes
``p_x^2-p_y^2`` and ``p_xp_y``. Dependent modes are dropped on coarse angular grids.
Retaining stress gives the reduced system viscous momentum relaxation when
`gamma_mr=0`.
We use [`TransportSweep`](@ref) alone when
``v_F/(\gamma_{\mathrm{mr}}+\gamma_{\mathrm{mc}})`` is at least the device's
bounding-box diameter, including when both rates vanish. This is a heuristic
for the nearly collisionless limit, where the truncated moment system can have
undamped modes. Other supported sources also use `TransportSweep`. `TreeMesh`
uses no package preconditioner by default.
The exact kinetic residual always controls convergence.

We build the moment correction from the kinetic operator itself, using the same
streaming, boundaries, and sources as the full problem. We assemble independent
element neighborhoods together to avoid one full kinetic operator call per
reduced column.

`solve` prepares a preconditioner automatically. Julia frees its stored factors
when the preparation is no longer referenced and garbage collection runs, so we
usually do not need to manage this ourselves.

To reuse the factors across separate solves of the same problem, call
[`prepare_preconditioner`](@ref) once and pass the result to `GMRES`:

```julia
prepared = prepare_preconditioner(problem, AutomaticPreconditioner())
algorithm = GMRES(; precond = prepared)
solution = solve(problem, algorithm)
```

Keep `algorithm` for further solves. If we need to free the sparse factors
immediately, call [`release_preconditioner!`](@ref) when finished and do not reuse
the preparation afterward. Otherwise, cleanup is automatic. Separate calls to
`solve(problem)` each build a new preparation.

## Flow balance

If no source creates particles, reservoir particle fluxes should sum to zero.
Insulating walls should carry no net particle flux. We check these when setting
up a new device.
