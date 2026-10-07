# [Currents and saved results](@id analysis)

## Local currents

We get local currents by integrating the angular state at a spatial point.
The state is unweighted; [`particle_current`](@ref) and [`charge_current`](@ref)
include the response weights.

Let's use `solution` from the
[README quickstart](https://github.com/jackhfarrell/FermiSea.jl#quickstart):

```julia
surface = solution.problem.semi.equations.surface
phi = state(solution)
local_particle_current = particle_current(view(phi, :, 1), surface)
local_charge_current = charge_current(view(phi, :, 1), surface)
```

Each column of `phi` holds the angular values at one spatial DG node. Charge
current is particle current multiplied by the signed carrier charge.
`state(solution)` is a view, so changing `phi` changes the solution.

[`particle_density`](@ref) gives the electrochemical density-like moment.
[`momentum_moment`](@ref) integrates a custom kernel, and
[`fit_local_equilibrium`](@ref) separates a constant departure and drift from the
higher angular content.

## Contact fluxes

For total flow through a contact, we use the DG boundary numerical flux.
Positive [`contact_particle_flux`](@ref) leaves the device.
[`contact_current`](@ref) multiplies it once by the signed carrier charge.

For the rectangle in the quickstart, the right contact is `:x_pos`:

```julia
contact_particle_flux(solution, :x_pos)
contact_current(solution, :x_pos)
```

Particles enter on the left and leave on the right, so the contact fluxes have
opposite signs. Their sum should be zero up to solver error. With `charge=-1`,
charge current has the opposite sign to particle flux.

## Sampling

To get the current at a point, make a [`SpatialSampler`](@ref):

```julia
current_at = SpatialSampler(solution, particle_current)
current_at(0.0, 0.0)
```

The sampler interpolates the departure before evaluating the current. With no
observable, it returns a copied angular state. Outside the device, it returns
NaNs by default; use `outside=:nothing` or `outside=:error` if you prefer.
Each sampler has scratch arrays, so use a separate one for each concurrent task.

For saved fields, the sampler interpolates the observable values instead. For a
nonlinear observable, these two operations need not give the same result.

## Plotting

Now plot the current. Colour shows its magnitude; streamlines and arrows show
where it goes:

```julia
using CairoMakie
figure = plot(solution, particle_current)
streamplot!(content(figure[1, 1]), solution, particle_current;
            color = _ -> :black, linewidth = 1, gridsize = (16, 16), density = 0.5)
content(figure[1, 2]).label = "|particle current| (model units)"
save("current.png", figure)
```

Pass `component=:x` or `component=:y` to `plot` to see a signed component.
Axes and currents use model units.

Use `plot!` to draw a field on an existing axis. [`PlotData`](@ref) stores
observables at the DG nodes when we only need the fields. It accepts scalars,
two-component vectors, and nested NamedTuples. Give a custom
callback a name with `:name => callback`. `refine=1` uses the DG nodes; larger
values make a smoother display without adding physical resolution.

## Save and load

Save the solution if we want to come back to the angular state. The file also
contains the surface, mesh, boundaries, sources, solver diagnostics, contact results,
and selected observables. [`read_solution`](@ref) rebuilds the problem and checks
the geometry. Preconditioner factors are rebuilt when needed. Custom collision
callbacks cannot be saved.

```julia
save_solution(solution, "state.h5")
loaded = read_solution("state.h5")
stored_current = read_plotdata("state.h5")
plot(stored_current, :particle_current)
```

`loaded` has the angular state for further analysis. `stored_current` reads the
saved observables and can be plotted without reconstructing the problem.

[`save_surface`](@ref) and [`read_surface`](@ref) preserve model parameters and numeric
quadrature. Surface and solution files use format version two. State files include
an unstructured mesh or a uniformly refined `TreeMesh`. The readers check
geometry and quadrature. They cannot read the v0.1 files that stored harmonics.

[`save_plotdata`](@ref) writes observables without a momentum state.
[`read_plotdata`](@ref) reads those files or the observable section of a solution.
Observable files retain format version one. `save_solution(...; state=false)`
also produces an observable-only file and must be read with `read_plotdata`.
We finish writing a temporary file before replacing the destination. Optional
compression preserves numeric values.

Saved fields can be refined for display, but we need the original state to
evaluate a different observable. Keep a solution file if we might need it later.
