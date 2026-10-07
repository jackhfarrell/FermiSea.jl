# # [Square bells](@id square-bells)
#
# We solve one hydrodynamic flow in the square-bells device from
# [Zhang et al. (2026)](https://arxiv.org/abs/2603.11175). Particle flow goes from
# the bottom reservoir to the top. We use diffuse walls and fast
# momentum-conserving collisions.

using FermiSea
using CairoMakie
using Trixi

# ## Momentum space discretization
#
# We use 128 equally spaced angles with unit Fermi momentum and speed. The carrier
# charge is minus one and the spin degeneracy is one.

surface = circular_fermi_surface(;
    p_fermi = 1.0, v_fermi = 1.0, nangle = 128,
    charge = -1.0, spin_degeneracy = 1)
model = BoltzmannEquation(surface)

plot(surface, :nodes; arrows = true)

# ## Spatial discretization
#
# First, load the included mesh and impose reservoir energy departures of
# ``+\Delta\mu/2`` at the bottom and ``-\Delta\mu/2`` at the top. Diffuse walls
# conserve particles but remove tangential momentum.

mesh_file = joinpath(pkgdir(FermiSea), "examples", "assets", "square_bells.mesh")
mesh = UnstructuredMesh2D(mesh_file)

bias = 0.2
boundaries = (; contact_bottom = FixedReservoir(bias / 2),
                contact_top = FixedReservoir(-bias / 2), walls = DiffuseWall())
nothing #hide

# We use ``\gamma_{\mathrm{mc}}=100`` and ``\gamma_{\mathrm{mr}}=0.05`` in inverse
# model time units. With unit Fermi speed, the momentum-conserving collision length
# is ``0.01`` in mesh units, short compared with the constriction width. Momentum
# relaxation is much rarer, with a length of ``20``. These separated lengths put
# the flow in the hydrodynamic regime.

collisions = Callaway(model; gamma_mr = 0.05, gamma_mc = 100.0)
nothing #hide

# ## Solve
#
# We use degree-three DG polynomials and stop at a relative residual of
# ``10^{-6}`` in the norm weighted by spatial quadrature and momentum weights. For
# ``A\phi=b``, the stopping test is
#
# ```math
# \frac{\lVert A\phi-b\rVert_{L^2(\mathbf{x},\mathbf{p})}}
#      {\lVert b\rVert_{L^2(\mathbf{x},\mathbf{p})}} \leq 10^{-6}
# ```

problem = BoltzmannProblem(model, mesh, boundaries;
                           source_terms = collisions, degree = 3)
solution = solve(problem; tolerance = 1e-6)

@assert solution.converged

# ## Save
#
# Save the full angular state so we can calculate other observables later. Outputs
# go in `examples/output/square_bells` inside the package checkout.

output_directory = joinpath(pkgdir(FermiSea), "examples", "output", "square_bells")
mkpath(output_directory)
save_solution(solution, joinpath(output_directory, "state.h5"))
nothing #hide

# ## Analysis
#
# Boundary fluxes use the outward normal, so bottom inflow is negative and top
# outflow is positive. The sum, including the walls, checks steady particle
# conservation.

flux_bottom = contact_particle_flux(solution, :contact_bottom)
flux_top = contact_particle_flux(solution, :contact_top)
flux_walls = contact_particle_flux(solution, :walls)
println("bottom particle flux: ", flux_bottom)
println("top particle flux: ", flux_top)
println("wall particle flux: ", flux_walls)
println("total outward particle flux: ", flux_bottom + flux_top + flux_walls)

# Now plot the current. Colour shows particle-current magnitude, and white
# streamlines follow the flow.
#
# Electrical current is the particle current multiplied by the signed carrier
# charge, so its direction is reversed here. Plot refinement changes only the
# display resolution, without changing the solution.

figure = Figure(; size = (900, 700))
axis = Axis(figure[1, 1]; aspect = DataAspect(),
            xlabel = "x (mesh units)", ylabel = "y (mesh units)")
magnitude = plot!(axis, solution, particle_current; refine = 5, colormap = :viridis)
streamplot!(axis, solution, particle_current;
            color = Returns(:white), linewidth = 1.1, density = 1.0, arrow_size = 8)
Colorbar(figure[1, 2], magnitude; label = "|particle current| (model units)")
figure

# Save the figure and current at the DG nodes. The observable file can be reloaded
# without rebuilding the kinetic state.

save(joinpath(output_directory, "current.png"), figure)
field_path = joinpath(output_directory, "current.h5")
save_plotdata(PlotData(solution, particle_current), field_path)
stored = read_plotdata(field_path)
sort!(collect(keys(stored.values)))

# To inspect the current at a point, interpolate the angular state and then
# evaluate the observable.

sampler = SpatialSampler(solution, particle_current)
sampler(0.0, 0.0)

# This tutorial is a Julia script in `docs/tutorials/square_bells.jl`. From a
# checkout, install the docs dependencies and run it with
#
# ```sh
# julia --project=docs -e 'using Pkg; Pkg.instantiate()'
# julia --project=docs docs/tutorials/square_bells.jl
# ```
#
# Refine `nangle` and `degree` to check resolution for quantitative work. See
# [Solving](@ref solving) for convergence checks, or
# [Currents and saved results](@ref analysis) for sampling and storage.
