# # Square Bells

# This example runs a small electron flow problem in the "square bells" geometry from 
# ([arXiv:2603.11175](https://arxiv.org/abs/2603.11175)).  We timestep the equations from 
# a trivial initial condition until it reaches a steady state.  Then we show how to 
# save and visualize the solution.
# 
# To run this example, from the repo root run the following command which should take 
# around 10 minutes on a laptop:
# ```bash
# julia --project=. docs/src/tutorials/square_bells.jl
# ```
# If needed, first add the script packages to the repo-root environment:
# ```julia
# import Pkg
# Pkg.add(["CairoMakie", "Interpolations", "OrdinaryDiffEqSSPRK"])
# ```
# First, load required packages.
using FermiSea
using CairoMakie
using HDF5
using Interpolations
using Logging
using OrdinaryDiffEqSSPRK
using StaticArrays
using Trixi

# Next, load the mesh. 
# Trixi can load an external 2D Abaqus-style mesh with `P4estMesh{2}`. The
# boundary names in the mesh file are the symbols we use later in the boundary
# condition tuple.

mesh = P4estMesh{2}(
    joinpath(@__DIR__, "..", "..", "..", "assets", "square_bells","square_bells.inp"); 
    polydeg=1,
    boundary_symbols=[:contact_bottom, :contact_top, :walls]
);

# Next, we choose which equations we want to solve.
# `IsotropicFermiHarmonics2D` keeps angular harmonics in the order
# `a0, a1, b1, a2, b2, ...`. The first mode is the density-like mode, while
# `a1` and `b1` are proportional to the current.  Let's keep 20 modes for now, which is
# plenty to get a good approximation in the hydrodynamic limit, though we want up to 200
# if we have weaker collisions.
equations = IsotropicFermiHarmonics2D(20; v_fermi=1.0);

# Next let's add the collision term. 
# A collision operator is just a Trixi source term. If you only need collisions,
# pass it directly as `source_terms`. If you want to combine collisions with a
# magnetic field source, wrap them with `SourceTerms(collision, magnetic_source)`.
collisions = LinearCollisionMatrix(equations; gamma_mr=0.0, gamma_mc=100.0);

# as an initial condition, let's just start with zero everywhere. 
initial_condition(x, t, equations) =
    zero(SVector{nvariables(equations), typeof(equations.v_fermi)});

# Next, we specify our boundary conditions.
# `CurrentContactBC(I)` enforces a total integrated current `I` through the whole
# named contact by solving for one shared contact potential. `MaxwellWallBC(1)`
# gives fully diffuse walls.
boundary_conditions = (;
    contact_bottom = CurrentContactBC(-0.1),
    contact_top = CurrentContactBC(0.1),
    walls = MaxwellWallBC(1.0),
);

# Next, we set up Trixi's solver, which is discontinuous Galerkin spectral element method
# (DGSEM), with a Lax-Friedrichs surface flux. The `polydeg` argument sets the polynomial
# degree for the DG method.
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs);

# then build the descretized system:
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms=collisions,
                                    boundary_conditions=boundary_conditions);

# which then just reduces to an ODE problem that we can solve with any ODE solver in
# Julia, like this.
ode = semidiscretize(semi, (0.0, 100.0));

# It is often the case that we want to do things during the solve, like printing log 
# messages, calcualting diagnostics, saving the solution, etc.  These operations, just
# functions that we want to call at various points during the solve, are called "callbacks".
# Here we set up a few, with the most important being the SteadyStateCallback, which will
# stop the solve once the solution has reached a steady state with some tolerance. 
callbacks = CallbackSet(
    SummaryCallback(),
    SteadyStateCallback(abstol=1e-2, reltol=0.0),
    AnalysisCallback(semi; interval=2000, analysis_errors=Symbol[]),
    StepsizeCallback(; cfl=0.5),
);

# finally, we just solve!
sol = solve(ode, SSPRK43(); adaptive=false, dt=1.0e-3, callback=callbacks,
            save_everystep=false);

# For visualization it is convenient to sample the DG solution on a Cartesian
# grid. The package can write that grid to HDF5; this is also a handy format for
# post-processing in Python, MATLAB, ParaView-adjacent scripts, or another Julia
# session.
cartesian_file = joinpath(@__DIR__, "square_bells_cartesian.h5");
with_logger(NullLogger()) do
    save_cartesian(sol, semi, cartesian_file; nvisnodes=200)
end;

# then we read it as follows and visualize in Makie
x, y, a1, b1, mask = h5open(cartesian_file, "r") do file
    read(file["x"]), read(file["y"]), read(file["a1"]), read(file["b1"]),
    read(file["mask"])
end;

current_magnitude = sqrt.(a1.^2 .+ b1.^2);
current_magnitude[.!mask] .= NaN;
stream_x = a1;
stream_y = b1;
stream_x[.!mask] .= 0;
stream_y[.!mask] .= 0;

stream_x_itp = linear_interpolation((x, y), stream_x; extrapolation_bc=0.0);
stream_y_itp = linear_interpolation((x, y), stream_y; extrapolation_bc=0.0);

current_vector(px, py) =
    Point2f(stream_x_itp(px, py), stream_y_itp(px, py));

fig = Figure();
ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y",
          title="Current streamlines");
hm = heatmap!(ax, x, y, current_magnitude; colormap=:viridis);
x_interval = CairoMakie.Makie.IntervalSets.ClosedInterval(minimum(x), maximum(x));
y_interval = CairoMakie.Makie.IntervalSets.ClosedInterval(minimum(y), maximum(y));
streamplot!(ax, current_vector, x_interval, y_interval;
            color=Returns(:white), linewidth=1.2, density=0.8,
            gridsize=(28, 28), arrow_size=8);
Colorbar(fig[1, 2], hm; label="|j|");
fig
