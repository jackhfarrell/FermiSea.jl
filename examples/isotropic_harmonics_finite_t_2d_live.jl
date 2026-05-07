# Live finite-temperature harmonic-moment example.
#
# Run from the repository root:
#
#     julia --project=. examples/isotropic_harmonics_finite_t_2d_live.jl
#
# This script requires GLMakie for the live window. If it is not installed in
# the active project, add it with:
#
#     import Pkg; Pkg.add("GLMakie")

using FermiSea
using OrdinaryDiffEqSSPRK
using StaticArrays
using Trixi

try
    @eval using GLMakie
catch err
    @error "GLMakie is required for the live visualization. Install it with `import Pkg; Pkg.add(\"GLMakie\")`."
    rethrow(err)
end

const ROOT = dirname(@__DIR__)
const MESH_FILE = joinpath(ROOT, "assets", "square_bells", "square_bells.inp")

mesh = P4estMesh{2}(MESH_FILE;
                    polydeg=1,
                    boundary_symbols=[:contact_bottom, :contact_top, :walls])

equations = IsotropicHarmonicsFiniteT2D(3, 5;
                                        mass=1.0,
                                        mu0=1.0,
                                        temperature=0.05,
                                        zmax=20.0,
                                        n_quad=128)

collisions = LinearCollisionMatrix(equations; gamma_mr=0.2, gamma_mc=15.0)

initial_condition(x, t, equations) =
    zero(SVector{nvariables(equations), Float64})

# Ohmic contacts prescribe the incoming density seed locally. This keeps the live
# demo simple while still exercising the finite-temperature boundary projector.
boundary_conditions = (;
    contact_bottom = OhmicContactBC(-0.04),
    contact_top = OhmicContactBC(0.04),
    walls = MaxwellWallBC(1.0),
)

solver = DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms=collisions,
                                    boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, (0.0, 8.0))

callbacks = CallbackSet(
    SummaryCallback(),
    AnalysisCallback(semi; interval=200, analysis_errors=Symbol[]),
    StepsizeCallback(; cfl=0.45),
)

integrator = init(ode, SSPRK43();
                  adaptive=false,
                  dt=5.0e-4,
                  callback=callbacks,
                  save_everystep=false)

function trixi_makie_extension()
    ext = Base.get_extension(Trixi, :TrixiMakieExt)
    ext === nothing &&
        error("Trixi's Makie extension did not load. Make sure GLMakie is loaded before plotting.")
    return ext
end

function field_data(u, semi, equations)
    plot_data = Trixi.PlotData2D(u, semi)
    density_index = 1
    jx_index = equations.n_radial + 1
    jy_index = 2 * equations.n_radial + 1

    density = vec(getindex.(plot_data.data, density_index)) # a0_r0
    jx = vec(getindex.(plot_data.data, jx_index))           # a1_r0
    jy = vec(getindex.(plot_data.data, jy_index))           # b1_r0
    current = sqrt.(jx.^2 .+ jy.^2)
    return plot_data, density, current
end

function scaled_extrema(values)
    lo, hi = extrema(values)
    if lo ≈ hi
        delta = max(abs(lo), one(abs(lo))) * 1.0e-5
        return (lo - delta, hi + delta)
    end
    return (lo, hi)
end

plot_data0, density0, current0 = field_data(integrator.u, semi, equations)
ext = trixi_makie_extension()

# Use Trixi's Makie triangulation helper rather than flattening the DG nodes into
# an unstructured scatter plot. The geometry is fixed; the live loop updates only
# the per-vertex colors.
plotting_mesh = ext.global_plotting_triangulation_makie(getindex(plot_data0, "a0_r0");
                                                        set_z_coordinate_zero=true)

fig = Figure(; size=(980, 430))
ax_density = Axis(fig[1, 1];
                  aspect=DataAspect(),
                  xlabel="x",
                  ylabel="y",
                  title="density seed a0_r0")
ax_current = Axis(fig[1, 2];
                  aspect=DataAspect(),
                  xlabel="x",
                  ylabel="y",
                  title="current magnitude from a1_r0,b1_r0")

density_obs = Observable(density0)
current_obs = Observable(current0)
density_range_obs = Observable(scaled_extrema(density0))
current_range_obs = Observable(scaled_extrema(current0))
time_obs = Observable("t = 0.000")

density_plot = mesh!(ax_density, plotting_mesh;
                     color=density_obs,
                     colorrange=density_range_obs,
                     colormap=:balance,
                     shading=NoShading)
current_plot = mesh!(ax_current, plotting_mesh;
                     color=current_obs,
                     colorrange=current_range_obs,
                     colormap=:viridis,
                     shading=NoShading)

wire_points = ext.convert_PlotData2D_to_mesh_Points(getindex(plot_data0, "a0_r0");
                                                    set_z_coordinate_zero=true)
lines!(ax_density, wire_points; color=(:black, 0.2), linewidth=0.5)
lines!(ax_current, wire_points; color=(:black, 0.2), linewidth=0.5)

Colorbar(fig[1, 3], density_plot; label="a0_r0")
Colorbar(fig[1, 4], current_plot; label="|j|")
Label(fig[2, 1:4], time_obs; tellwidth=false)
xlims!(ax_density, extrema(plot_data0.x))
ylims!(ax_density, extrema(plot_data0.y))
xlims!(ax_current, extrema(plot_data0.x))
ylims!(ax_current, extrema(plot_data0.y))

display(fig)

steps_per_frame = 50
while integrator.t < last(ode.tspan)
    for _ in 1:steps_per_frame
        integrator.t >= last(ode.tspan) && break
        step!(integrator)
    end

    _, density, current = field_data(integrator.u, semi, equations)
    density_obs[] = density
    current_obs[] = current
    density_range_obs[] = scaled_extrema(density)
    current_range_obs[] = scaled_extrema(current)
    time_obs[] = "t = $(round(integrator.t; digits=3))"
    sleep(0.001)
end

integrator
