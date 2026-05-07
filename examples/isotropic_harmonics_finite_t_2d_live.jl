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
using LinearAlgebra
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

equations = IsotropicHarmonicsFiniteT2D(10, 2;
                                        mass=1.0,
                                        mu0=1.0,
                                        temperature=0.05,
                                        zmax=20.0,
                                        n_quad=128,
                                        electrostatic_chi=10.0)

collisions = NonlinearBGKCollision(equations; gamma_mr=0.0, gamma_mc=50.0)

initial_condition(x, t, equations) =
    zero(SVector{nvariables(equations), Float64})

# These Ohmic contacts prescribe reservoir electrochemical potentials. With the
# parameters above, +/-2.1 corresponds to density offsets close to
# delta_n / n0 = +/-0.1, or delta_mu / mu0 ~= +/-0.1. Boundary projectors at
# contacts use the full gradual-channel characteristic split; walls remain
# kinetic scattering boundaries.
boundary_conditions = (;
    contact_bottom = OhmicContactBC(2.1),
    contact_top = OhmicContactBC(-2.1),
    walls = MaxwellWallBC(1.0),
)

solver = DGSEM(polydeg=2,
               surface_flux=(flux_lax_friedrichs,
                             flux_electrostatic_nonconservative),
               volume_integral=VolumeIntegralFluxDifferencing(
                   (flux_central, flux_electrostatic_nonconservative)))
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
    states = vec(plot_data.data)
    density_deviation = Vector{Float64}(undef, length(states))
    electrochemical_potential = Vector{Float64}(undef, length(states))
    vy = Vector{Float64}(undef, length(states))
    speed = Vector{Float64}(undef, length(states))
    density_floor = max(sqrt(eps(Float64)) * equations.equilibrium_density,
                        sqrt(eps(Float64)))

    @inbounds for i in eachindex(states)
        state = states[i]
        density_delta = dot(equations.density_row, state)
        density = equations.equilibrium_density + density_delta
        density_for_diagnostics = max(density, density_floor)
        delta_mu = FermiSea._finite_chemical_potential_shift_from_density(equations,
                                                                          density_for_diagnostics)
        momentum = hydrodynamic_momentum(equations, state)
        velocity = momentum ./ (equations.mass * density_for_diagnostics)
        density_deviation[i] = density_delta / equations.equilibrium_density
        electrochemical_potential[i] = delta_mu + equations.electrostatic_chi *
                                       density_delta
        vy[i] = velocity[2]
        speed[i] = hypot(velocity[1], velocity[2])
    end

    return plot_data, density_deviation, electrochemical_potential, vy, speed
end

function scaled_extrema(values; symmetric=false)
    lo, hi = extrema(values)
    if symmetric
        width = max(abs(lo), abs(hi))
        if iszero(width)
            width = 1.0e-5
        end
        return (-width, width)
    end
    if lo ≈ hi
        delta = max(abs(lo), one(abs(lo))) * 1.0e-5
        return (lo - delta, hi + delta)
    end
    return (lo, hi)
end

plot_data0, density_deviation0, electrochemical0, vy0, speed0 =
    field_data(integrator.u, semi, equations)
ext = trixi_makie_extension()

# Use Trixi's Makie triangulation helper rather than flattening the DG nodes into
# an unstructured scatter plot. The geometry is fixed; the live loop updates only
# the per-vertex colors.
plotting_mesh = ext.global_plotting_triangulation_makie(getindex(plot_data0, "a0_r0");
                                                        set_z_coordinate_zero=true)

fig = Figure(; size=(1120, 820))
ax_density = Axis(fig[1, 1];
                  aspect=DataAspect(),
                  xlabel="x",
                  ylabel="y",
                  title="relative density δn/n0")
ax_speed = Axis(fig[1, 2];
                aspect=DataAspect(),
                xlabel="x",
                ylabel="y",
                title="fluid speed |v|")
ax_electrochemical = Axis(fig[2, 1];
                          aspect=DataAspect(),
                          xlabel="x",
                          ylabel="y",
                          title="electrochemical potential")
ax_vy = Axis(fig[2, 2];
             aspect=DataAspect(),
             xlabel="x",
             ylabel="y",
             title="fluid velocity vy")

density_obs = Observable(density_deviation0)
speed_obs = Observable(speed0)
electrochemical_obs = Observable(electrochemical0)
vy_obs = Observable(vy0)
density_range_obs = Observable(scaled_extrema(density_deviation0; symmetric=true))
speed_range_obs = Observable(scaled_extrema(speed0))
electrochemical_range_obs = Observable(scaled_extrema(electrochemical0;
                                                      symmetric=true))
vy_range_obs = Observable(scaled_extrema(vy0; symmetric=true))
time_obs = Observable("t = 0.000")

density_plot = mesh!(ax_density, plotting_mesh;
                     color=density_obs,
                     colorrange=density_range_obs,
                     colormap=:balance,
                     shading=NoShading)
speed_plot = mesh!(ax_speed, plotting_mesh;
                   color=speed_obs,
                   colorrange=speed_range_obs,
                   colormap=:magma,
                   shading=NoShading)
electrochemical_plot = mesh!(ax_electrochemical, plotting_mesh;
                             color=electrochemical_obs,
                             colorrange=electrochemical_range_obs,
                             colormap=:balance,
                             shading=NoShading)
vy_plot = mesh!(ax_vy, plotting_mesh;
                color=vy_obs,
                colorrange=vy_range_obs,
                colormap=:balance,
                shading=NoShading)

wire_points = ext.convert_PlotData2D_to_mesh_Points(getindex(plot_data0, "a0_r0");
                                                    set_z_coordinate_zero=true)
for axis in (ax_density, ax_speed, ax_electrochemical, ax_vy)
    lines!(axis, wire_points; color=(:black, 0.18), linewidth=0.45)
    xlims!(axis, extrema(plot_data0.x))
    ylims!(axis, extrema(plot_data0.y))
end

Colorbar(fig[1, 3], density_plot; label="δn/n0")
Colorbar(fig[1, 4], speed_plot; label="|v|")
Colorbar(fig[2, 3], electrochemical_plot; label="δμ + χδn")
Colorbar(fig[2, 4], vy_plot; label="vy")
Label(fig[3, 1:4], time_obs; tellwidth=false)

display(fig)

steps_per_frame = 50
while integrator.t < last(ode.tspan)
    for _ in 1:steps_per_frame
        integrator.t >= last(ode.tspan) && break
        step!(integrator)
    end

    _, density_deviation, electrochemical, vy, speed =
        field_data(integrator.u, semi, equations)
    density_obs[] = density_deviation
    speed_obs[] = speed
    electrochemical_obs[] = electrochemical
    vy_obs[] = vy
    density_range_obs[] = scaled_extrema(density_deviation; symmetric=true)
    speed_range_obs[] = scaled_extrema(speed)
    electrochemical_range_obs[] = scaled_extrema(electrochemical; symmetric=true)
    vy_range_obs[] = scaled_extrema(vy; symmetric=true)
    time_obs[] = "t = $(round(integrator.t; digits=3))"
    sleep(0.001)
end

integrator
