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
                    polydeg=3,
                    boundary_symbols=[:contact_bottom, :contact_top, :walls])

equations = IsotropicHarmonicsFiniteT2D(5, 3;
                                        mass=1.0,
                                        mu0=1.0,
                                        temperature=0.05,
                                        zmax=20.0,
                                        n_quad=128,
                                        electrostatic_chi=1.0)

collisions = NonlinearBGKCollision(equations; gamma_mr=0.0, gamma_mc=200.0)

initial_condition(x, t, equations) =
    zero(SVector{nvariables(equations), Float64})

# These Ohmic contacts prescribe reservoir electrochemical potentials. With the
# parameters above, +/-0.1 is still below the earlier unstable large-bias runs.
# The reservoir state is isotropic and zero-drift, but at finite temperature it
# can populate multiple scalar radial modes through the nonlinear density/chemical-
# potential relation. Boundary projectors use the bare kinetic characteristic
# split; the gradual-channel conservative term stays out of the projector.
boundary_conditions = (;
    contact_bottom = OhmicContactBC(0.1),
    contact_top = OhmicContactBC(0.0),
    walls = MaxwellWallBC(1.0),
)
boundary_conditions_parabolic = (;
    contact_bottom = boundary_condition_do_nothing,
    contact_top = boundary_condition_do_nothing,
    walls = boundary_condition_do_nothing,
)

solver = DGSEM(polydeg=3,
               surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralFluxDifferencing(flux_central))

semi = if iszero(equations.electrostatic_chi)
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms=collisions,
                                 boundary_conditions=boundary_conditions)
else
    equations_parabolic = GradualChannelForce2D(equations)
    semi_local = SemidiscretizationHyperbolicParabolic(
        mesh, (equations, equations_parabolic), initial_condition, solver;
        solver_parabolic=ParabolicFormulationBassiRebay1(),
        source_terms=collisions,
        source_terms_parabolic=GradualChannelForceSource(),
        boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))
    # Ensure the finite-temperature boundary projector caches are populated before
    # the threaded boundary-flux kernels start using them.
    FermiSea.initialize_boundary_projectors!(semi_local)
    semi_local
end
ode = semidiscretize(semi, (0.0, 8.0))

function state_health_report(u, semi, equations)
    u_wrap = ndims(u) == 1 ? Trixi.wrap_array(u, semi) : u
    _, _, solver_local, cache_local = Trixi.mesh_equations_solver_cache(semi)

    min_density = Inf
    max_state_abs = 0.0
    bad_reason = nothing
    bad_element = 0
    bad_i = 0
    bad_j = 0
    bad_state = nothing
    bad_density = NaN
    bad_fields = nothing

    for element in Trixi.eachelement(solver_local, cache_local)
        for j in Trixi.eachnode(solver_local), i in Trixi.eachnode(solver_local)
            state = Trixi.get_node_vars(u_wrap, equations, solver_local, i, j, element)
            max_state_abs = max(max_state_abs, maximum(abs, state))

            if !all(isfinite, state)
                bad_reason = "non-finite state coefficients"
                bad_element, bad_i, bad_j = element, i, j
                bad_state = state
                bad_density = hydrodynamic_density(equations, state)
                break
            end

            density = hydrodynamic_density(equations, state)
            min_density = min(min_density, density)
            if !isfinite(density)
                bad_reason = "non-finite reconstructed density"
                bad_element, bad_i, bad_j = element, i, j
                bad_state = state
                bad_density = density
                break
            elseif density <= 0
                bad_reason = "non-positive reconstructed density"
                bad_element, bad_i, bad_j = element, i, j
                bad_state = state
                bad_density = density
                bad_fields = hydrodynamic_fields(equations, state)
                break
            end
        end
        bad_reason === nothing || break
    end

    return (; min_density, max_state_abs, bad_reason, bad_element, bad_i, bad_j,
            bad_state, bad_density, bad_fields)
end

function StateHealthCallback(semi, equations; interval=1)
    interval > 0 || throw(ArgumentError("interval must be positive"))

    affect! = function (integrator)
        report = state_health_report(integrator.u, semi, equations)
        if report.bad_reason !== nothing
            @error "state-health failure" t=integrator.t iter=integrator.iter element=report.bad_element node=(report.bad_i,
                                                                                                                  report.bad_j) reason=report.bad_reason density=report.bad_density maxabs=report.max_state_abs state=report.bad_state
            if report.bad_fields !== nothing
                @error "failing state hydrodynamics" velocity_x=report.bad_fields.velocity[1] velocity_y=report.bad_fields.velocity[2] speed=report.bad_fields.speed delta_mu=report.bad_fields.delta_mu momentum_x=report.bad_fields.momentum[1] momentum_y=report.bad_fields.momentum[2] electrochemical=report.bad_fields.electrochemical_potential
            end
            terminate!(integrator)
            return nothing
        end

        return nothing
    end

    return DiscreteCallback(
        (u, t, integrator) -> integrator.iter % interval == 0,
        affect!;
        save_positions=(false, false))
end

callbacks = CallbackSet(
    SummaryCallback(),
    AnalysisCallback(semi; interval=200, analysis_errors=Symbol[]),
    StateHealthCallback(semi, equations; interval=1),
    StepsizeCallback(; cfl=0.2),
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

    @inbounds for i in eachindex(states)
        fields = hydrodynamic_fields(equations, states[i])
        density_deviation[i] = fields.density_delta / equations.equilibrium_density
        electrochemical_potential[i] = fields.electrochemical_potential
        vy[i] = fields.velocity[2]
        speed[i] = fields.speed
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

for axis in (ax_density, ax_speed, ax_electrochemical, ax_vy)
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
