using FermiSea
using Trixi
using CairoMakie

"""
    square_bells_sweep(; rates, gamma_mr=0.05, nangle=16, degree=1,
                         bias=0.2, tolerance=1e-6, map_indices, mesh_path)

Sweep momentum-conserving rates in the square-bells device with unit Fermi speed
and momentum, signed charge -1, and diffuse walls. Rates have inverse-time units.
The streaming length at a rate `gamma` is `1/gamma` in mesh units. Reservoirs
impose bottom/top energy departures `+bias/2` and `-bias/2`.

Return rates, outward top particle flux divided by `bias`, selected solutions,
and solve timings. This response is in model units, not SI electrical conductance.
The default resolution demonstrates the workflow and needs refinement for research.
"""
function square_bells_sweep(; rates = [0.02, 0.1, 0.5, 2.0, 10.0],
                            gamma_mr = 0.05, nangle = 16, degree = 1,
                            bias = 0.2, tolerance = 1e-6,
                            map_indices = [1, 3, 5],
                            mesh_path = joinpath(@__DIR__, "assets", "square_bells.mesh"))
    isfinite(bias) && bias != 0 || throw(ArgumentError("bias must be finite and nonzero"))
    all(i -> i in eachindex(rates), map_indices) || throw(ArgumentError("invalid map index"))
    model = BoltzmannEquation(circular_fermi_surface(;
        p_fermi = 1.0, v_fermi = 1.0, nangle, charge = -1.0))
    mesh = UnstructuredMesh2D(mesh_path)
    boundaries = (; contact_bottom = FixedReservoir(bias / 2),
                    contact_top = FixedReservoir(-bias / 2), walls = DiffuseWall())
    response = Float64[]
    selected = Dict{Int,BoltzmannSolution}()
    seconds = Float64[]
    previous = nothing
    for (index, gamma_mc) in enumerate(rates)
        collision = Callaway(model; gamma_mr, gamma_mc)
        problem = BoltzmannProblem(model, mesh, boundaries; source_terms = collision, degree)
        elapsed = @elapsed solution = solve(problem; tolerance, initial_guess = previous)
        push!(seconds, elapsed)
        push!(response, contact_particle_flux(solution, :contact_top) / bias)
        index in map_indices && (selected[index] = solution)
        previous = solution
        @info "square-bells" gamma_mc residual = solution.residual iterations = solution.iterations seconds = elapsed
    end
    return (; rates = collect(rates), response, selected, seconds, bias, gamma_mr, nangle, degree)
end

"""
    square_bells_figures(result; refine=3)

Return a labelled response curve and current-magnitude maps. All maps share a
colour scale in particle-current model units. Coordinates are in mesh units.
"""
function square_bells_figures(result; refine = 3)
    response_figure = Figure(; size = (700, 460), fontsize = 18)
    response_axis = Axis(response_figure[1, 1]; xscale = log10,
        xlabel = "Momentum-conserving rate γmc (inverse model time)",
        ylabel = "Top particle flux / Δμ (model units)",
        title = "Square bells · ballistic–hydrodynamic crossover")
    scatterlines!(response_axis, result.rates, result.response;
                  color = :steelblue, markersize = 12, linewidth = 2)
    indices = sort!(collect(keys(result.selected)))
    fields = [PlotData(result.selected[i], particle_current; refine) for i in indices]
    current_limit = maximum(maximum(hypot.(field.values[:particle_current]...)) for field in fields)
    map_figure = Figure(; size = (360 * length(indices) + 100, 430), fontsize = 18)
    for (column, (index, data)) in enumerate(zip(indices, fields))
        axis = Axis(map_figure[1, column]; aspect = DataAspect(),
                    xlabel = "x (mesh units)", ylabel = "y (mesh units)",
                    title = "γmc = $(result.rates[index])")
        field_plot = plot!(axis, data, :particle_current; refine,
                           colorrange = (0.0, current_limit), colormap = :viridis)
        column == length(indices) && Colorbar(map_figure[1, column + 1], field_plot;
                                              label = "|particle current| (model units)")
    end
    return (; response = response_figure, maps = map_figure)
end

"""
    save_square_bells(result, figures, output_directory)

Save both figures, the response table, and a representative state and observable
file. Reload the stored current to exercise the backend-independent I/O path.
"""
function save_square_bells(result, figures, output_directory)
    mkpath(output_directory)
    save(joinpath(output_directory, "response.png"), figures.response)
    save(joinpath(output_directory, "current_maps.png"), figures.maps)
    open(joinpath(output_directory, "response.tsv"), "w") do io
        println(io, "gamma_mc\tparticle_flux_per_delta_mu\tsolve_seconds")
        for i in eachindex(result.rates)
            println(io, result.rates[i], '\t', result.response[i], '\t', result.seconds[i])
        end
    end
    solution = result.selected[sort!(collect(keys(result.selected)))[cld(length(result.selected), 2)]]
    save_solution(solution, joinpath(output_directory, "representative_state.h5"))
    field_path = joinpath(output_directory, "representative_current.h5")
    save_plotdata(PlotData(solution, particle_current), field_path)
    return read_plotdata(field_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_directory = isempty(ARGS) ? joinpath(@__DIR__, "output", "square_bells") : abspath(only(ARGS))
    result = square_bells_sweep()
    figures = square_bells_figures(result)
    save_square_bells(result, figures, output_directory)
    display(figures.response)
    display(figures.maps)
    println("Solve time including compilation ", sum(result.seconds), " s")
end
