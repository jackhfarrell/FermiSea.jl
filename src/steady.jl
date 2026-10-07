# This file builds and solves steady transport problems. A problem holds the linear
# equation, mesh, and boundary data. Before a solve, we check whether known inputs
# changed since assembly. The solver can also find the contact offsets needed to
# make floating contacts carry their requested currents

struct BoltzmannProblem{A,B,S,M,K,C,R}
    A::A
    b::B
    semi::S
    spatial_mass::M
    compatibility::K
    validity::UInt
    boundary_conditions::C
    source_terms::R
    degree::Int
end

"""
    BoltzmannProblem(model, mesh, boundary_conditions; source_terms=nothing, degree=2)

Build the steady equation `A * phi = b` on the given mesh. This represents
`v dot grad(phi) + C phi = d`. Trixi uses the evolution sign for `source_terms`,
so a `Callaway` collision contributes `-C phi` to `dphi/dt`. Boundary values
also contribute to `b`.

The flat state lists all momentum nodes at one spatial node before moving to the
next spatial node, as Trixi does.
"""
function BoltzmannProblem(model::BoltzmannEquation, mesh, boundary_conditions;
                          source_terms = nothing, degree::Integer = 2)
    degree >= 1 || throw(ArgumentError("degree must be at least 1"))
    mesh isa Trixi.UnstructuredMesh2D && degree < mesh.polydeg &&
        throw(
            ArgumentError(
                "degree must be at least the unstructured mesh geometry " *
                    "degree $(mesh.polydeg)"
            )
        )
    # Unstructured meshes use the packed path for the large momentum state
    volume_integral =
        mesh isa Trixi.UnstructuredMesh2D ? PackedWeakForm() :
        Trixi.VolumeIntegralWeakForm()
    solver = Trixi.DGSEM(; polydeg = Int(degree), surface_flux = flux_upwind,
                         volume_integral)
    prepared_sources = prepare_source(source_terms)
    prepared_boundaries = boundary_conditions isa NamedTuple ?
        map(prepare_boundary, boundary_conditions) :
        boundary_conditions isa AbstractDict ?
        (; (name => prepare_boundary(boundary)
            for (name, boundary) in boundary_conditions)...) :
        boundary_conditions
    semi = Trixi.SemidiscretizationHyperbolic(
        mesh, model, Trixi.initial_condition_constant, solver;
        boundary_conditions = prepared_boundaries, source_terms = prepared_sources)
    time_operator, time_drive = Trixi.linear_structure(semi)
    mass = spatial_mass(semi)
    # Negate Trixi's time equation to put the steady operator on the left
    return BoltzmannProblem(-time_operator, -time_drive, semi, mass,
                            problem_compatibility(model, semi, mass),
                            operator_signature(semi), prepared_boundaries,
                            source_terms, Int(degree))
end

"""
`prepare_preconditioner(problem, configuration)` builds reusable factors and work
arrays for one problem. `solve` prepares these automatically; call this explicitly
to reuse them across separate solves with `GMRES(; precond=prepared)`. Julia frees
the factors when the preparation is no longer referenced and garbage collection
runs. Do not share a preparation between concurrent solves. Build separate
problems and preparations for those solves.

We check known surface, geometry, and source arrays before reuse. We cannot inspect
data held inside a custom collision callback. If that data changes, rebuild the
problem and its preconditioner.
"""
prepare_preconditioner

"""
    GMRES(; restart=40, maxiters=1000, precond=AutomaticPreconditioner())

Choose settings for GMRES, the iterative linear solver. `maxiters` limits the
total iterations. `restart` sets how many it takes before starting a new cycle.
For `precond`, give a preconditioner setting, a prepared preconditioner, or
`nothing`. A prepared preconditioner belongs to one problem. `solve` checks
convergence using the weighted residual of the transport equation.
"""
struct GMRES{P}
    restart::Int
    maxiters::Int
    precond::P
    function GMRES(; restart::Integer = 40, maxiters::Integer = 1000,
                   precond = AutomaticPreconditioner())
        restart > 0 || throw(ArgumentError("restart must be positive"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        new{typeof(precond)}(Int(restart), Int(maxiters), precond)
    end
end

"""
    AutomaticPreconditioner()

Choose a preconditioner from the mesh and local source. On a conforming
unstructured mesh, supported Callaway sources use `MomentSweep`, which falls back
to `TransportSweep` in the nearly collisionless limit. Other sources use
`TransportSweep`. Tree meshes use no package preconditioner. This choice
does not change the transport equation.
"""
struct AutomaticPreconditioner end

"""
BoltzmannSolution

The result of a steady solve. `phi` is the flat, unweighted momentum state.
The result also keeps its `problem`, weighted relative `residual`, convergence
flag, iteration count, tolerance, and status. For floating contacts, it records
their offsets and outward charge currents. Use `state`, the observable
functions, or `SpatialSampler` to inspect the solution.
"""
struct BoltzmannSolution{T,P,O,C}
    phi::Vector{T}
    problem::P
    residual::Float64
    converged::Bool
    iterations::Int
    tolerance::Float64
    status::Symbol
    contact_offsets::O
    contact_currents::C
end

"""
    state(solution)

View `solution.phi` as a matrix. Momentum nodes run down the first axis. The
second axis lists spatial DG nodes in `(xi, eta, element)` order, with `xi`
changing fastest. This is a view of the stored state, so changing it changes
the solution. It does not have Trixi's four-axis shape.
"""
state(solution::BoltzmannSolution) =
    reshape(solution.phi, length(solution.problem.semi.equations.surface), :)

"""
    solve(problem, algorithm=GMRES(); tolerance=1e-9, initial_guess=nothing,
          throw_on_failure=true, reference=nothing, current_tolerance=1e-9)
    solve(channel; ncells=48, grading=1.5, tolerance=1e-9)

Solve a device problem and return a `BoltzmannSolution`. For a channel, return a
`ChannelProfile`. The device residual measures `A*phi-b` with the spatial mass
and momentum weights. We divide by the weighted size of `b`. If `b` is zero,
we use the starting residual instead, or one if that is also zero.

By default, a failed device solve throws an error. Set `throw_on_failure=false`
to inspect its `converged`, `status`, and `residual` fields. `initial_guess` can
be a flat state or a compatible earlier solution. If a rate or field changes,
build a new problem and preconditioner. An earlier solution can still be the
starting guess.

For floating contacts, `current_tolerance` sets the allowed error in outward
charge current. If every contact floats, `reference` names the contact whose
offset is fixed at zero. Channel solves also check their gauges and residual.
"""
function solve(problem::BoltzmannProblem, algorithm::GMRES = GMRES();
               tolerance::Real = 1.0e-9, initial_guess = nothing,
               throw_on_failure::Bool = true, verbose::Bool = false,
               reference::Union{Nothing,Symbol} = nothing,
               current_tolerance::Real = 1.0e-9)
    isfinite(current_tolerance) && current_tolerance >= 0 || throw(ArgumentError(
        "current_tolerance must be finite and nonnegative"))
    isempty(floating_names(problem)) && return _solve_linear(problem, algorithm;
        tolerance, initial_guess, throw_on_failure, verbose)
    return solve_floating(problem, algorithm; tolerance, initial_guess,
        throw_on_failure,
        verbose, reference, current_tolerance)
end

function content_signature(values, seed::UInt = zero(UInt))
    result = seed
    for value in values
        result = hash(value, result)
    end
    return result
end

source_signature(::Nothing, seed) = hash(:nothing, seed)
source_signature(source::Callaway, seed) =
    content_signature(source.momentum.weighted_modes,
        content_signature(source.momentum.modes,
            content_signature(source.population.weighted_modes,
                content_signature(source.population.modes,
                    hash((source.gamma_mr, source.gamma_mc), seed)))))
source_signature(source::PreparedLocalSource{<:MagneticField}, seed) =
    hash((source.operator.field, source.operator.charge, source.operator.frequency), seed)
source_signature(source::PreparedLocalSource{<:TomographicCollision}, seed) =
    content_signature(source.operator.rates, seed)
source_signature(source::PreparedLocalSource{<:CustomCollision}, seed) = begin
    result = hash((objectid(source.operator.action!), source.operator.rate_bound), seed)
    source.operator.diagonal === nothing ? result :
        content_signature(source.operator.diagonal, result)
end
source_signature(source::SourceTerms, seed) =
    foldl((result, term) -> source_signature(term, result), source.terms; init = seed)
source_signature(source, seed) = hash(objectid(source), seed)

function operator_signature(semi)
    surface = semi.equations.surface
    # Signatures catch edits to data used when the operator was assembled
    result =
        hash((surface.p_fermi, surface.v_fermi, surface.mu, surface.charge, surface.spin_degeneracy), UInt(0))
    for values in (surface.momenta, surface.velocities, surface.energies, surface.weights, surface.theta)
        result = content_signature(values, result)
    end
    elements = semi.cache.elements
    for name in (:node_coordinates, :contravariant_vectors, :inverse_jacobian,
                 :normal_directions)
        hasproperty(elements, name) || continue
        result = content_signature(getproperty(elements, name), result)
    end
    result = source_signature(semi.source_terms, result)
    return boundary_signature(semi.boundary_conditions, result)
end

boundary_signature(boundary::FixedReservoir, seed) =
    hash(boundary.delta_mu, seed)
boundary_signature(boundary::FloatingContact, seed) =
    hash(boundary.target_current, seed)
boundary_signature(::DiffuseWall, seed) = hash(:diffuse, seed)
boundary_signature(::PreparedDiffuseWall, seed) = hash(:diffuse, seed)
boundary_signature(boundary::MaxwellWall, seed) =
    hash(boundary.p_scatter, seed)
boundary_signature(boundary::PreparedMaxwellWall, seed) =
    boundary_signature(boundary.wall, seed)
boundary_signature(boundary::IncomingProfile, seed) =
    content_signature(boundary.profile.states,
        content_signature(boundary.profile.coordinates,
            hash((boundary.profile.width, boundary.profile.axis,
                  boundary.profile.transverse, boundary.origin), seed)))
boundary_signature(boundaries::NamedTuple, seed) =
    foldl((value, boundary) -> boundary_signature(boundary, value), values(boundaries);
          init = seed)
boundary_signature(boundaries::AbstractDict, seed) =
    foldl((value, name) -> boundary_signature(boundaries[name], hash(name, value)),
          sort!(collect(keys(boundaries))); init = seed)
boundary_signature(boundary, seed) = if hasproperty(boundary, :boundary_condition_types)
    foldl((value, condition) -> boundary_signature(condition, value),
          boundary.boundary_condition_types; init = seed)
else
    hash(typeof(boundary), seed)
end

function validate_problem(problem)
    operator_signature(problem.semi) == problem.validity || throw(ArgumentError(
        "problem surface, spatial geometry, or local source changed after preparation; " *
        "rebuild BoltzmannProblem and its preconditioner"))
    return nothing
end

function spatial_mass(semi::Trixi.SemidiscretizationHyperbolic{<:Trixi.TreeMesh})
    inverse_jacobian = semi.cache.elements.inverse_jacobian
    weights = semi.solver.basis.weights
    n = length(weights)
    mass = Vector{Float64}(undef, length(inverse_jacobian) * n^2)
    k = 0
    for element in eachindex(inverse_jacobian), j in 1:n, i in 1:n
        mass[k += 1] = weights[i] * weights[j] / inverse_jacobian[element]^2
    end
    return mass
end

function spatial_mass(
    semi::Trixi.SemidiscretizationHyperbolic{<:Trixi.UnstructuredMesh2D}
)
    inverse_jacobian = semi.cache.elements.inverse_jacobian
    weights = semi.solver.basis.weights
    n = length(weights)
    mass = Vector{Float64}(undef, length(inverse_jacobian))
    k = 0
    for element in axes(inverse_jacobian, 3), j in 1:n, i in 1:n
        # Unstructured inverse_jacobian is the inverse determinant, unlike the TreeMesh
        # scalar inverse length used above
        mass[k += 1] = weights[i] * weights[j] / inverse_jacobian[i, j, element]
    end
    return mass
end

# A warm start needs the same state layout even if rates or fields have changed
problem_compatibility(model, semi, mass) =
    (nvars = length(model.surface), ndofs = length(mass) * length(model.surface),
     momenta = content_signature(model.surface.momenta),
     velocities = content_signature(model.surface.velocities),
     energies = content_signature(model.surface.energies),
     weights = content_signature(model.surface.weights),
     mesh = typeof(semi.mesh),
     degree = Trixi.nnodes(semi.solver) - 1,
     coordinates = content_signature(semi.cache.elements.node_coordinates))

"""
    weighted_norm(values, problem)

Measure the size of a flat state using the momentum weights and the DG spatial
mass. Each momentum node and spatial node contributes once. `values` must have
one full momentum block for every spatial node.
"""
function weighted_norm(values, problem::BoltzmannProblem)
    surface_weights = problem.semi.equations.surface.weights
    mass = problem.spatial_mass
    nvars = length(surface_weights)
    length(values) == nvars * length(mass) || throw(DimensionMismatch(
        "state has $(length(values)) entries, expected $(nvars * length(mass))"))
    # Judge solver error with the same momentum weights used by the model
    accumulator = 0.0
    @inbounds for node in eachindex(mass)
        base = (node - 1) * nvars
        block = 0.0
        for i in 1:nvars
            block = muladd(surface_weights[i] * values[base + i], values[base + i], block)
        end
        accumulator = muladd(mass[node], block, accumulator)
    end
    return sqrt(max(accumulator, 0.0))
end

relative_scale(bnorm, initial_residual) = bnorm > 0 ? bnorm :
    (initial_residual > 0 ? initial_residual : 1.0)

function _solve_linear(problem::BoltzmannProblem, algorithm::GMRES = GMRES();
               tolerance::Real = 1.0e-9, initial_guess = nothing,
               throw_on_failure::Bool = true, verbose::Bool = false,
               rhs = problem.b)
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("tolerance must be finite and nonnegative"))
    validate_problem(problem)
    if initial_guess isa BoltzmannSolution
        initial_guess.problem.compatibility == problem.compatibility || throw(
            ArgumentError(
                "initial solution uses an incompatible Fermi surface or " *
                    "spatial discretization"
            )
        )
        initial_guess = initial_guess.phi
    end
    initial_guess === nothing || length(initial_guess) == length(problem.b) ||
        throw(DimensionMismatch("initial_guess has the wrong state size"))
    length(rhs) == length(problem.b) ||
        throw(DimensionMismatch("rhs has the wrong size"))
    x = initial_guess === nothing ? zeros(length(rhs)) : copy(initial_guess)
    residual = similar(rhs)
    mul!(residual, problem.A, x)
    @. residual = rhs - residual
    residual_norm = weighted_norm(residual, problem)
    # A zero right-hand side still needs a scale for relative convergence
    scale = relative_scale(weighted_norm(rhs, problem), residual_norm)
    metric = residual_norm / scale
    best_x = copy(x)
    best_metric = metric
    total_iterations = 0
    converged = metric <= tolerance
    workspace = nothing
    preconditioner = prepare_preconditioner(problem, algorithm.precond)

    while !converged && total_iterations < algorithm.maxiters
        round_limit = min(algorithm.restart, algorithm.maxiters - total_iterations)
        workspace === nothing &&
            (workspace = Krylov.GmresWorkspace(problem.A, residual;
                                                memory = algorithm.restart))
        Krylov.gmres!(workspace, problem.A, residual;
                      N = something(preconditioner, I),
                      ldiv = preconditioner !== nothing, restart = false,
                      itmax = round_limit, rtol = 0.0, atol = 0.0, verbose = 0)
        iterations = Krylov.statistics(workspace).niter
        iterations == 0 && break
        total_iterations += iterations
        x .+= Krylov.solution(workspace)

        # Krylov's Euclidean estimate is not our physical residual
        mul!(residual, problem.A, x)
        @. residual = rhs - residual
        metric = weighted_norm(residual, problem) / scale
        # A later GMRES restart can do worse, so keep the best physical iterate
        if metric < best_metric
            copyto!(best_x, x)
            best_metric = metric
        end
        converged = metric <= tolerance
        verbose && @info "GMRES restart" iterations = total_iterations residual = metric
    end

    status = converged ? :converged : :maxiters
    solution = BoltzmannSolution(best_x, problem, Float64(best_metric), converged,
                                 total_iterations, Float64(tolerance), status, (;), (;))
    if throw_on_failure && !converged
        error("steady solve failed after $total_iterations iterations; true weighted " *
              "residual $(solution.residual) exceeds tolerance $(solution.tolerance)")
    end
    return solution
end

function replace_boundaries(boundaries::NamedTuple, replacements::Dict{Symbol,Any})
    names = keys(boundaries)
    values_new =
        ntuple(i -> get(replacements, names[i], values(boundaries)[i]), length(names))
    return NamedTuple{names}(values_new)
end

function replace_boundaries(boundaries::AbstractDict, replacements::Dict{Symbol,Any})
    return Dict(name => get(replacements, name, boundary)
                for (name, boundary) in boundaries)
end

get_boundary_condition(boundaries::NamedTuple, name) = getproperty(boundaries, name)
get_boundary_condition(boundaries::AbstractDict, name) = boundaries[name]

function floating_names(problem)
    problem.boundary_conditions isa Union{NamedTuple,AbstractDict} || return Symbol[]
    return [
        name for name in keys(problem.boundary_conditions)
        if get_boundary_condition(problem.boundary_conditions, name) isa FloatingContact
    ]
end

function solve_floating(problem, algorithm; tolerance, initial_guess, throw_on_failure,
                        verbose, reference, current_tolerance)
    names = floating_names(problem)
    boundaries = problem.boundary_conditions
    fixed_gauge = any(boundary -> boundary isa Union{FixedReservoir,IncomingProfile},
                      values(boundaries))
    # With no fixed contact, one offset must set the otherwise arbitrary zero
    if !fixed_gauge
        reference === nothing && throw(
            ArgumentError(
                "all contacts are floating; give `reference` to fix the " *
                    "electrochemical gauge"
            )
        )
        reference in names ||
            throw(ArgumentError("reference must name a floating contact"))
    elseif reference !== nothing
        throw(ArgumentError("reference is only used when every contact is floating"))
    end
    unknowns = reference === nothing ? names : filter(!=(reference), names)
    base_replacements = Dict{Symbol,Any}()
    # Zero floating offsets isolate the imposed currents from the contact responses
    for name in names
        base_replacements[name] = FixedReservoir(0)
    end
    base_boundaries = replace_boundaries(boundaries, base_replacements)
    base_problem =
        BoltzmannProblem(problem.semi.equations, problem.semi.mesh, base_boundaries;
            source_terms = problem.source_terms,
            degree = problem.degree)
    prepared = prepare_preconditioner(problem, algorithm.precond)
    response_algorithm =
        GMRES(; restart = algorithm.restart, maxiters = algorithm.maxiters,
            precond = prepared)
    base = _solve_linear(problem, response_algorithm; tolerance, initial_guess,
                         throw_on_failure, verbose, rhs = base_problem.b)
    base = BoltzmannSolution(base.phi, base_problem, base.residual, base.converged,
                             base.iterations, base.tolerance, base.status, (;), (;))
    responses = BoltzmannSolution[]
    unit_problems = BoltzmannProblem[]
    rhs_responses = Vector{Vector{Float64}}()
    # Linearity lets one unit-offset solve describe each unknown contact offset
    for name in unknowns
        replacements = copy(base_replacements)
        replacements[name] = FixedReservoir(1)
        unit_boundaries = replace_boundaries(boundaries, replacements)
        unit_problem = BoltzmannProblem(problem.semi.equations, problem.semi.mesh,
                                        unit_boundaries;
                                        source_terms = problem.source_terms,
                                        degree = problem.degree)
        rhs = unit_problem.b - base_problem.b
        push!(unit_problems, unit_problem)
        push!(rhs_responses, rhs)
        push!(responses, _solve_linear(problem, response_algorithm; tolerance,
                                       throw_on_failure, verbose, rhs))
    end
    base_currents = [contact_current(base, name) for name in unknowns]
    response_matrix = zeros(length(unknowns), length(unknowns))
    for column in eachindex(unknowns)
        total_phi = base.phi + responses[column].phi
        total = BoltzmannSolution(total_phi, unit_problems[column], 0.0, true, 0, 0.0,
                                  :response, (;), (;))
        for (row, name) in enumerate(unknowns)
            response_matrix[row, column] = contact_current(total, name) -
                                           contact_current(base, name)
        end
    end
    targets =
        [get_boundary_condition(boundaries, name).target_current for name in unknowns]
    # A singular response cannot determine all requested contact offsets
    isempty(unknowns) || cond(response_matrix) < inv(sqrt(eps(Float64))) ||
        throw(
            ArgumentError(
                "floating-contact response is singular; the gauge or contact " *
                    "constraints are underdetermined"
            )
        )
    offsets_vector =
        isempty(unknowns) ? Float64[] :
        response_matrix \ (targets - base_currents)
    resolved = copy(base_replacements)
    for (name, offset) in zip(unknowns, offsets_vector)
        resolved[name] = FixedReservoir(offset)
    end
    reference !== nothing && (resolved[reference] = FixedReservoir(0))
    resolved_boundaries = replace_boundaries(boundaries, resolved)
    resolved_problem = BoltzmannProblem(problem.semi.equations, problem.semi.mesh,
                                        resolved_boundaries;
                                        source_terms = problem.source_terms,
                                        degree = problem.degree)
    phi = copy(base.phi)
    for (offset, response) in zip(offsets_vector, responses)
        @. phi += offset * response.phi
    end
    # Check the combined state against the resolved problem, not just its parts
    defect = resolved_problem.A * phi - resolved_problem.b
    scale = relative_scale(weighted_norm(resolved_problem.b, resolved_problem),
                           weighted_norm(defect, resolved_problem))
    residual = weighted_norm(defect, resolved_problem) / scale
    offset_names = Tuple(names)
    offset_values = ntuple(i -> begin
        name = offset_names[i]
        name === reference ? 0.0 : offsets_vector[findfirst(==(name), unknowns)]
    end, length(offset_names))
    offsets = NamedTuple{offset_names}(offset_values)
    provisional = BoltzmannSolution(phi, resolved_problem, residual,
        base.converged && all(solution -> solution.converged, responses) &&
            residual <= tolerance,
        base.iterations + sum(solution -> solution.iterations, responses; init = 0),
        Float64(tolerance),
        residual <= tolerance ? :converged : :maxiters, offsets, (;))
    current_values = ntuple(i -> contact_current(provisional, offset_names[i]),
                            length(offset_names))
    currents = NamedTuple{offset_names}(current_values)
    # A small kinetic residual alone does not guarantee the contact currents match
    current_error = maximum((abs(current_values[i] -
        get_boundary_condition(boundaries, offset_names[i]).target_current)
        for i in eachindex(offset_names)); init = 0.0)
    current_ok = current_error <= current_tolerance
    converged = provisional.converged && current_ok
    status = !provisional.converged ? provisional.status :
             current_ok ? :converged : :contact_constraint
    result = BoltzmannSolution(phi, resolved_problem, residual, converged,
        provisional.iterations, Float64(tolerance), provisional.status,
        offsets, currents)
    if throw_on_failure && !converged
        error(
            "floating-contact solve failed: kinetic residual $residual, " *
                "maximum current constraint error $current_error"
        )
    end
    return BoltzmannSolution(result.phi, result.problem, result.residual, converged,
        result.iterations, result.tolerance, status, offsets, currents)
end

prepare_preconditioner(problem::BoltzmannProblem, ::Nothing) = nothing
prepare_preconditioner(problem::BoltzmannProblem, config::TransportSweep) =
    build_transport_sweep(problem, config)
# Reusing factors from another operator would apply the wrong approximate inverse
prepare_preconditioner(problem::BoltzmannProblem, prepared) = begin
    if hasproperty(prepared, :semi)
        prepared.semi === problem.semi || throw(ArgumentError(
            "prepared preconditioner is stale or belongs to an incompatible problem"))
    end
    if hasproperty(prepared, :validity)
        prepared.validity == operator_signature(problem.semi) || throw(ArgumentError(
            "prepared preconditioner is stale because operator-defining data changed"))
    end
    prepared
end

"""
    release_preconditioner!(prepared)

Free any external sparse-factor resources held by a prepared preconditioner.
This is optional: Julia also frees them when they are no longer referenced and
garbage collection runs. Call this to free them immediately, and do not reuse the
preparation after release. Preparations without such resources need no cleanup.
Returns `nothing`.
"""
release_preconditioner!(::Nothing) = nothing
