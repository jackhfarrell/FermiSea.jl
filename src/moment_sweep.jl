# A transport sweep can leave slow Callaway modes behind. We correct density,
# momentum, and stress together, then sweep again. Stress supplies the viscous
# relaxation missing from a density and momentum correction

"""
    MomentSweep(; passes=1, factor_storage=Float64)

Use this preconditioner for supported Callaway sources on a conforming
unstructured mesh. It runs a transport sweep, solves for the remaining error in
the density, momentum, and stress modes, then sweeps again. `passes` sets the
number of passes in each transport sweep. The sweep includes the local source
diagonal but skips its separate collision corrections. `factor_storage` chooses
`Float32` or `Float64` for saved factors.

We restrict the kinetic operator to density, two momentum modes, and two stress
modes, dropping any dependent modes on coarse angular grids. Retaining stress
supplies viscous momentum relaxation in the hydrodynamic limit.
The reduced solve uses the same streaming, boundaries, and local sources as
the full problem, including when `gamma_mr == 0`. As a heuristic for the nearly
collisionless limit, we use `TransportSweep` alone when
`v_fermi / (gamma_mr + gamma_mc)` is at least the device's bounding-box diameter.
This includes zero collision rates. The fallback includes collision corrections
and avoids nearly undamped modes in the reduced system.
"""
Base.@kwdef struct MomentSweep
    passes::Int = 1
    factor_storage::DataType = Float64
end

struct MomentSweepPreconditioner{S,F,T,W}
    sweep::S
    fine_operator::F
    factorization::T
    modes::Matrix{Float64}
    weighted_modes::Matrix{Float64}
    moment_count::Int
    momentum_node_count::Int
    spatial_node_count::Int
    semi
    validity::UInt
    workspace::W
end

function LinearAlgebra.ldiv!(y, preconditioner::MomentSweepPreconditioner, x)
    work = preconditioner.workspace
    sweep = preconditioner.sweep
    ldiv!(y, sweep, x)
    # The first sweep leaves an error that the small moment solve can correct
    mul!(work.residual, preconditioner.fine_operator, y)
    @. work.residual = x - work.residual
    restrict_moments!(work.moment_rhs, work.residual, preconditioner)
    ldiv!(work.moment_solution, preconditioner.factorization, work.moment_rhs)
    prolong_add_moments!(y, work.moment_solution, preconditioner)
    # A final sweep picks up the error outside the selected moment modes
    mul!(work.residual, preconditioner.fine_operator, y)
    @. work.residual = x - work.residual
    ldiv!(work.correction, sweep, work.residual)
    y .+= work.correction
    return y
end

function restrict_moments!(beta, phi, preconditioner)
    # Weighted modes measure the part of each spatial state kept in the small solve
    weighted = preconditioner.weighted_modes
    momentum_node_count = preconditioner.momentum_node_count
    moment_count = preconditioner.moment_count
    spatial_node_count = preconditioner.spatial_node_count
    @inbounds for node in 1:spatial_node_count
        base_phi = (node - 1) * momentum_node_count
        base_beta = (node - 1) * moment_count
        for alpha in 1:moment_count
            value = 0.0
            for p in 1:momentum_node_count
                value = muladd(weighted[p, alpha], phi[base_phi + p], value)
            end
            beta[base_beta + alpha] = value
        end
    end
    return beta
end

function prolong_add_moments!(phi, beta, preconditioner)
    modes = preconditioner.modes
    momentum_node_count = preconditioner.momentum_node_count
    moment_count = preconditioner.moment_count
    spatial_node_count = preconditioner.spatial_node_count
    @inbounds for node in 1:spatial_node_count
        base_phi = (node - 1) * momentum_node_count
        base_beta = (node - 1) * moment_count
        for p in 1:momentum_node_count
            increment = 0.0
            for alpha in 1:moment_count
                increment = muladd(modes[p, alpha], beta[base_beta + alpha], increment)
            end
            phi[base_phi + p] += increment
        end
    end
    return phi
end

function moment_sweep_sources(source_terms)
    # Only supported source combinations can use this moment correction
    collision = magnetic = drive = nothing
    function visit(source)
        source === nothing && return
        if source isa Callaway
            collision === nothing || throw(ArgumentError(
                "MomentSweep accepts exactly one Callaway term"))
            collision = source
        elseif source isa PreparedLocalSource{<:MagneticField}
            magnetic === nothing || throw(ArgumentError(
                "MomentSweep accepts at most one magnetic field term"))
            magnetic = source.operator
        elseif source isa ElectricDrive
            drive === nothing || throw(ArgumentError(
                "MomentSweep accepts at most one electric drive term"))
            drive = source
        elseif source isa SourceTerms
            foreach(visit, source.terms)
        elseif source isa PreparedLocalSource{<:CustomCollision}
            throw(
                ArgumentError(
                    "MomentSweep cannot rebuild an action-only custom collision; " *
                        "use TransportSweep or an external prepared preconditioner"
                )
            )
        elseif source isa PreparedLocalSource{<:TomographicCollision}
            throw(
                ArgumentError(
                    "MomentSweep does not rebuild tomographic collisions; " *
                        "use TransportSweep or an external prepared preconditioner"
                )
            )
        else
            throw(ArgumentError("MomentSweep cannot rebuild source $(typeof(source))"))
        end
    end
    visit(source_terms)
    collision === nothing &&
        throw(ArgumentError("MomentSweep requires a Callaway term"))
    return collision, magnetic, drive
end

function moment_callaway(source_terms)
    moment_sweep_sources(source_terms)[1]
end

function build_moment_sweep(problem, config)
    semi = problem.semi
    semi.mesh isa Trixi.UnstructuredMesh2D || throw(ArgumentError(
        "MomentSweep requires a conforming UnstructuredMesh2D"))
    config.passes >= 1 || throw(ArgumentError("moment sweep passes must be positive"))
    config.factor_storage in (Float32, Float64) || throw(ArgumentError(
        "moment factor storage must be Float32 or Float64"))
    collision, _, _ = moment_sweep_sources(semi.source_terms)
    surface = semi.equations.surface
    # Nearly collisionless moment systems can have undamped modes
    # We use the higher-harmonic mean free path relative to the device as a heuristic
    coordinates = reshape(semi.cache.elements.node_coordinates, 2, :)
    extent = maximum(coordinates; dims = 2) - minimum(coordinates; dims = 2)
    speed = maximum(hypot(v[1], v[2]) for v in eachrow(surface.velocities))
    if (collision.gamma_mr + collision.gamma_mc) * norm(extent) <= speed
        return build_transport_sweep(problem, TransportSweep(;
            passes = config.passes, factor_storage = config.factor_storage))
    end
    # Stress couples momentum to its viscous relaxation even when gamma_mr is zero
    px, py = surface.momenta[:, 1], surface.momenta[:, 2]
    projector = weighted_projector(hcat(collision.momentum.modes,
        px .^ 2 .- py .^ 2, px .* py), surface.weights)
    modes = projector.modes
    weighted_modes = projector.weighted_modes
    moment_count = size(modes, 2)
    momentum_node_count = length(semi.equations.surface)
    # The reduced solve handles selected modes, so the sweep skips source corrections
    sweep = build_transport_sweep(problem, TransportSweep(;
        passes = config.passes, collision_passes = 0,
        factor_storage = config.factor_storage))
    spatial_node_count = sweep.element_node_count^2 * sweep.nelements
    operator = moment_restricted_operator(problem, modes, weighted_modes,
                                          momentum_node_count, moment_count,
                                          spatial_node_count)
    factorization = moment_factorization(operator, config.factor_storage)
    moment_dofs = moment_count * spatial_node_count
    workspace = (; residual = zeros(length(problem.b)), correction = similar(problem.b),
                   moment_rhs = zeros(moment_dofs), moment_solution = zeros(moment_dofs))
    return MomentSweepPreconditioner(sweep, problem.A, factorization, modes, weighted_modes,
                                     moment_count, momentum_node_count, spatial_node_count,
                                     semi, problem.validity, workspace)
end

# Columns in disjoint element neighborhoods can share one fine operator call
# The DG operator couples an element only to itself and its face neighbors
function moment_element_groups(semi)
    nelements = Trixi.nelements(semi.solver, semi.cache)
    neighborhoods = [[element] for element in 1:nelements]
    for face in axes(semi.cache.interfaces.element_ids, 2)
        left, right = semi.cache.interfaces.element_ids[:, face]
        push!(neighborhoods[left], right)
        push!(neighborhoods[right], left)
    end
    foreach(unique!, neighborhoods)
    colors = zeros(Int, nelements)
    for element in 1:nelements
        used = Set(colors[other] for neighbor in neighborhoods[element]
                   for other in neighborhoods[neighbor])
        color = 1
        while color in used
            color += 1
        end
        colors[element] = color
    end
    groups = [findall(==(color), colors) for color in 1:maximum(colors)]
    return groups, neighborhoods
end

# Use the fine operator so the reduced system sees the same mesh and boundaries
function moment_restricted_operator(problem, modes, weighted_modes, momentum_node_count,
                                    moment_count, spatial_node_count)
    moment_dofs = moment_count * spatial_node_count
    rows, columns, values = Int[], Int[], Float64[]
    phi = zeros(length(problem.b))
    action = similar(phi)
    moment_rhs = zeros(moment_dofs)
    scratch =
        (; modes, weighted_modes, momentum_node_count, moment_count, spatial_node_count)
    groups, neighborhoods = moment_element_groups(problem.semi)
    nodes_per_element = Trixi.nnodes(problem.semi.solver)^2
    for group in groups, node in 1:nodes_per_element, alpha in 1:moment_count
        fill!(phi, 0.0)
        for element in group
            spatial_node = (element - 1) * nodes_per_element + node
            offset = (spatial_node - 1) * momentum_node_count
            phi[(offset + 1):(offset + momentum_node_count)] .= view(modes, :, alpha)
        end
        mul!(action, problem.A, phi)
        restrict_moments!(moment_rhs, action, scratch)
        for element in group
            column = ((element - 1) * nodes_per_element + node - 1) * moment_count + alpha
            for neighbor in neighborhoods[element]
                first_row = (neighbor - 1) * nodes_per_element * moment_count + 1
                last_row = neighbor * nodes_per_element * moment_count
                for row in first_row:last_row
                    value = moment_rhs[row]
                    iszero(value) && continue
                    push!(rows, row)
                    push!(columns, column)
                    push!(values, value)
                end
            end
        end
    end
    return sparse(rows, columns, values, moment_dofs, moment_dofs)
end

function moment_sweep_available(source_terms)
    try
        moment_sweep_sources(source_terms)
        return true
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
end

release_preconditioner!(preconditioner::MomentSweepPreconditioner) =
    release_factorization!(preconditioner.factorization)

function prepare_preconditioner(problem::BoltzmannProblem, ::AutomaticPreconditioner)
    problem.semi.mesh isa Trixi.UnstructuredMesh2D || return nothing
    # Other source combinations use the transport sweep without a moment solve
    return moment_sweep_available(problem.semi.source_terms) ?
           build_moment_sweep(problem, MomentSweep()) :
           build_transport_sweep(problem, TransportSweep())
end

prepare_preconditioner(problem::BoltzmannProblem, config::MomentSweep) =
    build_moment_sweep(problem, config)
