# We sweep each momentum direction through the mesh to approximate streaming
# Local source corrections account for coupling between momentum directions

"""
    TransportSweep(; passes=1, collision_passes=1, factor_storage=Float32)

Set up a sweep through a conforming unstructured mesh. `passes` controls how many
times we walk through the elements for each momentum direction. `collision_passes`
adds corrections for local sources that mix momentum nodes. The source diagonal
is included even when `collision_passes=0`.

`factor_storage` chooses `Float32` or `Float64` for the saved element inverses.
The transport equation and convergence check still use `Float64`.
"""
Base.@kwdef struct TransportSweep
    passes::Int = 1
    collision_passes::Int = 1
    factor_storage::DataType = Float32
end

# Keep the element solves and scratch so GMRES can reuse them at each iteration

struct TransportSweepPreconditioner{S,I,W}
    semi::S
    validity::UInt
    element_node_count::Int
    nelements::Int
    momentum_node_count::Int
    vx::Vector{Float64}
    vy::Vector{Float64}
    inverses::Vector{I}
    order::Matrix{Int}
    cell_interfaces::Vector{Vector{Tuple{Int,Bool}}}
    passes::Int
    collision_passes::Int
    diagonal::Vector{Float64}
    workspace::W
end

function LinearAlgebra.ldiv!(y, preconditioner::TransportSweepPreconditioner, x)
    rhs = reshape(x, preconditioner.momentum_node_count, :)
    result = reshape(y, preconditioner.momentum_node_count, :)
    work = preconditioner.workspace
    # Spatial nodes come first here so each momentum direction can be swept alone
    permutedims!(work.rhs, rhs, (2, 1))
    sweep_directions!(work.solution, work.rhs, preconditioner)
    source_corrections!(result, preconditioner)
    permutedims!(result, work.solution, (2, 1))
    return y
end

function source_corrections!(result, preconditioner)
    preconditioner.collision_passes == 0 && return result
    work = preconditioner.workspace
    term, correction = work.term, work.correction
    copyto!(term, work.solution)
    for _ in 1:preconditioner.collision_passes
        permutedims!(result, term, (2, 1))
        coordinates = reshape(preconditioner.semi.cache.elements.node_coordinates, 2, :)
        apply_positive_source!(work.source, preconditioner.semi.source_terms, result,
                               work.source_workspace, coordinates, 0.0,
                               preconditioner.semi.equations)
        # Remove any affine drive before correcting the homogeneous source coupling
        work.source .-= work.zero_source
        permutedims!(work.rhs, work.source, (2, 1))
        # The local solve already includes the diagonal, so correct what it missed
        @inbounds for direction in 1:preconditioner.momentum_node_count,
            node in axes(work.term, 1)

            work.rhs[node, direction] =
                preconditioner.diagonal[direction] * term[node, direction] -
                work.rhs[node, direction]
        end
        sweep_directions!(correction, work.rhs, preconditioner)
        work.solution .+= correction
        term, correction = correction, term
    end
    return result
end

function sweep_directions!(solution, rhs, preconditioner)
    fill!(solution, 0.0)
    buffers = preconditioner.workspace.local_rhs
    Threads.@threads :dynamic for chunk in eachindex(buffers)
        first_direction =
            fld((chunk - 1) * preconditioner.momentum_node_count, length(buffers)) + 1
        last_direction =
            fld(chunk * preconditioner.momentum_node_count, length(buffers))
        sweep_directions!(solution, rhs, preconditioner,
                          first_direction:last_direction, buffers[chunk])
    end
    return solution
end

function sweep_directions!(solution, rhs, preconditioner, directions, local_rhs)
    q = preconditioner.element_node_count^2
    # Upstream element values supply the incoming faces of downstream elements
    for direction in directions, _ in 1:preconditioner.passes
        @inbounds for k in axes(preconditioner.order, 1)
            element = preconditioner.order[k, direction]
            offset = (element - 1) * q
            for node in 1:q
                local_rhs[node] = rhs[offset + node, direction]
            end
            subtract_inflow!(local_rhs, solution, preconditioner, element, direction)
            inverse = preconditioner.inverses[direction]
            for i in 1:q
                value = 0.0
                for j in 1:q
                    value = muladd(inverse[i, j, element], local_rhs[j], value)
                end
                solution[offset + i, direction] = value
            end
        end
    end
    return solution
end

function subtract_inflow!(local_rhs, solution, preconditioner, element, direction)
    semi = preconditioner.semi
    interfaces = semi.cache.interfaces
    normals = semi.cache.elements.normal_directions
    inverse_jacobian = semi.cache.elements.inverse_jacobian
    face_factor = semi.solver.basis.inverse_weights[1]
    n = preconditioner.element_node_count
    vx, vy = preconditioner.vx[direction], preconditioner.vy[direction]
    speed_scale = hypot(vx, vy)
    # Nearly grazing faces have no reliable upstream side for this direction
    @inbounds for (face, primary) in preconditioner.cell_interfaces[element]
        primary_element = interfaces.element_ids[1, face]
        secondary_element = interfaces.element_ids[2, face]
        primary_side = interfaces.element_side_ids[1, face]
        secondary_side = interfaces.element_side_ids[2, face]
        secondary_node = interfaces.start_index[face]
        increment = interfaces.index_increment[face]
        for primary_node in 1:n
            nx = normals[1, primary_node, primary_side, primary_element]
            ny = normals[2, primary_node, primary_side, primary_element]
            speed = vx * nx + vy * ny
            tolerance = 1.0e-12 * speed_scale * hypot(nx, ny)
            if primary && speed < -tolerance
                i, j = face_node(primary_side, primary_node, n)
                iu, ju = face_node(secondary_side, secondary_node, n)
                coefficient =
                    inverse_jacobian[i, j, primary_element] * face_factor * speed
                local_rhs[spatial_node(i, j, n)] -=
                    coefficient *
                        solution[
                        global_spatial_node(iu, ju, secondary_element, n),
                        direction
                    ]
            elseif !primary && speed > tolerance
                i, j = face_node(secondary_side, secondary_node, n)
                iu, ju = face_node(primary_side, primary_node, n)
                coefficient =
                    -inverse_jacobian[i, j, secondary_element] * face_factor * speed
                local_rhs[spatial_node(i, j, n)] -=
                    coefficient *
                        solution[global_spatial_node(iu, ju, primary_element, n),
                                 direction]
            end
            secondary_node += increment
        end
    end
    return local_rhs
end

@inline spatial_node(i, j, n) = (j - 1) * n + i
@inline global_spatial_node(i, j, element, n) =
    (element - 1) * n^2 + spatial_node(i, j, n)
@inline function face_node(side, node, n)
    side == 1 && return (node, 1)
    side == 2 && return (n, node)
    side == 3 && return (node, n)
    return (1, node)
end

source_diagonal(::Nothing, model) = zeros(length(model.surface))
source_diagonal(::ElectricDrive, model) = zeros(length(model.surface))
source_diagonal(source::PreparedLocalSource{<:MagneticField}, model) =
    zeros(length(model.surface))
# Every node has the same diagonal in this angular Fourier collision operator
source_diagonal(source::PreparedLocalSource{<:TomographicCollision}, model) = begin
    rates = source.operator.rates
    full_sum = rates[1] + 2sum(view(rates, 2:length(rates)))
    iseven(source.operator.n) && (full_sum -= rates[end])
    fill(full_sum / source.operator.n, source.operator.n)
end
source_diagonal(source::PreparedLocalSource{<:CustomCollision}, model) =
    source.operator.diagonal === nothing ? zeros(length(model.surface)) :
    copy(source.operator.diagonal)
source_diagonal(source::Callaway, model) =
    (source.gamma_mr + source.gamma_mc) .-
        source.gamma_mr .*
        vec(sum(source.population.modes .* source.population.weighted_modes;
        dims = 2)) .-
        source.gamma_mc .* vec(sum(
            source.momentum.modes .* source.momentum.weighted_modes;
            dims = 2))
source_diagonal(source::SourceTerms, model) =
    reduce(+, (source_diagonal(term, model) for term in source.terms);
           init = zeros(length(model.surface)))
source_diagonal(source, model) = zeros(length(model.surface))

source_rate_bound(::Nothing) = 0.0
source_rate_bound(::ElectricDrive) = 0.0
# In the response-weighted metric the nested orthogonal projectors bound decay
# by gamma_mr + gamma_mc even when the Euclidean operator norm is larger
source_rate_bound(source::Callaway) = source.gamma_mr + source.gamma_mc
source_rate_bound(source::PreparedLocalSource{<:MagneticField}) =
    abs(source.operator.frequency) * source.operator.n / 2
source_rate_bound(source::PreparedLocalSource{<:TomographicCollision}) =
    maximum(source.operator.rates; init = 0.0)
source_rate_bound(source::PreparedLocalSource{<:CustomCollision}) =
    source.operator.rate_bound !== nothing ? source.operator.rate_bound :
    (source.operator.diagonal === nothing ? 0.0 :
     maximum(abs, source.operator.diagonal; init = 0.0))
source_rate_bound(source::SourceTerms) = sum(source_rate_bound, source.terms)
source_rate_bound(source) = 0.0

function build_transport_sweep(problem, config)
    config.passes >= 1 ||
        throw(ArgumentError("transport sweep passes must be positive"))
    config.collision_passes >= 0 || throw(ArgumentError(
        "transport sweep collision passes must be nonnegative"))
    config.factor_storage in (Float32, Float64) || throw(ArgumentError(
        "transport factor storage must be Float32 or Float64"))
    semi = problem.semi
    semi.mesh isa Trixi.UnstructuredMesh2D || throw(ArgumentError(
        "TransportSweep requires a conforming UnstructuredMesh2D"))
    n = Trixi.nnodes(semi.solver)
    nelements = Trixi.nelements(semi.solver, semi.cache)
    model = semi.equations
    momentum_node_count = length(model.surface)
    vx = collect(model.surface.velocities[:, 1])
    vy = collect(model.surface.velocities[:, 2])
    diagonal = source_diagonal(semi.source_terms, model)
    # Reject a truly negative diagonal but ignore roundoff at the rate scale
    tolerance = 100eps(Float64) * max(source_rate_bound(semi.source_terms), 1.0)
    minimum(diagonal; init = 0.0) >= -tolerance || throw(ArgumentError(
        "the local source has a negative directional diagonal"))
    diagonal .= max.(diagonal, 0.0)
    inverses = directional_element_inverses(semi, vx, vy, diagonal,
                                             config.factor_storage)
    order, interfaces = transport_sweep_order(semi, vx, vy)
    spatial_node_count = n^2 * nelements
    corrections =
        iszero(source_rate_bound(semi.source_terms)) ? 0 : config.collision_passes
    zero_source = zeros(momentum_node_count, spatial_node_count)
    zero_state = zeros(momentum_node_count, spatial_node_count)
    zero_workspace = packed_source_workspace(
        semi.source_terms,
        spatial_node_count,
        momentum_node_count
    )
    # Save the zero-state source so later corrections exclude an affine drive
    coordinates = reshape(semi.cache.elements.node_coordinates, 2, :)
    apply_positive_source!(zero_source, semi.source_terms, zero_state, zero_workspace,
        coordinates, 0.0, semi.equations)
    workspace = (; rhs = zeros(spatial_node_count, momentum_node_count),
        solution = zeros(spatial_node_count, momentum_node_count),
        term = zeros(spatial_node_count, momentum_node_count),
        correction = zeros(spatial_node_count, momentum_node_count),
        source = zeros(momentum_node_count, spatial_node_count),
        zero_source,
        source_workspace = packed_source_workspace(semi.source_terms,
            spatial_node_count,
            momentum_node_count),
        local_rhs = [
            Vector{Float64}(undef, n^2)
            for _ in 1:min(momentum_node_count, Threads.nthreads(:default))
        ])
    return TransportSweepPreconditioner(semi, problem.validity,
        n, nelements,
        momentum_node_count, vx, vy, inverses, order, interfaces, config.passes,
        corrections, diagonal, workspace)
end

function directional_element_inverses(semi, vx, vy, diagonal, storage)
    n = Trixi.nnodes(semi.solver)
    q = n^2
    nelements = Trixi.nelements(semi.solver, semi.cache)
    inverses = [Array{storage,3}(undef, q, q, nelements) for _ in eachindex(vx)]
    Threads.@threads :dynamic for direction in eachindex(vx)
        matrix = Matrix{Float64}(undef, q, q)
        for element in 1:nelements
            directional_element_matrix!(matrix, semi, element, vx[direction],
                                        vy[direction], diagonal[direction], n)
            # Build in Float64, then store in the requested precision to save memory
            inverses[direction][:, :, element] .= inv(matrix)
        end
    end
    return inverses
end

function directional_element_matrix!(matrix, semi, element, vx, vy, diagonal, n)
    fill!(matrix, 0.0)
    contravariant = semi.cache.elements.contravariant_vectors
    inverse_jacobian = semi.cache.elements.inverse_jacobian
    normals = semi.cache.elements.normal_directions
    derivative_hat = semi.solver.basis.derivative_hat
    face_factor = semi.solver.basis.inverse_weights[1]
    @inbounds for j in 1:n, i in 1:n
        a1 = contravariant[1, 1, i, j, element] * vx +
             contravariant[2, 1, i, j, element] * vy
        a2 = contravariant[1, 2, i, j, element] * vx +
             contravariant[2, 2, i, j, element] * vy
        column = spatial_node(i, j, n)
        for ii in 1:n
            matrix[spatial_node(ii, j, n), column] +=
                inverse_jacobian[ii, j, element] * derivative_hat[ii, i] * a1
        end
        for jj in 1:n
            matrix[spatial_node(i, jj, n), column] +=
                inverse_jacobian[i, jj, element] * derivative_hat[jj, j] * a2
        end
        matrix[column, column] += diagonal
    end
    @inbounds for side in 1:4, node in 1:n
        i, j = face_node(side, node, n)
        speed = vx * normals[1, node, side, element] +
                vy * normals[2, node, side, element]
        index = spatial_node(i, j, n)
        # Outgoing face flux stays local. Incoming flux comes from the sweep
        matrix[index, index] += inverse_jacobian[i, j, element] * face_factor *
                                max(speed, 0.0)
    end
    return matrix
end

function transport_sweep_order(semi, vx, vy)
    cache, dg = semi.cache, semi.solver
    interfaces = cache.interfaces
    normals = cache.elements.normal_directions
    n = Trixi.nnodes(dg)
    nelements = Trixi.nelements(dg, cache)
    nfaces = Trixi.ninterfaces(dg, cache)
    primary = view(interfaces.element_ids, 1, :)
    secondary = view(interfaces.element_ids, 2, :)
    cell_interfaces = [Tuple{Int,Bool}[] for _ in 1:nelements]
    for face in 1:nfaces
        push!(cell_interfaces[primary[face]], (face, true))
        push!(cell_interfaces[secondary[face]], (face, false))
    end
    order = Matrix{Int}(undef, nelements, length(vx))
    centers = dropdims(sum(cache.elements.node_coordinates; dims = (2, 3));
                       dims = (2, 3)) / n^2
    for direction in eachindex(vx)
        outgoing = [Int[] for _ in 1:nelements]
        speed_scale = hypot(vx[direction], vy[direction])
        for face in 1:nfaces
            p, s = primary[face], secondary[face]
            side = interfaces.element_side_ids[1, face]
            primary_to_secondary = secondary_to_primary = false
            # A curved face may send different nodes in opposite directions
            for node in 1:n
                nx, ny = normals[1, node, side, p], normals[2, node, side, p]
                speed = vx[direction] * nx + vy[direction] * ny
                tolerance = 1.0e-12 * speed_scale * hypot(nx, ny)
                primary_to_secondary |= speed > tolerance
                secondary_to_primary |= speed < -tolerance
            end
            primary_to_secondary && push!(outgoing[p], s)
            secondary_to_primary && push!(outgoing[s], p)
        end
        direction_order = sweep_order(outgoing, centers,
                                                           vx[direction], vy[direction])
        order[:, direction] .= direction_order
    end
    return order, cell_interfaces
end

function sweep_order(outgoing, centers, vx, vy)
    indegree = zeros(Int, length(outgoing))
    for source in eachindex(outgoing), target in outgoing[source]
        indegree[target] += 1
    end
    order = findall(iszero, indegree)
    sort!(order; by = cell -> vx * centers[1, cell] + vy * centers[2, cell])
    head = 1
    while head <= length(order)
        source = order[head]
        head += 1
        for target in outgoing[source]
            indegree[target] -= 1
            indegree[target] == 0 && push!(order, target)
        end
    end
    acyclic = length(order) == length(outgoing)
    if !acyclic
        # Cycles have no exact upstream order, so finish by position along the flow
        remainder = findall(>(0), indegree)
        sort!(remainder; by = cell -> vx * centers[1, cell] + vy * centers[2, cell])
        append!(order, remainder)
    end
    return order
end

release_preconditioner!(::TransportSweepPreconditioner) = nothing
