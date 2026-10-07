# Trixi is designed for small sets of hyperbolic PDEs, so its default volume integrl
# is not well-optimized for this large linear system.  So we implement custom volume
# and surface integrals for speed

"""
    PackedWeakForm()

Use this with Trixi's unstructured DG mesh. It processes momentum values in
batches across each element, with reusable arrays for streaming and local sources.
The surface flux must be `flux_upwind`.
"""
struct PackedWeakForm <: Trixi.AbstractVolumeIntegral end

# Source workspaces keep repeated batches from allocating new arrays
packed_source_workspace(::Nothing, ncolumns, nvars) = nothing
packed_source_workspace(source::Callaway, ncolumns, nvars) =
    collision_workspace(source, ncolumns)
packed_source_workspace(source::PreparedLocalSource{<:MagneticField}, ncolumns, nvars) =
    magnetic_workspace(source.operator, ncolumns)
packed_source_workspace(
    source::PreparedLocalSource{<:TomographicCollision},
    ncolumns,
    nvars
) =
    collision_workspace(source.operator, ncolumns)
packed_source_workspace(
    source::PreparedLocalSource{<:CustomCollision},
    ncolumns,
    nvars
) =
    collision_workspace(source.operator)
packed_source_workspace(source::SourceTerms, ncolumns, nvars) =
    (terms = map(term -> packed_source_workspace(term, ncolumns, nvars), source.terms),
     work = zeros(Float64, nvars, ncolumns))
packed_source_workspace(source, ncolumns, nvars) = nothing

apply_positive_source!(out, source::Callaway, input, workspace, x, t, equations) =
    apply_collision!(out, source, input, workspace)

# Positive left-hand source actions give the electric drive a minus sign here
# The drive does not depend on phi. Write it in place instead of building one
# momentum-sized static vector at every node
function apply_positive_source!(out, source::ElectricDrive, input, workspace, x, t,
                                equations)
    velocities = equations.surface.velocities
    charged_ex, charged_ey = source.charged_field
    @inbounds for column in axes(out, 2), mode in axes(out, 1)
        out[mode, column] =
            -(charged_ex * velocities[mode, 1] + charged_ey * velocities[mode, 2])
    end
    return out
end

apply_positive_source!(out, source::PreparedLocalSource{<:MagneticField}, input, work,
                       x, t, equations) =
    apply_magnetic!(out, source.operator, input, work)
apply_positive_source!(out,
                       source::PreparedLocalSource{<:Union{TomographicCollision,
                                                           CustomCollision}},
                       input, work, x, t, equations) =
    apply_collision!(out, source.operator, input, work)

function apply_positive_source!(
    out,
    source::SourceTerms,
    input,
    workspace,
    x,
    t,
    equations
)
    fill!(out, 0.0)
    for (term, term_workspace) in zip(source.terms, workspace.terms)
        apply_positive_source!(workspace.work, term, input, term_workspace,
                               x, t, equations)
        out .+= workspace.work
    end
    return out
end

apply_positive_source!(out, ::Nothing, input, workspace, x, t, equations) =
    fill!(out, 0.0)

function apply_positive_source!(out, source, input, workspace, x, t, equations)
    nvars = size(input, 1)
    for column in axes(input, 2)
        value = source(SVector{nvars}(view(input, :, column)),
                       SVector{2}(view(x, :, column)), t, equations)
        @views out[:, column] .= -value
    end
    return out
end

function Trixi.create_cache(mesh::Trixi.UnstructuredMesh2D,
                            equations::BoltzmannEquation, ::PackedWeakForm,
                            dg, cache, uEltype)
    dg.surface_integral.surface_flux === flux_upwind || throw(ArgumentError(
        "PackedWeakForm requires flux_upwind"))
    nvars = Trixi.nvariables(equations)
    n = Trixi.nnodes(dg)
    # Threads work on different elements and faces, so each needs its own scratch
    scratch = [packed_volume_scratch(nvars, n) for _ in 1:Threads.maxthreadid()]
    faces = [packed_face_scratch(nvars, n) for _ in 1:Threads.maxthreadid()]
    return (; packed = (; scratch, faces))
end

function packed_volume_scratch(nvars, n)
    q = n^2
    return (; states = Matrix{Float64}(undef, nvars, q),
            flux_x = Matrix{Float64}(undef, nvars, q),
            flux_y = Matrix{Float64}(undef, nvars, q),
            source_action = Matrix{Float64}(undef, nvars, q),
            flux_xi = Array{Float64}(undef, nvars, n, n),
            flux_eta = Array{Float64}(undef, nvars, n, n),
            source = Ref{Any}(nothing), source_key = Ref{Any}(nothing))
end

packed_face_scratch(nvars, n) =
    (; left = Matrix{Float64}(undef, nvars, n),
     right = Matrix{Float64}(undef, nvars, n),
     flux = Matrix{Float64}(undef, nvars, n),
     nx = Vector{Float64}(undef, n), ny = Vector{Float64}(undef, n))

function Trixi.rhs!(du, u, t, mesh::Trixi.UnstructuredMesh2D,
                    equations::BoltzmannEquation, boundary_conditions, source_terms,
                    dg::Trixi.DG{<:Any,<:Any,<:Any,<:PackedWeakForm}, cache)
    Trixi.set_zero!(du, dg, cache)
    packed_volume_integral!(du, u, equations, source_terms, dg, cache, t)
    Trixi.prolong2interfaces!(cache, u, mesh, equations, dg)
    packed_interface_flux!(equations, dg, cache)
    Trixi.prolong2boundaries!(cache, u, mesh, equations, dg)
    packed_boundary_flux!(boundary_conditions, equations, dg, cache)
    Trixi.calc_surface_integral!(du, u, mesh, equations, dg.surface_integral, dg, cache)
    Trixi.apply_jacobian!(du, mesh, equations, dg, cache)
    return nothing
end

function packed_interface_flux!(equations, dg, cache)
    interfaces = cache.interfaces
    normals = cache.elements.normal_directions
    output = cache.elements.surface_flux_values
    velocities = equations.surface.velocities
    n = Trixi.nnodes(dg)
    nvars = Trixi.nvariables(equations)
    # Faces may list their nodes in opposite directions on the two elements
    Threads.@threads :dynamic for face in 1:Trixi.ninterfaces(dg, cache)
        work = cache.packed.faces[Threads.threadid()]
        primary = interfaces.element_ids[1, face]
        secondary = interfaces.element_ids[2, face]
        primary_side = interfaces.element_side_ids[1, face]
        secondary_side = interfaces.element_side_ids[2, face]
        secondary_node = interfaces.start_index[face]
        increment = interfaces.index_increment[face]
        @inbounds for node in 1:n
            nx = normals[1, node, primary_side, primary]
            ny = normals[2, node, primary_side, primary]
            work.nx[node], work.ny[node] = nx, ny
            for mode in 1:nvars
                left = interfaces.u[1, mode, node, face]
                right = interfaces.u[2, mode, secondary_node, face]
                speed = nx * velocities[mode, 1] + ny * velocities[mode, 2]
                work.flux[mode, node] = speed * ifelse(speed >= 0, left, right)
            end
            secondary_node += increment
        end
        secondary_node = interfaces.start_index[face]
        @inbounds for node in 1:n, mode in 1:nvars
            flux = work.flux[mode, node]
            # The same interface flux enters the two elements with opposite signs
            output[mode, node, primary_side, primary] = flux
            output[mode, secondary_node, secondary_side, secondary] = -flux
            mode == nvars && (secondary_node += increment)
        end
    end
    return nothing
end

function packed_boundary_flux!(conditions, equations, dg, cache)
    foreach(
        conditions.boundary_condition_types,
        conditions.boundary_indices
    ) do bc, indices
        packed_boundary_group!(bc, indices, equations, dg, cache)
    end
    return nothing
end

function packed_boundary_group!(bc, indices, equations, dg, cache)
    boundaries = cache.boundaries
    normals = cache.elements.normal_directions
    output = cache.elements.surface_flux_values
    velocities = equations.surface.velocities
    n = Trixi.nnodes(dg)
    nvars = Trixi.nvariables(equations)
    Threads.@threads :dynamic for local_index in eachindex(indices)
        boundary = indices[local_index]
        work = cache.packed.faces[Threads.threadid()]
        element = boundaries.element_id[boundary]
        side = boundaries.element_side_id[boundary]
        @inbounds for node in 1:n
            nx = normals[1, node, side, element]
            ny = normals[2, node, side, element]
            normal = SVector(nx, ny)
            x = boundary_node_coordinate(cache.elements.node_coordinates, node, side,
                                         element, n)
            for mode in 1:nvars
                work.left[mode, node] = boundaries.u[mode, node, boundary]
            end
            # Wall laws may mix momentum nodes, so build the whole incoming state first
            fill_boundary_state!(view(work.right, :, node), bc,
                                 view(work.left, :, node), normal, equations, x, 0.0)
            for mode in 1:nvars
                speed = nx * velocities[mode, 1] + ny * velocities[mode, 2]
                output[mode, node, side, element] =
                    speed * ifelse(speed >= 0, work.left[mode, node],
                                   work.right[mode, node])
            end
        end
    end
    return nothing
end

function packed_volume_integral!(du, u, equations, sources, dg, cache, t)
    n = Trixi.nnodes(dg)
    nvars = Trixi.nvariables(equations)
    for scratch in cache.packed.scratch
        # Rebuild source scratch only when the source object changes
        if scratch.source_key[] !== sources
            scratch.source[] = packed_source_workspace(sources, n^2, nvars)
            scratch.source_key[] = sources
        end
    end
    Threads.@threads :dynamic for element in 1:Trixi.nelements(dg, cache)
        scratch = cache.packed.scratch[Threads.threadid()]
        packed_element!(du, u, element, equations, sources, scratch,
                        scratch.source[], cache.elements.node_coordinates, t,
                        cache.elements.contravariant_vectors,
                        cache.elements.inverse_jacobian, dg.basis.derivative_hat,
                        n, nvars)
    end
    return nothing
end

function packed_element!(du, u, element, equations, sources, scratch, source_workspace,
                         coordinates, t,
                         contravariant, inverse_jacobian, derivative_hat, n, nvars)
    # Flatten spatial nodes into columns so transport and sources work on whole arrays
    @inbounds for j in 1:n, i in 1:n
        column = (j - 1) * n + i
        for v in 1:nvars
            scratch.states[v, column] = u[v, i, j, element]
        end
    end
    streaming_fluxes!(scratch.flux_x, scratch.flux_y, equations, scratch.states)
    apply_positive_source!(scratch.source_action, sources, scratch.states,
        source_workspace,
        reshape(view(coordinates, :, :, :, element), 2, :),
        t, equations)
    Dt = transpose(derivative_hat)
    # The mesh vectors turn physical fluxes into fluxes on the reference element
    @inbounds for j in 1:n, i in 1:n
        column = (j - 1) * n + i
        a11 = contravariant[1, 1, i, j, element]
        a12 = contravariant[2, 1, i, j, element]
        a21 = contravariant[1, 2, i, j, element]
        a22 = contravariant[2, 2, i, j, element]
        for v in 1:nvars
            fx, fy = scratch.flux_x[v, column], scratch.flux_y[v, column]
            scratch.flux_xi[v, i, j] = a11 * fx + a12 * fy
            scratch.flux_eta[v, i, j] = a21 * fx + a22 * fy
        end
    end
    @inbounds for j in 1:n, i in 1:n, v in 1:nvars
        derivative = 0.0
        for k in 1:n
            derivative = muladd(scratch.flux_xi[v, k, j], Dt[k, i], derivative)
            derivative = muladd(scratch.flux_eta[v, i, k], Dt[k, j], derivative)
        end
        column = (j - 1) * n + i
        jacobian = inv(inverse_jacobian[i, j, element])
        du[v, i, j, element] += derivative + jacobian * scratch.source_action[v, column]
    end
    return nothing
end
