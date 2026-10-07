# a 1D channel solver with uniform longitudinal electric field forcing the flow,
# and we solve for the transverse current profile.  We want this so that we can
# use it as a BC on a 2D solve.  Here rather than DG we use finite-volume methods
# for the 1D problem, as resolution and performance are not so much an issue

struct DevelopedChannel{M,S,W}
    model::M
    width::Float64
    axis::SVector{2,Float64}
    transverse::SVector{2,Float64}
    walls::W
    sources::S
    charged_field::Float64
end

"""
    developed_channel(model; width, axis=(1, 0), walls, source_terms, electric_field)

Set up an infinitely long channel with a longitudinal electric field `E`.
We multiply by the signed carrier charge once, giving forcing `q E v_axis`.
Use consistent model units for widths, velocities, and rates.
The former `drive` keyword is no longer supported; it represented `qE`, with
charge already included.
"""
function developed_channel(model::BoltzmannEquation; width::Real, axis = (1.0, 0.0),
                           walls = (DiffuseWall(), DiffuseWall()),
                           source_terms = nothing, electric_field::Real)
    isfinite(width) && width > 0 || throw(ArgumentError(
        "channel width must be finite and positive"))
    direction = SVector{2,Float64}(axis)
    norm(direction) > 0 || throw(ArgumentError("channel axis must be nonzero"))
    direction /= norm(direction)
    transverse = SVector(-direction[2], direction[1])
    length(walls) == 2 ||
        throw(ArgumentError("walls must contain lower and upper laws"))
    isfinite(electric_field) ||
        throw(ArgumentError("channel electrical drive must be finite"))
    prepared_walls = map(prepare_boundary, walls)
    prepared_sources = prepare_source(source_terms)
    return DevelopedChannel(model, Float64(width), direction, transverse,
        prepared_walls,
        prepared_sources, Float64(model.surface.charge * electric_field))
end

"""
ChannelProfile

The solved state of a long channel. Each column of `states` holds the momentum
values at one position across the channel. `coordinates` gives those positions,
and `cell_widths` gives the width of each cell. We also store the average particle
current, the solve residual, and the outward particle flux at each wall.

Call `profile(y)` to get a new momentum vector at position `y`. It interpolates
between cell centres and extrapolates from the nearest two centres near a wall.
The allowed range is `[-width/2, width/2]`. `charged_field` stores `qE`, the
longitudinal electric field multiplied by the carrier charge.
"""
struct ChannelProfile{C,W,S}
    model
    width::Float64
    axis::SVector{2,Float64}
    transverse::SVector{2,Float64}
    walls::W
    sources::S
    charged_field::Float64
    coordinates::C
    cell_widths::C
    states::Matrix{Float64}
    mean_current::SVector{2,Float64}
    residual::Float64
    wall_particle_flux::SVector{2,Float64}
end

function (profile::ChannelProfile)(transverse_coordinate::Real)
    transverse_coordinate = Float64(transverse_coordinate)
    half_width = profile.width / 2
    -half_width <= transverse_coordinate <= half_width || throw(
        DomainError(
            transverse_coordinate,
            "channel profile support is [-$half_width, $half_width]")
    )
    points = profile.coordinates
    # The same two-centre line reaches the wall even though no cell is centred there
    right = searchsortedfirst(points, transverse_coordinate)
    left, right = right <= 1 ? (1, 2) : right > length(points) ?
                  (length(points) - 1, length(points)) : (right - 1, right)
    fraction = (transverse_coordinate - points[left]) / (points[right] - points[left])
    return @views (1 - fraction) .* profile.states[:, left] .+
                  fraction .* profile.states[:, right]
end

"""
    channel_current(profile; component=:longitudinal)

Return the particle current averaged across the channel width. By default, this
is the current along the channel. Use `component=:transverse` for the current
across it, or `component=:global` for both components in the model's coordinates.
Multiply the result by `profile.model.surface.charge` to get charge current.
"""
function channel_current(profile::ChannelProfile; component::Symbol = :longitudinal)
    component === :global && return profile.mean_current
    component === :longitudinal && return dot(profile.axis, profile.mean_current)
    component === :transverse && return dot(profile.transverse, profile.mean_current)
    throw(ArgumentError("component must be :longitudinal, :transverse, or :global"))
end

function solve(channel::DevelopedChannel; ncells::Integer = 48, grading::Real = 1.5,
               tolerance::Real = 1.0e-9)
    ncells >= 2 || throw(ArgumentError("channel ncells must be at least two"))
    isfinite(grading) && grading >= 0 || throw(ArgumentError(
        "channel grading must be finite and nonnegative"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError(
        "channel tolerance must be finite and positive"))
    model, surface = channel.model, channel.model.surface
    n = length(surface)
    centers, widths = channel_mesh(channel.width, Int(ncells), Float64(grading))
    # Each momentum node takes its face state from the side its velocity comes from
    across = Diagonal(surface.velocities * channel.transverse)
    positive = Diagonal(max.(diag(across), 0.0))
    negative = Diagonal(min.(diag(across), 0.0))
    lower_map = channel_wall_map(model, channel.walls[1], -channel.transverse)
    upper_map = channel_wall_map(model, channel.walls[2], channel.transverse)
    collision = local_source_matrix(channel.sources, model)
    left, right, lower, upper, divergence = channel_reconstructions(centers, widths)
    # Spatial reconstruction and momentum transport stay separate in these products
    flux = kron(left, sparse(positive)) + kron(right, sparse(negative)) +
           kron(lower, sparse(across * lower_map)) +
           kron(upper, sparse(across * upper_map))
    operator = kron(divergence, sparse(I, n, n)) * flux +
               kron(spdiagm(0 => widths), sparse(collision))
    gauges_local = conserved_channel_gauges(surface, collision)
    check_channel_constant_nullspace(collision, lower_map, upper_map, gauges_local)
    average_weights = widths ./ channel.width
    gauges = kron(sparse(reshape(average_weights, 1, :)), sparse(gauges_local))
    ngauges = size(gauges, 1)
    forcing = @. channel.charged_field *
                 (channel.axis[1] * surface.velocities[:, 1] +
                  channel.axis[2] * surface.velocities[:, 2])
    state_rhs = kron(widths, forcing)
    rhs = vcat(state_rhs, zeros(ngauges))
    # The bordered system fixes conserved offsets while enforcing the channel equation
    system = [operator sparse(gauges'); gauges spzeros(ngauges, ngauges)]
    full_solution = system \ rhs
    all(isfinite, full_solution) || throw(ArgumentError(
        "developed-channel system is singular"))
    state_vector = view(full_solution, 1:(n * ncells))
    multipliers = view(full_solution, (n * ncells + 1):length(full_solution))
    equation_defect = operator * state_vector - state_rhs
    gauge_defect = gauges * state_vector
    scale = norm(state_rhs)
    iszero(scale) && (scale = norm(operator, Inf) * norm(state_vector))
    iszero(scale) && (scale = 1.0)
    residual = max(norm(equation_defect) / scale,
        norm(gauge_defect) / max(norm(gauges, Inf) * norm(state_vector), eps()))
    reaction = norm(gauges' * multipliers) / scale
    # A nonzero constraint reaction means the drive excites a conserved mode with no
    # steady balance
    reaction <= tolerance || throw(ArgumentError(
        "incompatible electrical channel drive excites a conserved mode " *
        "(constraint reaction $reaction)"))
    residual <= tolerance || throw(ArgumentError(
        "developed-channel residual $residual exceeds tolerance $tolerance"))
    states = Matrix(reshape(state_vector, n, ncells))
    currents =
        reduce(hcat, particle_current(view(states, :, i), surface) for i in 1:ncells)
    mean_current = SVector{2,Float64}(currents * average_weights)
    lower_state = states * vec(lower[1, :])
    upper_state = states * vec(upper[end, :])
    wall_flux =
        SVector(channel_wall_flux(lower_state, lower_map, -channel.transverse, surface),
            channel_wall_flux(upper_state, upper_map, channel.transverse, surface))
    return ChannelProfile(model, channel.width, channel.axis, channel.transverse,
        channel.walls, channel.sources, channel.charged_field, centers, widths,
        states, mean_current, residual, wall_flux)
end

function channel_mesh(width, ncells, grading)
    xi = range(-1.0, 1.0; length = ncells + 1)
    # Narrower cells near the walls resolve the profiles set by wall scattering
    faces = iszero(grading) ? width .* xi ./ 2 :
            width .* tanh.(grading .* xi) ./ (2tanh(grading))
    widths = diff(faces)
    all(>(0), widths) || throw(ArgumentError("channel grading collapses cells"))
    return (faces[1:(end - 1)] .+ faces[2:end]) ./ 2, widths
end

function add_channel_reconstruction!(matrix, row, centers, cell, face)
    # Use nearby cell centres to estimate a face value on the graded mesh
    lo, hi = cell == 1 ? (1, 2) : cell == length(centers) ?
             (length(centers) - 1, length(centers)) : (cell - 1, cell + 1)
    fraction = (face - centers[cell]) / (centers[hi] - centers[lo])
    matrix[row, cell] += 1
    matrix[row, lo] -= fraction
    matrix[row, hi] += fraction
end

function channel_reconstructions(centers, widths)
    ncells = length(centers)
    faces = vcat(centers[1] - widths[1] / 2, centers .+ widths ./ 2)
    left = spzeros(ncells + 1, ncells)
    right = spzeros(ncells + 1, ncells)
    lower = spzeros(ncells + 1, ncells)
    upper = spzeros(ncells + 1, ncells)
    add_channel_reconstruction!(lower, 1, centers, 1, faces[1])
    add_channel_reconstruction!(upper, ncells + 1, centers, ncells, faces[end])
    for face in 2:ncells
        add_channel_reconstruction!(left, face, centers, face - 1, faces[face])
        add_channel_reconstruction!(right, face, centers, face, faces[face])
    end
    divergence = spzeros(ncells, ncells + 1)
    for cell in 1:ncells
        divergence[cell, cell] = -1
        divergence[cell, cell + 1] = 1
    end
    return left, right, lower, upper, divergence
end

function local_source_matrix(source, model)
    n = length(model.surface)
    source === nothing && return zeros(n, n)
    identity = Matrix{Float64}(I, n, n)
    zero_state = zeros(n, n)
    output = zeros(n, n)
    offset = zeros(n, n)
    # Removing the zero-state response leaves only the linear source action
    workspace = packed_source_workspace(source, n, n)
    coordinates = zeros(2, n)
    apply_positive_source!(output, source, identity, workspace, coordinates, 0.0, model)
    apply_positive_source!(
        offset,
        source,
        zero_state,
        workspace,
        coordinates,
        0.0,
        model
    )
    return output - offset
end

# Boundary laws can be affine, so subtracting their zero-state value isolates the
# homogeneous map
function channel_wall_map(model, wall, normal)
    n = length(model.surface)
    map = zeros(n, n)
    state = zeros(n)
    zero_state = zeros(n)
    offset = zeros(n)
    fill_boundary_state!(
        offset,
        wall,
        zero_state,
        normal,
        model,
        SVector(0.0, 0.0),
        0.0
    )
    for column in 1:n
        state[column] = 1
        fill_boundary_state!(view(map, :, column), wall, state, normal, model,
                             SVector(0.0, 0.0), 0.0)
        @views map[:, column] .-= offset
        state[column] = 0
    end
    return map
end

# A conserved population leaves the constant channel state undetermined
# Its weighted moment fixes the energy gauge
function conserved_channel_gauges(surface, collision)
    candidates = ones(length(surface), 1)
    rows = transpose(surface.weights .* candidates)
    defect = rows * collision
    factor = svd(transpose(defect); full = true)
    tolerance = max(size(defect)...) * sqrt(eps(Float64)) *
                norm(rows, Inf) * norm(collision, Inf)
    rank_defect = count(>(tolerance), factor.S)
    gauges = factor.V[:, (rank_defect + 1):end]' * rows
    isempty(gauges) && return zeros(0, length(surface))
    row_factor = svd(gauges; full = false)
    row_tolerance = max(size(gauges)...) * eps(Float64) *
                    maximum(row_factor.S; init = 0.0)
    row_rank = count(>(row_tolerance), row_factor.S)
    return row_rank == 0 ? zeros(0, length(surface)) : Matrix(row_factor.Vt[1:row_rank, :])
end

# A singular constant mode is valid only when the selected gauges constrain it
# completely
function check_channel_constant_nullspace(collision, lower_map, upper_map, gauges)
    n = size(collision, 1)
    blocks = (collision, lower_map - I, upper_map - I)
    scaled = map(blocks) do block
        scale = norm(block, Inf)
        iszero(scale) ? Matrix(block) : Matrix(block) ./ scale
    end
    conditions = vcat(scaled...)
    factor = svd(conditions; full = true)
    tolerance = max(size(conditions)...) * sqrt(eps(Float64)) *
                maximum(factor.S; init = 0.0)
    source_rank = count(>(tolerance), factor.S)
    nullity = n - source_rank
    iszero(nullity) && return nothing
    nullspace = factor.V[:, (source_rank + 1):end]
    constrained = gauges * nullspace
    constrained_rank = isempty(constrained) ? 0 : rank(constrained;
        atol = max(size(constrained)...) * sqrt(eps(Float64)) *
               maximum(abs, constrained; init = 0.0), rtol = 0.0)
    constrained_rank == nullity || throw(ArgumentError(
        "singular developed channel leaves $(nullity - constrained_rank) constant " *
        "kinetic mode(s) outside the population and heat gauges"))
    return nothing
end

function channel_wall_flux(state, wall_map, normal, surface)
    ghost = wall_map * state
    speed = surface.velocities * normal
    # Outgoing flux uses the channel state, while incoming flux uses the wall state
    donor = ifelse.(speed .>= 0, state, ghost)
    return dot(surface.weights, speed .* donor)
end
