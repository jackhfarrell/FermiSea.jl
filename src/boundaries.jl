# Reservoirs and walls supply states for particles entering the device
# The flux at each edge then uses that incoming state and the interior outgoing state

"""
    FixedReservoir(delta_mu=0)

Impose the constant electrochemical energy departure `delta_mu` on incoming
particles. Positive normal velocity leaves the device and retains its state.
"""
struct FixedReservoir{T<:Real}
    delta_mu::T

    function FixedReservoir(delta_mu::Real = 0)
        isfinite(delta_mu) || throw(ArgumentError("reservoir offset must be finite"))
        return new{Float64}(Float64(delta_mu))
    end
end

"""
    FloatingContact(; target_current=0)

Solve for the reservoir energy departure at a prescribed outward charge
current. The default zero current models an ideal voltage probe.
"""
struct FloatingContact{T<:Real}
    target_current::T
end

function FloatingContact(; target_current::Real = 0)
    isfinite(target_current) || throw(ArgumentError("target_current must be finite"))
    return FloatingContact{Float64}(Float64(target_current))
end

"""
    DiffuseWall()

Emit a constant incoming departure chosen to balance outward particle flux.
The normal points out of the device. Positive normal velocity leaves it.
"""
struct DiffuseWall end

mutable struct PreparedDiffuseWall
    kernels::Dict{Tuple{UInt,Float64,Float64},Any}
    lock::ReentrantLock
end

PreparedDiffuseWall() = PreparedDiffuseWall(Dict{Tuple{UInt,Float64,Float64},Any}(),
                                            ReentrantLock())
prepare_boundary(::DiffuseWall) = PreparedDiffuseWall()

"""
    MaxwellWall(; p_scatter=0)

Reflect the circular Fermi surface by matching flux in tangential-momentum
order. The discrete map preserves particle flux and a constant equilibrium.
`p_scatter` is the diffuse scattering probability: zero gives fully specular
reflection and one gives fully diffuse reflection. Intermediate values mix the
two maps.
"""
struct MaxwellWall{T<:Real}
    p_scatter::T
end

mutable struct PreparedMaxwellWall{W}
    wall::W
    kernels::Dict{Tuple{UInt,Float64,Float64},Any}
    lock::ReentrantLock
end

PreparedMaxwellWall(wall::MaxwellWall) =
    PreparedMaxwellWall(wall, Dict{Tuple{UInt,Float64,Float64},Any}(), ReentrantLock())
prepare_boundary(wall::MaxwellWall) = PreparedMaxwellWall(wall)
prepare_boundary(boundary) = boundary

function MaxwellWall(; p_scatter::Real = 0)
    isfinite(p_scatter) && 0 <= p_scatter <= 1 || throw(ArgumentError(
        "p_scatter must lie between zero (specular) and one (diffuse)"))
    return MaxwellWall(Float64(p_scatter))
end

"""
    IncomingProfile(profile; origin=(0, 0))

Set the incoming state from a solved `ChannelProfile`. We evaluate the profile
at the transverse coordinate relative to `origin`. Its Fermi surface must match
the device surface. Outgoing particles keep the interior state. Use the same model
length units for coordinates, width, and device geometry. We use this boundary
to approximate an infinite lead.
"""
struct IncomingProfile{P,O}
    profile::P
    origin::O
end

IncomingProfile(profile; origin = (0.0, 0.0)) =
    IncomingProfile(profile, SVector{2,Float64}(origin))

function compatible_profile_surface(profile_surface, surface)
    # A profile is indexed by momentum node, so every node must mean the same thing
    return profile_surface.momenta == surface.momenta &&
           profile_surface.velocities == surface.velocities &&
           profile_surface.energies == surface.energies &&
           profile_surface.weights == surface.weights &&
           profile_surface.mu == surface.mu &&
           profile_surface.charge == surface.charge &&
           profile_surface.spin_degeneracy == surface.spin_degeneracy
end

@inline outward_normal(orientation, direction, T) = orientation == 1 ?
    SVector{2,T}(isodd(direction) ? -1 : 1, 0) :
    SVector{2,T}(0, isodd(direction) ? -1 : 1)

@inline function boundary_node_coordinate(coordinates, node, side, element, n)
    side == 1 && return SVector(coordinates[1, node, 1, element],
                               coordinates[2, node, 1, element])
    side == 2 && return SVector(coordinates[1, n, node, element],
                               coordinates[2, n, node, element])
    side == 3 && return SVector(coordinates[1, node, n, element],
                               coordinates[2, node, n, element])
    return SVector(coordinates[1, 1, node, element], coordinates[2, 1, node, element])
end

reservoir_value(bc::FixedReservoir) = bc.delta_mu
reservoir_value(::FloatingContact) = 0.0

function fill_boundary_state!(output, bc::Union{FixedReservoir,FloatingContact},
                              u_inner, normal, equations::BoltzmannEquation,
                              x = nothing, t = 0)
    surface = equations.surface
    # The contact sets incoming particles. Outgoing particles carry the interior state
    @inbounds for i in eachindex(output)
        vn = normal[1] * surface.velocities[i, 1] + normal[2] * surface.velocities[i, 2]
        output[i] = vn < 0 ? reservoir_value(bc) : u_inner[i]
    end
    return output
end

function wall_flux_data(surface, normal)
    speed = [dot(view(surface.velocities, i, :), normal) for i in eachindex(surface.weights)]
    # Near-grazing nodes have too little normal flux to assign reliably to either side
    grazing_tolerance = sqrt(eps(Float64)) * maximum(abs, speed; init = 0.0)
    outgoing = findall(>(grazing_tolerance), speed)
    incoming = findall(<(-grazing_tolerance), speed)
    isempty(outgoing) &&
        throw(ArgumentError("wall has no outgoing momentum nodes"))
    isempty(incoming) &&
        throw(ArgumentError("wall has no incoming momentum nodes"))
    outgoing_measure = [surface.weights[j] * speed[j] for j in outgoing]
    incoming_measure = [-surface.weights[i] * speed[i] for i in incoming]
    # Equal incoming and outgoing measures let a constant equilibrium survive the wall
    # map
    scale = max(sum(outgoing_measure), sum(incoming_measure), eps(Float64))
    abs(sum(outgoing_measure) - sum(incoming_measure)) <= 2.0e-10 * scale || throw(
        ArgumentError(
            "wall quadrature cannot preserve equilibrium: outgoing and incoming flux " *
                "measures differ by $(sum(outgoing_measure) - sum(incoming_measure))"
        )
    )
    return (; speed, outgoing, incoming, outgoing_measure, incoming_measure)
end

function apply_diffuse_data!(output, u_inner, data)
    copyto!(output, u_inner)
    (; outgoing, incoming, outgoing_measure, incoming_measure) = data
    # A constant incoming amplitude closes particle flux and preserves equilibrium
    amplitude = sum(outgoing_measure[j] * u_inner[outgoing[j]]
                    for j in eachindex(outgoing_measure)) / sum(incoming_measure)
    output[incoming] .= amplitude
    return output
end

function fill_boundary_state!(output, ::DiffuseWall, u_inner, normal,
                              equations::BoltzmannEquation, x = nothing, t = 0)
    return apply_diffuse_data!(output, u_inner, wall_flux_data(equations.surface, normal))
end

function prepared_diffuse_data!(wall::PreparedDiffuseWall, surface, normal)
    magnitude = hypot(normal[1], normal[2])
    unit = normal ./ magnitude
    # The kernel depends on surface and normal direction, not the normal's surface scale
    key = (objectid(surface.weights), round(unit[1]; digits = 13),
           round(unit[2]; digits = 13))
    lock(wall.lock) do
        return get!(wall.kernels, key) do
            wall_flux_data(surface, unit)
        end
    end
end


function fill_boundary_state!(output, wall::PreparedDiffuseWall, u_inner, normal,
                              equations::BoltzmannEquation, x = nothing, t = 0)
    return apply_diffuse_data!(output, u_inner,
                               prepared_diffuse_data!(wall, equations.surface, normal))
end

function cumulative_coupling(source_mass, source_order, target_mass, target_order)
    # Match flux in tangential-momentum order while preserving each node's flux measure
    coupling = zeros(length(target_mass), length(source_mass))
    source_left = copy(source_mass)
    target_left = copy(target_mass)
    i = j = 1
    mass_scale = max(sum(source_mass), sum(target_mass))
    mass_scale > 0 || return coupling
    tolerance = 64eps(Float64) * mass_scale
    while i <= length(target_order) && j <= length(source_order)
        row, column = target_order[i], source_order[j]
        amount = min(target_left[row], source_left[column])
        coupling[row, column] += amount
        target_left[row] -= amount
        source_left[column] -= amount
        target_left[row] <= tolerance && (i += 1)
        source_left[column] <= tolerance && (j += 1)
    end
    maximum(abs, source_left; init = 0.0) <= tolerance || throw(ArgumentError(
        "outgoing flux does not supply the incoming flux measure"))
    maximum(abs, target_left; init = 0.0) <= tolerance || throw(ArgumentError(
        "outgoing flux does not balance the incoming flux measure"))
    return coupling
end

function specular_flux_kernel(surface, normal)
    (; outgoing, incoming, outgoing_measure, incoming_measure) =
        wall_flux_data(surface, normal)
    tangent = SVector(-normal[2], normal[1]) / hypot(normal[1], normal[2])
    source_order = sortperm(outgoing; by = node ->
        dot(view(surface.momenta, node, :), tangent))
    target_order = sortperm(incoming; by = node ->
        dot(view(surface.momenta, node, :), tangent))
    # Rescaling only removes the quadrature roundoff between opposite velocity pairs
    scaled_incoming = incoming_measure .* (sum(outgoing_measure) / sum(incoming_measure))
    coupling = cumulative_coupling(outgoing_measure, source_order,
                                   scaled_incoming, target_order)
    probability = sparse(coupling ./ reshape(outgoing_measure, 1, :))
    maximum(abs.(vec(sum(probability; dims = 1)) .- 1)) <= 5.0e-12 ||
        error("internal specular probability normalization failed")
    maximum(abs.(probability * outgoing_measure .- incoming_measure)) <=
        5.0e-10 * sum(incoming_measure) ||
        error("internal specular equilibrium balance failed")
    return (; incoming, outgoing, probability, outgoing_measure, incoming_measure)
end

function prepared_specular_data!(wall::PreparedMaxwellWall, surface, normal)
    magnitude = hypot(normal[1], normal[2])
    unit = normal ./ magnitude
    key = (objectid(surface.weights), round(unit[1]; digits = 13),
           round(unit[2]; digits = 13))
    lock(wall.lock) do
        return get!(wall.kernels, key) do
            specular_flux_kernel(surface, unit)
        end
    end
end

function apply_specular_data!(output, wall, u_inner, kernel)
    copyto!(output, u_inner)
    (; incoming, outgoing, probability, outgoing_measure, incoming_measure) = kernel
    diffuse = sum(outgoing_measure[column] * u_inner[outgoing[column]]
                  for column in eachindex(outgoing_measure)) / sum(incoming_measure)
    output[incoming] .= 0
    for column in axes(probability, 2)
        incident_flux = outgoing_measure[column] * u_inner[outgoing[column]]
        for pointer in nzrange(probability, column)
            row = probability.rowval[pointer]
            output[incoming[row]] += probability.nzval[pointer] * incident_flux
        end
    end
    for (row, node) in enumerate(incoming)
        # Convert reflected flux back to a state before mixing in diffuse reflection
        specular = output[node] / incoming_measure[row]
        output[node] = (1 - wall.p_scatter) * specular + wall.p_scatter * diffuse
    end
    return output
end

function fill_boundary_state!(output, wall::MaxwellWall, u_inner, normal,
                              equations::BoltzmannEquation, x = nothing, t = 0)
    wall.p_scatter == 1 &&
        return fill_boundary_state!(
            output,
            DiffuseWall(),
            u_inner,
            normal,
            equations,
            x,
            t
        )
    return apply_specular_data!(output, wall, u_inner,
                                specular_flux_kernel(equations.surface, normal))
end


function fill_boundary_state!(output, prepared::PreparedMaxwellWall, u_inner, normal,
                              equations::BoltzmannEquation, x = nothing, t = 0)
    wall = prepared.wall
    wall.p_scatter == 1 &&
        return fill_boundary_state!(
            output,
            DiffuseWall(),
            u_inner,
            normal,
            equations,
            x,
            t
        )
    return apply_specular_data!(output, wall, u_inner,
        prepared_specular_data!(prepared, equations.surface, normal))
end

function fill_boundary_state!(output, bc::IncomingProfile, u_inner, normal,
    equations::BoltzmannEquation, x = nothing, t = 0)
    x === nothing &&
        throw(ArgumentError("incoming profiles require boundary coordinates"))
    profile_surface = bc.profile.model.surface
    surface = equations.surface
    compatible_profile_surface(profile_surface, surface) ||
        throw(ArgumentError("incoming profile uses an incompatible Fermi surface"))
    unit_normal = SVector{2,Float64}(normal) / hypot(normal[1], normal[2])
    abs(abs(dot(unit_normal, bc.profile.axis)) - 1) <= sqrt(eps(Float64)) ||
        throw(ArgumentError(
            "incoming profile axis is not aligned with the contact normal"))
    q = dot(SVector{2,Float64}(x) - bc.origin, bc.profile.transverse)
    half_width = bc.profile.width / 2
    endpoint_tolerance = 32eps(Float64) * max(1.0, half_width)
    -half_width - endpoint_tolerance <= q <= half_width + endpoint_tolerance ||
        throw(DomainError(q, "contact coordinate lies outside the incoming profile"))
    q = clamp(q, -half_width, half_width)
    profile_state = bc.profile(q)
    @inbounds for i in eachindex(output)
        vn = dot(view(surface.velocities, i, :), normal)
        output[i] = vn < 0 ? profile_state[i] : u_inner[i]
    end
    return output
end

function boundary_state(bc, u, normal, equations, x = nothing, t = 0)
    output = MVector{length(u),promote_type(eltype(u),Float64)}(undef)
    fill_boundary_state!(output, bc, u, normal, equations, x, t)
    return SVector{length(u)}(output)
end

diffuse_state(bc::DiffuseWall, u, normal, equations) =
    boundary_state(bc, u, normal, equations)
reservoir_state(bc::FixedReservoir, u, normal, equations) =
    boundary_state(bc, u, normal, equations)

function boundary_flux(bc, u_inner, orientation, direction, x, t, equations)
    normal = outward_normal(orientation, direction, eltype(u_inner))
    u_boundary = boundary_state(bc, u_inner, normal, equations, x, t)
    # Trixi's upwind flux expects left and right states in coordinate order
    return isodd(direction) ? flux_upwind(u_boundary, u_inner, orientation, equations) :
           flux_upwind(u_inner, u_boundary, orientation, equations)
end

const PhysicalBoundary = Union{FixedReservoir,FloatingContact,DiffuseWall,
                               PreparedDiffuseWall,MaxwellWall,PreparedMaxwellWall,
                               IncomingProfile}

@inline (bc::PhysicalBoundary)(u_inner, orientation, direction, x, t,
                               surface_flux, equations) =
    boundary_flux(bc, u_inner, orientation, direction, x, t, equations)

# Unstructured normals include the surface Jacobian. Donor selection only uses their
# direction, while the numerical flux retains their full magnitude
@inline function (bc::PhysicalBoundary)(u_inner, normal::AbstractVector, x, t,
                                        surface_flux, equations)
    u_boundary = boundary_state(bc, u_inner, normal, equations, x, t)
    return surface_flux(u_inner, u_boundary, normal, equations)
end
