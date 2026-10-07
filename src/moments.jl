# Contact observables use the boundary numerical flux rather than interior interpolation

"""
    particle_density(phi, surface)

Return the response-weighted electrochemical density-like moment of `phi`.
To recover the physical induced particle or charge density, we also need the
electrostatic potential.
"""
function particle_density(phi, surface::CircularFermiSurface)
    length(phi) == length(surface) || throw(DimensionMismatch("phi and surface disagree"))
    return dot(surface.weights, phi)
end

"""
    particle_current(phi, surface)

Return a two-component static vector `sum(weights[i] * velocities[i, :] * phi[i])`.
The state is unweighted and the result is particle current in model units.
No carrier charge or physical-unit conversion is applied.
"""
function particle_current(phi, surface::CircularFermiSurface)
    length(phi) == length(surface) || throw(DimensionMismatch("phi and surface disagree"))
    weighted_phi = surface.weights .* phi
    return SVector(dot(view(surface.velocities, :, 1), weighted_phi),
                   dot(view(surface.velocities, :, 2), weighted_phi))
end

"""
    charge_current(phi, surface)

Return the two-component particle current multiplied once by the signed
`surface.charge`. The default electron charge is negative in model units.
"""
charge_current(phi, surface::CircularFermiSurface) = surface.charge .* particle_current(phi, surface)

"""
    momentum_moment(phi, surface, kernel)

Evaluate `sum(surface.weights[i] * phi[i] * kernel(surface, i))`. The kernel may
return a scalar or a small static vector. The state remains unweighted.
"""
function momentum_moment(phi, surface::CircularFermiSurface, kernel)
    length(phi) == length(surface) || throw(DimensionMismatch("phi and surface disagree"))
    result = surface.weights[1] * phi[1] * kernel(surface, 1)
    for i in 2:length(surface)
        result += surface.weights[i] * phi[i] * kernel(surface, i)
    end
    return result
end

"""
    fit_local_equilibrium(phi, surface; drift=true)

Project onto `delta_mu + u dot p` in the response inner product. Return the
scalar `chemical_potential`, two-component `drift`, `fitted` state, `remainder`,
and `relative_residual`. With `drift=false`, keep only the constant mode and
return `nothing` for `drift`. We use the response weights of the circular surface
and do not convert physical units automatically.
"""
function fit_local_equilibrium(phi, surface::CircularFermiSurface; drift::Bool = true)
    length(phi) == length(surface) || throw(DimensionMismatch("phi and surface disagree"))
    delta_mu = particle_density(phi, surface) / sum(surface.weights)
    fitted = fill(delta_mu, length(surface))
    velocity = if drift
        # Circular midpoint quadrature makes the two momentum columns orthogonal
        ux = momentum_moment(phi, surface, (g, i) -> g.momenta[i, 1]) /
             sum(surface.weights .* abs2.(surface.momenta[:, 1]))
        uy = momentum_moment(phi, surface, (g, i) -> g.momenta[i, 2]) /
             sum(surface.weights .* abs2.(surface.momenta[:, 2]))
        fitted .+= surface.momenta * SVector(ux, uy)
        SVector(ux, uy)
    else
        nothing
    end
    remainder = phi .- fitted
    scale = sqrt(sum(surface.weights .* abs2.(phi)))
    residual = sqrt(sum(surface.weights .* abs2.(remainder))) / max(scale, eps(Float64))
    return (; chemical_potential = delta_mu, drift = velocity, fitted, remainder,
            relative_residual = residual)
end

"""
    contact_particle_flux(solution, side)

Return the total outward particle flux through the named boundary as a scalar.
We use the DG boundary numerical flux, with momentum response weights and the
spatial face measure included. Positive flux leaves the device. To include
carrier charge, use [`contact_current`](@ref).
"""
contact_particle_flux(solution::BoltzmannSolution, side::Symbol) =
    contact_weighted_flux(solution, side, (surface, i) -> 1.0)

"""
    contact_current(solution, side)

Return outward particle flux multiplied once by signed carrier charge.
Electron charge can make its sign opposite to particle flow. Reservoir offsets
are energies in model units.
"""
contact_current(solution::BoltzmannSolution, side::Symbol) =
    solution.problem.semi.equations.surface.charge * contact_particle_flux(solution, side)

function boundary_direction(side::Symbol)
    side === :x_neg && return (1, 1, SVector(-1.0, 0.0))
    side === :x_pos && return (1, 2, SVector(1.0, 0.0))
    side === :y_neg && return (2, 3, SVector(0.0, -1.0))
    side === :y_pos && return (2, 4, SVector(0.0, 1.0))
    throw(ArgumentError("side must be :x_neg, :x_pos, :y_neg, or :y_pos"))
end

function contact_weighted_flux(solution::BoltzmannSolution, side::Symbol, moment)
    problem = solution.problem
    semi = problem.semi
    mesh = semi.mesh
    mesh isa Union{Trixi.TreeMesh{2},Trixi.UnstructuredMesh2D} ||
        throw(
            ArgumentError(
                "contact integration supports TreeMesh{2} and UnstructuredMesh2D"
            )
        )
    bc = get_boundary_condition(problem.boundary_conditions, side)
    bc isa PhysicalBoundary || throw(ArgumentError("$side is not a physical boundary"))

    # Contact observables need only the interior trace. Running the full residual here
    # would repeat all volume physics and compile the large native state representation
    u = Trixi.wrap_array_native(solution.phi, semi)
    Trixi.prolong2boundaries!(semi.cache, u, mesh, semi.equations, semi.solver)
    if mesh isa Trixi.UnstructuredMesh2D
        boundaries = semi.cache.boundaries
        indices = semi.boundary_conditions.boundary_symbol_indices[side]
        weights = semi.solver.basis.weights
        normals = semi.cache.elements.normal_directions
        surface = semi.equations.surface
        flux = 0.0
        for boundary in indices
            element = boundaries.element_id[boundary]
            side_index = boundaries.element_side_id[boundary]
            for node in eachindex(weights)
                inner = view(boundaries.u, :, node, boundary)
                normal = view(normals, :, node, side_index, element)
                x = view(boundaries.node_coordinates, :, node, boundary)
                donor = boundary_state(bc, inner, normal, semi.equations, x, 0.0)
                for i in eachindex(surface.weights)
                    vn = dot(view(surface.velocities, i, :), normal)
                    flux +=
                        weights[node] * surface.weights[i] * moment(surface, i) * vn *
                            donor[i]
                end
            end
        end
        return flux
    end

    orientation, direction, normal = boundary_direction(side)
    boundaries = semi.cache.boundaries
    counts = boundaries.n_boundaries_per_direction
    first_boundary = direction == 1 ? 1 : 1 + sum(counts[1:(direction - 1)])
    last_boundary = sum(counts[1:direction])
    weights = semi.solver.basis.weights
    surface = semi.equations.surface
    flux = 0.0
    for boundary in first_boundary:last_boundary
        boundaries.orientations[boundary] == orientation ||
            error("boundary orientation mismatch")
        side_index = boundaries.neighbor_sides[boundary]
        element = boundaries.neighbor_ids[boundary]
        inverse_jacobian = semi.cache.elements.inverse_jacobian[element]
        surface_scale = inv(inverse_jacobian)
        for node in eachindex(weights)
            inner = SVector{length(surface)}(ntuple(i ->
                boundaries.u[side_index, i, node, boundary], Val(length(surface))))
            x = view(boundaries.node_coordinates, :, node, boundary)
            donor = boundary_state(bc, inner, normal, semi.equations, x, 0.0)
            for i in eachindex(surface.weights)
                vn = dot(view(surface.velocities, i, :), normal)
                flux += weights[node] * surface_scale * surface.weights[i] *
                        moment(surface, i) * vn * donor[i]
            end
        end
    end
    return flux
end
