"""
    BoltzmannEquation(surface)

Connect a Fermi surface to Trixi's two-dimensional transport equations. Each
unweighted angular value `phi[i]` streams with `surface.velocities[i, :]`.
We store the surface as `model.surface`. Use consistent model units for coordinates,
velocities, collision rates, and fields. You can also use this model to build a
Trixi semidiscretization directly.
"""
struct BoltzmannEquation{N,S<:CircularFermiSurface} <: Trixi.AbstractEquations{2,N}
    surface::S
    max_speed::Float64
end

function BoltzmannEquation(surface::CircularFermiSurface)
    n = length(surface)
    n > 0 || throw(ArgumentError("the Fermi surface is empty"))
    speed = maximum(hypot(surface.velocities[i, 1], surface.velocities[i, 2]) for i in 1:n)
    return BoltzmannEquation{n,typeof(surface)}(surface, Float64(speed))
end

# Trixi treats each momentum node as one variable with a fixed streaming speed
Trixi.varnames(::typeof(Trixi.cons2cons), equations::BoltzmannEquation{N}) where {N} =
    ntuple(i -> "phi_$i", Val(N))
Trixi.have_constant_speed(::BoltzmannEquation) = Trixi.True()

# Streaming is diagonal in momentum, so quadrature weights do not enter the spatial flux
@inline function Trixi.flux(u, orientation::Integer,
                            equations::BoltzmannEquation{N}) where {N}
    velocities = equations.surface.velocities
    return SVector{N}(ntuple(i -> velocities[i, orientation] * u[i], Val(N)))
end

@inline function Trixi.flux(u, normal::AbstractVector,
                            equations::BoltzmannEquation{N}) where {N}
    velocities = equations.surface.velocities
    nx, ny = normal
    return SVector{N}(ntuple(i ->
        (nx * velocities[i, 1] + ny * velocities[i, 2]) * u[i], Val(N)))
end

# Different momentum nodes can travel in opposite directions across the same face
# Upwinding each node separately preserves the kinetic transport direction
"""
    flux_upwind(phi_left, phi_right, normal_or_orientation, model)

Return a static momentum vector of upwind streaming fluxes. A positive normal
velocity takes the left state, and a negative one takes the right state.
Integer orientations select Cartesian axes. Vector normals include their magnitude.
"""
@inline function flux_upwind(u_left, u_right, orientation::Integer,
                             equations::BoltzmannEquation{N}) where {N}
    velocities = equations.surface.velocities
    return SVector{N}(ntuple(i -> begin
        speed = velocities[i, orientation]
        speed * ifelse(speed >= 0, u_left[i], u_right[i])
    end, Val(N)))
end

@inline function flux_upwind(u_left, u_right, normal::AbstractVector,
                             equations::BoltzmannEquation{N}) where {N}
    velocities = equations.surface.velocities
    nx, ny = normal
    return SVector{N}(ntuple(i -> begin
        speed = nx * velocities[i, 1] + ny * velocities[i, 2]
        speed * ifelse(speed >= 0, u_left[i], u_right[i])
    end, Val(N)))
end

# Flux bounds use the fastest projected node, including on faces with oblique normals
@inline Trixi.max_abs_speed_naive(u_left, u_right, orientation::Integer,
                                  equations::BoltzmannEquation) =
    maximum(abs, view(equations.surface.velocities, :, orientation))

@inline function Trixi.max_abs_speed_naive(u_left, u_right, normal::AbstractVector,
                                           equations::BoltzmannEquation)
    velocities = equations.surface.velocities
    nx, ny = normal
    return maximum(i -> abs(nx * velocities[i, 1] + ny * velocities[i, 2]),
                   axes(velocities, 1))
end

@inline Trixi.max_abs_speeds(u, equations::BoltzmannEquation) =
    (maximum(abs, view(equations.surface.velocities, :, 1)),
     maximum(abs, view(equations.surface.velocities, :, 2)))

# Zero departure is the equilibrium state used to initialize Trixi's spatial solver
@inline Trixi.initial_condition_constant(
    x,
    t,
    equations::BoltzmannEquation{N}
) where {N} =
    zero(SVector{N,eltype(x)})

"""
    streaming_fluxes!(flux_x, flux_y, model, states)

Overwrite two momentum-first matrices with `vx * phi` and `vy * phi`.
All arrays have the same shape, with one spatial state per column. Return `nothing`.
The output arrays must be distinct from one another.
"""
function streaming_fluxes!(flux_x, flux_y, equations::BoltzmannEquation, states)
    velocities = equations.surface.velocities
    size(flux_x) == size(states) == size(flux_y) ||
        throw(DimensionMismatch("state and flux arrays must have the same shape"))
    size(states, 1) == length(equations.surface) ||
        throw(DimensionMismatch("the first state axis must be momentum"))
    @inbounds for column in axes(states, 2), i in axes(states, 1)
        value = states[i, column]
        flux_x[i, column] = velocities[i, 1] * value
        flux_y[i, column] = velocities[i, 2] * value
    end
    return nothing
end
