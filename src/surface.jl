"""
    CircularFermiSurface(; p_fermi, v_fermi, mu=0, nangle=16,
                         charge=-1, spin_degeneracy=1)

One circular Fermi surface at zero temperature with midpoint angular nodes.
`momenta` and `velocities` have one two-component row per node. `energies`
equal `mu`. The unweighted state `phi` is an electrochemical energy departure.
`weights` include spin degeneracy and phase-space normalization.

Use an even `nangle` of at least four so every velocity has an opposite partner.
Keep the surface arrays unchanged after building a problem.
"""
struct CircularFermiSurface{T<:AbstractFloat}
    momenta::Matrix{T}
    velocities::Matrix{T}
    energies::Vector{T}
    weights::Vector{T}
    theta::Vector{T}
    p_fermi::T
    v_fermi::T
    mu::T
    charge::T
    spin_degeneracy::Int

    function CircularFermiSurface(; p_fermi::Real, v_fermi::Real, mu::Real = 0,
                                  nangle::Integer = 16, charge::Real = -1,
                                  spin_degeneracy::Integer = 1)
        p_fermi > 0 && isfinite(p_fermi) ||
            throw(ArgumentError("p_fermi must be finite and positive"))
        v_fermi > 0 && isfinite(v_fermi) ||
            throw(ArgumentError("v_fermi must be finite and positive"))
        nangle >= 4 && iseven(nangle) ||
            throw(ArgumentError("nangle must be even and at least four"))
        isfinite(mu) || throw(ArgumentError("mu must be finite"))
        isfinite(charge) || throw(ArgumentError("charge must be finite"))
        spin_degeneracy > 0 ||
            throw(ArgumentError("spin_degeneracy must be positive"))
        T = promote_type(Float64, typeof(p_fermi), typeof(v_fermi), typeof(mu),
                         typeof(charge))
        p, v = T(p_fermi), T(v_fermi)
        theta = T.((2pi / nangle) .* ((1:nangle) .- 0.5))
        directions = hcat(cos.(theta), sin.(theta))
        # The zero-temperature response measure is p_F / v_F per unit angle
        weight = T(spin_degeneracy * p / (2pi * v * nangle))
        isfinite(weight) && weight > 0 ||
            throw(ArgumentError("response weight must be finite and positive"))
        return new{T}(p .* directions, v .* directions, fill(T(mu), nangle),
                      fill(weight, nangle), theta, p, v, T(mu), T(charge),
                      Int(spin_degeneracy))
    end
end

Base.length(surface::CircularFermiSurface) = length(surface.weights)

"""
    circular_fermi_surface(; p_fermi, v_fermi, mu=0, nangle=16,
                           charge=-1, spin_degeneracy=1)

Build one circular Fermi surface at zero temperature, using [`CircularFermiSurface`](@ref).
The total response weight is `spin_degeneracy * p_fermi / (2pi * v_fermi)`.
Use consistent model units; we do not convert physical units automatically.
`nangle` must be even and at least four.
"""
circular_fermi_surface(; kwargs...) = CircularFermiSurface(; kwargs...)
