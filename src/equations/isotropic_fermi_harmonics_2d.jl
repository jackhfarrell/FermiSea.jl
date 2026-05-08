# This file implements the core equations object for the isotropic harmonic-moment model in
# 2D at zero temperature. We hard code a flux kernel to get a fast implementation of the
# sparse streaming matrix.

# ------------------------------------------------------------------------------------------
# Equation type and streaming flux
# ------------------------------------------------------------------------------------------

@doc raw"""
    IsotropicFermiHarmonics2D(n_harmonics; v_fermi=1.0)

Two-dimensional linear harmonic-moment model for quasiparticle streaming on an
isotropic Fermi surface.

This equations type represents a truncated angular-moment form of ballistic
transport, with propagation speed set by the Fermi velocity `v_fermi`.

The conserved state vector `u` collects angular harmonics up to a truncated order
``M = n_harmonics`` and has size  ``N = 2M + 1``. The model solves

```math
\partial_t \mathbf{u} + \partial_x\!\left(A_x\mathbf{u}\right)
+ \partial_y\!\left(A_y\mathbf{u}\right) = 0,
```

where `A_x` and `A_y` are the streaming operators induced by angular advection.
In harmonic space, streaming couples only neighboring angular modes, reflecting
that directional transport shifts phase by one harmonic index.

The harmonic basis ordering is

```math
a_0,\; a_1,\; b_1,\; a_2,\; b_2,\; \ldots,\; a_M,\; b_M,
```

`a_0` is proportional to charge density, while `a_1` and `b_1` are the first
cosine and sine harmonics and are proportional to the two components of current
density. Most observables are therefore concentrated in the lowest modes.

See also: [`build_streaming_matrices`](@ref), `Trixi.flux`.
"""
struct IsotropicFermiHarmonics2D{NVARS} <: AbstractEquations{2, NVARS}
    n_harmonics::Int
    v_fermi::Float64
    Ax::Matrix{Float64}
    Ay::Matrix{Float64}
end

# Constructor.
function IsotropicFermiHarmonics2D(n_harmonics::Integer; v_fermi::Real=1.0)
    n_harmonics = Int(n_harmonics)
    NVARS = 2 * n_harmonics + 1
    v_fermi = Float64(v_fermi)
    Ax, Ay = build_streaming_matrices(n_harmonics, v_fermi)
    return IsotropicFermiHarmonics2D{NVARS}(n_harmonics, v_fermi, Ax, Ay)
end

"""
    build_streaming_matrices(n_harmonics, v_fermi)

Build the dense streaming matrices `(Ax, Ay)` used by
[`IsotropicFermiHarmonics2D`](@ref).

The matrices act on the harmonic basis ordered as
`a0, a1, b1, a2, b2, ...` and encode the linear fluxes
`Ax * u` and `Ay * u`.
"""
function build_streaming_matrices(n_harmonics::Integer, v_fermi::Real)
    nvars = 2 * n_harmonics + 1
    v_fermi = Float64(v_fermi)
    Ax = zeros(Float64, nvars, nvars)
    Ay = zeros(Float64, nvars, nvars)
    cidx(m) = 2m
    sidx(m) = 2m + 1
    Ax[cidx(1), 1] = v_fermi
    Ay[sidx(1), 1] = v_fermi
    for m in 1:n_harmonics
        c, s = cidx(m), sidx(m)
        if m == 1
            Ax[1, c] += v_fermi / 2
            Ay[1, s] += v_fermi / 2
        else
            Ax[cidx(m-1), c] += v_fermi / 2
            Ax[sidx(m-1), s] += v_fermi / 2
            Ay[sidx(m-1), c] -= v_fermi / 2
            Ay[cidx(m-1), s] += v_fermi / 2
        end
        if m < n_harmonics
            Ax[cidx(m+1), c] += v_fermi / 2
            Ax[sidx(m+1), s] += v_fermi / 2
            Ay[sidx(m+1), c] += v_fermi / 2
            Ay[cidx(m+1), s] -= v_fermi / 2
        end
    end
    return Ax, Ay
end

# Fast hyperbolic flux implementation for isotropic harmonics, matrix free.
@inline function flux(u, 
                      normal_direction::AbstractVector,
                      equations::IsotropicFermiHarmonics2D{NVARS}) where {NVARS}
    nx  = normal_direction[1]
    ny  = normal_direction[2]
    v   = equations.v_fermi
    M   = equations.n_harmonics

    f = MVector{NVARS, eltype(u)}(undef)

    # a0
    f[1] = (v / 2) * (nx * u[2] + ny * u[3])

    # m = 1
    if M == 1
        f[2] = nx * v * u[1]
        f[3] = ny * v * u[1]
    else
        f[2] = nx * (v * u[1] + (v / 2) * u[4]) + ny * (v / 2) * u[5]
        f[3] = nx * (v / 2) * u[5] + ny * (v * u[1] - (v / 2) * u[4])

        # interior harmonics
        for m in 2:(M - 1)
            f[2m]     = nx * (v / 2) * (u[2m-2] + u[2m+2]) + ny * (v / 2) * (u[2m+3] - u[2m-1])
            f[2m + 1] = nx * (v / 2) * (u[2m-1] + u[2m+3]) + ny * (v / 2) * (u[2m-2] - u[2m+2])
        end

        # m = M  (no upper neighbour)
        f[2M]     = (v / 2) * (nx * u[2M-2] - ny * u[2M-1])
        f[2M + 1] = (v / 2) * (nx * u[2M-1] + ny * u[2M-2])
    end

    return SVector(f)
end

# Also need a method for axis-aligned fluxes in case we ever want to use a Cartesian
# grid in Trixi etc.
@inline function flux(u, orientation::Integer,
                      equations::IsotropicFermiHarmonics2D{NVARS}) where {NVARS}
    if orientation == 1
        return flux(u, SVector(1.0, 0.0), equations)
    else
        return flux(u, SVector(0.0, 1.0), equations)
    end
end

# Conservative and primitive variables are the exact same for this linear model, and entropy
# is just the L2 norm of the state vector.
@inline cons2prim(u, ::IsotropicFermiHarmonics2D)    = u
@inline cons2cons(u, ::IsotropicFermiHarmonics2D)    = u
@inline cons2entropy(u, ::IsotropicFermiHarmonics2D) = u
@inline entropy(u, ::IsotropicFermiHarmonics2D)      = dot(u, u) / 2

# For plotting and access expose the variable names.
function varnames(::Any, equations::IsotropicFermiHarmonics2D)
    names = Vector{String}(undef, nvariables(equations))
    names[1] = "a0"
    for m in 1:equations.n_harmonics
        names[2m]     = "a$m"
        names[2m + 1] = "b$m"
    end
    return Tuple(names)
end


# ------------------------------------------------------------------------------------------
# Trixi equation interface
# ------------------------------------------------------------------------------------------

# Various standard methods we have to implement to make LLF work, analysis callbacks, etc.
@inline max_abs_speeds(u, equations::IsotropicFermiHarmonics2D) =
    (equations.v_fermi, equations.v_fermi)

@inline max_abs_speeds(equations::IsotropicFermiHarmonics2D) =
    (equations.v_fermi, equations.v_fermi)

@inline have_constant_speed(::IsotropicFermiHarmonics2D) = Trixi.True()

@inline max_abs_speed_naive(u_ll, u_rr, ::Integer,
                             equations::IsotropicFermiHarmonics2D) = equations.v_fermi

@inline max_abs_speed_naive(u_ll, u_rr, normal::AbstractVector,
                             equations::IsotropicFermiHarmonics2D) =
    equations.v_fermi * norm(normal)

@inline residual_steady_state(du, ::IsotropicFermiHarmonics2D) = maximum(abs, du)

# Was having CFL issues with the default max_dt implementation for some reason so we
# override it using the known speed of the system.
@inline _transformed_speed(equations::IsotropicFermiHarmonics2D, Ja1, Ja2) =
    equations.v_fermi * sqrt(Ja1^2 + Ja2^2)

function Trixi.max_dt(u, t,
                      mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                  Trixi.T8codeMesh{2}, Trixi.StructuredMesh{2},
                                  Trixi.StructuredMeshView{2}, Trixi.UnstructuredMesh2D},
                      constant_speed::Trixi.True,
                      equations::IsotropicFermiHarmonics2D, dg::Trixi.DG, cache)
    max_scaled_speed = nextfloat(zero(t))

    contravariant_vectors = cache.elements.contravariant_vectors
    inverse_jacobian = cache.elements.inverse_jacobian

    for element in Trixi.eachelement(dg, cache)
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors,
                                                        i, j, element)
            Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors,
                                                        i, j, element)
            inv_jac = abs(inverse_jacobian[i, j, element])
            lambda1 = _transformed_speed(equations, Ja11, Ja12) * inv_jac
            lambda2 = _transformed_speed(equations, Ja21, Ja22) * inv_jac
            max_scaled_speed = Base.max(max_scaled_speed, lambda1 + lambda2)
        end
    end

    return 2 / (Trixi.nnodes(dg) * max_scaled_speed)
end

# ------------------------------------------------------------------------------------------
# Source terms
# ------------------------------------------------------------------------------------------

# Source terms outside of the streaming fluxes, which are hyperbolic. Here we implement
# collision matrices and magnetic field sources for the isotropic Fermi harmonics equations
# in 2D.

"""
    LinearCollisionMatrix(equations, W)
    LinearCollisionMatrix(equations, rates)
    LinearCollisionMatrix(equations; gamma_mr, gamma_mc, gamma_3=gamma_mc)

Linear collision source for `IsotropicFermiHarmonics2D`.  It implements a linear collision
operator `W * u` in harmonic space.

`W` may be a matrix with size `nvariables(equations) × nvariables(equations)` or a vector
of diagonal rates with length `nvariables(equations)`. For the rate-based collision model,
use the keyword constructor with `gamma_mr` for momentum relaxation, `gamma_mc` for
momentum conservation, and `gamma_3` for tomographic odd-even effects. With these rates,
the 0th harmonic is left undamped, the first harmonic is damped by `gamma_mr`, even higher
harmonics are damped by `gamma_mr + gamma_mc`, and odd harmonics `m ≥ 3` are damped by
`gamma_mr + min(gamma_3 * m^4 / 81, gamma_mc)`.
"""
struct LinearCollisionMatrix
    W::Matrix{Float64}
end

@inline function (source::LinearCollisionMatrix)(u, x, t, equations)
    return -(source.W * u)
end

function LinearCollisionMatrix(equations::IsotropicFermiHarmonics2D, W::AbstractMatrix)
    nvars = nvariables(equations)
    size(W) == (nvars, nvars) || throw(ArgumentError("W must be $(nvars)×$(nvars)"))
    return LinearCollisionMatrix(Matrix{Float64}(W))
end

function LinearCollisionMatrix(equations::IsotropicFermiHarmonics2D,
                               rates::AbstractVector)
    nvars = nvariables(equations)
    length(rates) == nvars || throw(ArgumentError("rates must have length $(nvars)"))
    W = Matrix(Diagonal(Float64.(rates)))
    return LinearCollisionMatrix(W)
end

function LinearCollisionMatrix(equations::IsotropicFermiHarmonics2D;
                               gamma_mr::Real,
                               gamma_mc::Real,
                               gamma_3::Real=gamma_mc)
    gamma_mr = Float64(gamma_mr)
    gamma_mc = Float64(gamma_mc)
    gamma_3  = Float64(gamma_3)
    gamma_mr >= 0 || throw(ArgumentError("gamma_mr must be non-negative"))
    gamma_mc >= 0 || throw(ArgumentError("gamma_mc must be non-negative"))
    gamma_3  >= 0 || throw(ArgumentError("gamma_3 must be non-negative"))
    nvars = nvariables(equations)

    rates = SVector{nvars}(map(1:nvars) do i
        m = i == 1 ? 0 : i ÷ 2
        m == 0 && return 0.0
        m == 1 && return gamma_mr
        iseven(m) && return gamma_mr + gamma_mc
        return gamma_mr + min(gamma_3 * m^4 / 81.0, gamma_mc)
    end)
    W = Matrix(Diagonal(rates))
    return LinearCollisionMatrix(W)
end

"""
    MagneticFieldSource(equations, B; charge_over_hbar=1)

Perpendicular magnetic field source for `IsotropicFermiHarmonics2D`.
The density mode stays fixed, while each harmonic pair `(a_m, b_m)` rotates at frequency
`m * charge_over_hbar * B`. In usual units where `charge_over_hbar = 1`, and `v_fermi = 1`,
this frequency is also just `m` times the inverse of the magnetic length or cyclotron radius.
"""
struct MagneticFieldSource
    n_harmonics::Int
    cyclotron_frequency::Float64
end

function MagneticFieldSource(equations::IsotropicFermiHarmonics2D, B::T;
                             charge_over_hbar::Real=one(T)) where {T<:Real}
    n_harmonics = equations.n_harmonics
    n_harmonics >= 1 || throw(ArgumentError("n_harmonics must be positive"))
    omega = Float64(charge_over_hbar * B)
    return MagneticFieldSource(Int(n_harmonics), omega)
end

@inline function (source::MagneticFieldSource)(u, x, t, equations)
    omega = source.cyclotron_frequency
    M     = source.n_harmonics
    f     = MVector{length(u), eltype(u)}(undef)
    f[1]  = zero(eltype(u))
    for m in 1:M
        mw = eltype(u)(m) * omega
        f[2m]     =  mw * u[2m + 1]
        f[2m + 1] = -mw * u[2m]
    end
    return SVector(f)
end

"""
    SourceTerms(source, sources...)

Combine one or more source terms into the callable object expected by Trixi.

Use this to bundle collision, magnetic, or other additive source terms into a
single `source_terms` object for semidiscretization. Each source is called as
`source(u, x, t, equations)` and the results are summed in order.
"""
struct SourceTerms{Sources<:Tuple}
    sources::Sources
end

SourceTerms(source, sources...) = SourceTerms((source, sources...))

@inline function (source_terms::SourceTerms)(u, x, t,
                                             equations::IsotropicFermiHarmonics2D{NVARS}) where {NVARS}
    result = zero(SVector{NVARS, eltype(u)})
    @inbounds for source in source_terms.sources
        result += source(u, x, t, equations)
    end
    return result
end

# ------------------------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------------------------

struct ProjectorCache{T, NVARS, KIN, L}
    normal::SVector{2, T}
    P_in::SMatrix{NVARS, NVARS, T, L}
    p_in_e1::SVector{NVARS, T}      
    rho_row::SVector{NVARS, T}        
end

abstract type AbstractBoundaryCondition end

mutable struct BoundaryProjectorCache
    projectors::Vector{ProjectorCache}
    projector_face_indices::Dict{Int, Int}
    projector_indices::Dict{Tuple{Int, Int}, Int}
    boundary_indices::Vector{Int}
end

BoundaryProjectorCache() =
    BoundaryProjectorCache(ProjectorCache[], Dict{Int, Int}(),
                           Dict{Tuple{Int, Int}, Int}(), Int[])

mutable struct SemidiscretizationProjectorCaches
    by_boundary_condition::IdDict{Any, BoundaryProjectorCache}
end

SemidiscretizationProjectorCaches() =
    SemidiscretizationProjectorCaches(IdDict{Any, BoundaryProjectorCache}())

const _semidiscretization_projector_caches =
    IdDict{Any, SemidiscretizationProjectorCaches}()

function _projector_caches(cache)
    return get!(_semidiscretization_projector_caches, cache) do
        SemidiscretizationProjectorCaches()
    end
end

function _boundary_projector_cache(cache, bc::AbstractBoundaryCondition)
    caches = _projector_caches(cache)
    return get!(caches.by_boundary_condition, bc) do
        BoundaryProjectorCache()
    end
end

function _boundary_projector_cache(semi::Trixi.AbstractSemidiscretization,
                                   bc::AbstractBoundaryCondition)
    _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
    return _boundary_projector_cache(cache, bc)
end

"""
    MaxwellWallBC(accommodation=1.0)

Simple wall boundary condition with Maxwell-type accommodation.  It's meant to model 
particles that hit the wall and with probability `accommodation` are re-emitted diffusely,
and with probability `1 - accommodation` are reflected specularly like from a mirror.
We also make sure there is no net flux of particles into the wall. 
"""
mutable struct MaxwellWallBC{T<:Real} <: AbstractBoundaryCondition
    accommodation::T

    function MaxwellWallBC(accommodation::Real=1.0)
        accommodation = Float64(accommodation)
        zero(accommodation) <= accommodation <= one(accommodation) ||
            throw(ArgumentError("accommodation must be in [0, 1]"))
        return new{typeof(accommodation)}(accommodation)
    end
end

@doc raw"""
    OhmicContactBC(bias)

Ohmic contact boundary condition with prescribed density-like bias.  It emits an isotropic
distribution of particles with $a_0$ set bias and the other incoming modes set to zero
"""
mutable struct OhmicContactBC{T<:Real} <: AbstractBoundaryCondition
    bias::T
    OhmicContactBC(bias::Real) = new{Float64}(Float64(bias))
end

"""
    FloatingProbeBC()

A floating contact with one shared contact potential.

This is the zero-current version of [`CurrentContactBC`](@ref). During the
boundary-flux calculation, the probe solves for a single scalar potential shared
by every boundary node belonging to this boundary condition. That shared
potential fills the incoming characteristic distribution, and is chosen so that
the integrated normal current through the whole probe is zero.
"""
mutable struct FloatingProbeBC <: AbstractBoundaryCondition end

"""
    CurrentContactBC(current)

A current-biased contact with one shared contact potential.

The value `current` is the total integrated current through the contact, using
the same surface quadrature as `ContactCurrent`. During the boundary-flux
calculation, the contact solves for a single scalar potential shared by every
boundary node belonging to this boundary condition. That shared potential fills
the incoming characteristic distribution, and is chosen so that the integrated
normal flux through the whole contact is `current`.

The sign convention follows Trixi's outward normals: positive `current` means
positive current flux in the outward normal direction.
"""
mutable struct CurrentContactBC{T<:Real} <: AbstractBoundaryCondition
    current::T
    CurrentContactBC(current::Real) = new{Float64}(Float64(current))
end

# ------------------------------------------------------------------------------------------
# Internal projector helpers
# ------------------------------------------------------------------------------------------

@inline function _apply_P_in(cache::ProjectorCache{T, NVARS, KIN, L},
                             u::SVector{NVARS, T}) where {T, NVARS, KIN, L}
    return cache.P_in * u
end

function _coeff_projector_from_orthogonal_basis(V_in::AbstractMatrix{T},
                                                gram_sqrt::AbstractVector{T}) where {T}
    nvars = length(gram_sqrt)
    P_orth = V_in * V_in'
    P = Matrix{T}(undef, nvars, nvars)
    @inbounds for j in 1:nvars, i in 1:nvars
        P[i, j] = P_orth[i, j] * gram_sqrt[j] / gram_sqrt[i]
    end
    return P
end

function _normal_matrix(n_harmonics::Integer, normal::AbstractVector{T},
                        v_fermi::T=one(T)) where {T<:Real}
    Ax, Ay = build_streaming_matrices(n_harmonics, v_fermi)
    nrm = norm(normal)
    nrm > zero(T) || throw(ArgumentError("normal must be nonzero"))
    nx = normal[1] / nrm
    ny = normal[2] / nrm
    return nx .* Ax .+ ny .* Ay
end

function _gram_sqrt_weights(nvars::Integer, ::Type{T}) where {T<:Real}
    weights = ones(T, nvars)
    weights[1] = sqrt(T(2))
    return weights
end

function build_projector_cache(n_harmonics::Integer, normal::AbstractVector{T},
                               rho_row::AbstractVector{T}) where {T<:Real}
    A     = _normal_matrix(n_harmonics, normal, one(T))
    nvars = 2 * n_harmonics + 1
    S     = _gram_sqrt_weights(nvars, T)

    A_orth = similar(A)
    @inbounds for j in 1:nvars, i in 1:nvars
        A_orth[i, j] = S[i] * A[i, j] / S[j]
    end

    F        = eigen(Symmetric(A_orth))
    incoming = F.values .< -sqrt(eps(T))
    V_in     = F.vectors[:, incoming]
    kin      = size(V_in, 2)

    # The model's symmetry guarantees a fixed incoming rank per face; if it
    # ever disagrees we fail loudly rather than silently mismatching types.
    kin == n_harmonics || throw(ErrorException(
        "unexpected incoming eigenspace dimension $kin (expected $n_harmonics) " *
        "for normal $normal; projector cache static sizing assumes symmetry."))

    e1 = zeros(T, nvars)
    e1[1] = one(T)
    P_in = _coeff_projector_from_orthogonal_basis(V_in, S)
    p_in_e1 = P_in * e1

    return ProjectorCache{T, nvars, n_harmonics, nvars * nvars}(
        SVector{2, T}(normal),
        SMatrix{nvars, nvars, T}(P_in),
        SVector{nvars, T}(p_in_e1),
        SVector{nvars, T}(rho_row),
    )
end

build_projector_cache(equations::IsotropicFermiHarmonics2D, normal, rho_row) =
    build_projector_cache(equations.n_harmonics, normal, rho_row)

@inline function solve_bc_constant(cache::ProjectorCache{T, NVARS, KIN, L},
                                   u_inner::SVector{NVARS, T};
                                   u_bc_template::SVector{NVARS, T}=zero(u_inner),
                                   target_flux::T=dot(cache.rho_row, u_inner)) where {T, NVARS, KIN, L}
    # base = P_in · u_template + P_out · u_inner = u_inner + P_in · (u_template − u_inner)
    base        = u_inner + _apply_P_in(cache, u_bc_template - u_inner)
    numerator   = target_flux - dot(cache.rho_row, base)
    denominator = dot(cache.rho_row, cache.p_in_e1)
    abs(denominator) > sqrt(eps(T)) ||
        throw(ArgumentError("boundary constant is singular for this projector"))
    C = numerator / denominator
    return base + C * cache.p_in_e1, C
end

normal_flux_row(equations::IsotropicFermiHarmonics2D, normal) =
    vec((normal[1] .* equations.Ax .+ normal[2] .* equations.Ay)[1, :])

# ------------------------------------------------------------------------------------------
# Internal boundary cache initialization
# ------------------------------------------------------------------------------------------

@inline function _normalized_normal(normal)
    return SVector(normal[1], normal[2]) / norm(normal)
end

@inline _normal_cache_tolerance(::Type{T}) where {T<:Real} = T(1024) * eps(T)

@inline function _normal_cache_key(normal::SVector{2, T}) where {T<:Real}
    tol = _normal_cache_tolerance(T)
    return ntuple(Val(2)) do i
        component = abs(normal[i]) <= tol ? zero(T) : normal[i]
        round(Int, component / tol)
    end
end

@inline function _boundary_projector_cache_for(equations, n, rho_row, bc)
    return build_projector_cache(equations, n, rho_row)
end

@inline function _projector_index!(storage::BoundaryProjectorCache, projector_by_normal,
                                   n::SVector, normal_key, equations, bc)
    return get!(projector_by_normal, normal_key) do
        rho_row = normal_flux_row(equations, n)
        push!(storage.projectors,
              _boundary_projector_cache_for(equations, n, rho_row, bc))
        length(storage.projectors)
    end
end

# Populate one BC's projector cache for a given boundary face node. Only called serially
# at setup from `initialize_boundary_projectors!`. The hot path uses
# `_lookup_projectors` which is read-only.
function _populate_projector!(storage::BoundaryProjectorCache, projector_by_normal,
                              boundary, node, normal, equations, bc)
    n = _normalized_normal(normal)
    projector_index = _projector_index!(storage, projector_by_normal, n,
                                        _normal_cache_key(n), equations, bc)
    storage.projector_indices[(boundary, node)] = projector_index
    return nothing
end

# Read-only lookup used inside the threaded RHS. In normal operation, the cache
# was fully populated serially by the auto-init hook on `Trixi.semidiscretize`,
# so this is a pure read keyed by the boundary face node.
@inline function _lookup_projectors(storage::BoundaryProjectorCache, boundary, node, normal,
                                    equations)
    projector_index = get(storage.projector_face_indices, boundary, nothing)
    projector_index === nothing || return storage.projectors[projector_index]

    key = (boundary, node)
    projector_index = get(storage.projector_indices, key, nothing)
    projector_index === nothing || return storage.projectors[projector_index]
    return _build_uncached_projectors(normal, equations)
end

@inline function _lookup_projectors(cache, bc::AbstractBoundaryCondition, boundary, node,
                                    normal, equations)
    storage = _boundary_projector_cache(cache, bc)
    return _lookup_projectors(storage, boundary, node, normal, equations)
end

# Direct/manual boundary-condition calls do not know their boundary face node, so
# keep them correct without touching the threaded cache.
@inline function _build_uncached_projectors(normal, equations)
    n = _normalized_normal(normal)
    rho_row = normal_flux_row(equations, n)
    return build_projector_cache(equations, n, rho_row)
end

"""
    initialize_boundary_projectors!(semi)

Serially walk every boundary face node of `semi`, compute each node's outward
normal, and pre-populate the projector cache on the matching boundary
condition. After this returns, the boundary flux hot path only reads from the
caches, which is safe under Trixi's Polyester-threaded surface-flux kernel.

Call this once, before `solve(...)`.
"""
function initialize_boundary_projectors!(semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    _initialize_boundary_projectors!(semi.boundary_conditions, mesh, equations,
                                     solver, cache)
    return semi
end

function _initialize_boundary_projectors!(bcs::Trixi.UnstructuredSortedBoundaryTypes,
                                          mesh::Trixi.P4estMesh{2}, equations, solver,
                                          cache)
    boundaries            = cache.boundaries
    contravariant_vectors = cache.elements.contravariant_vectors
    index_range           = eachnode(solver)
    projector_caches      = _projector_caches(cache)
    empty!(projector_caches.by_boundary_condition)

    for bc in values(bcs.boundary_conditions)
        bc isa AbstractBoundaryCondition || continue
        projector_caches.by_boundary_condition[bc] = BoundaryProjectorCache()
    end

    for (name, bc) in pairs(bcs.boundary_conditions)
        bc isa AbstractBoundaryCondition || continue
        storage = _boundary_projector_cache(cache, bc)
        projector_by_normal = Dict{NTuple{2, Int}, Int}()
        indices = get(bcs.boundary_symbol_indices, name, Int[])
        bc isa Union{CurrentContactBC, FloatingProbeBC} &&
            append!(storage.boundary_indices, indices)
        for boundary in indices
            element      = boundaries.neighbor_ids[boundary]
            node_indices = boundaries.node_indices[boundary]
            direction    = Trixi.indices2direction(node_indices)

            i_start, i_step = Trixi.index_to_start_step_2d(node_indices[1], index_range)
            j_start, j_step = Trixi.index_to_start_step_2d(node_indices[2], index_range)

            i_node = i_start
            j_node = j_start
            face_projector = true
            face_normal = nothing
            face_normal_key = nothing
            face_nodes = Int[]
            for node in index_range
                normal = Trixi.get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)
                n = _normalized_normal(normal)
                normal_key = _normal_cache_key(n)
                push!(face_nodes, node)

                if face_normal_key === nothing
                    face_normal = n
                    face_normal_key = normal_key
                else
                    face_projector &= normal_key == face_normal_key
                end

                i_node += i_step
                j_node += j_step
            end

            if face_projector
                storage.projector_face_indices[boundary] = _projector_index!(
                    storage, projector_by_normal, face_normal, face_normal_key,
                    equations, bc)
            else
                i_node = i_start
                j_node = j_start
                for node in face_nodes
                    normal = Trixi.get_normal_direction(direction, contravariant_vectors,
                                                        i_node, j_node, element)
                    _populate_projector!(storage, projector_by_normal, boundary, node,
                                         normal, equations, bc)
                    i_node += i_step
                    j_node += j_step
                end
            end
        end
    end
    return nothing
end

# Fallback: if a user passes a raw NamedTuple instead of the sorted container,
# or a non-P4est mesh, do nothing. Direct boundary calls still build projectors
# from the supplied normal.
_initialize_boundary_projectors!(::Any, ::Any, ::Any, ::Any, ::Any) = nothing

# Auto-init hook: every call to `Trixi.semidiscretize` on a semi whose equations
# are `IsotropicFermiHarmonics2D` pre-populates the BC projector caches before
# delegating to the default implementation. Required for thread-safe surface
# fluxes under Trixi's Polyester backend.
function Trixi.semidiscretize(semi::Trixi.SemidiscretizationHyperbolic{<:Any,
                                  <:IsotropicFermiHarmonics2D}, tspan; kwargs...)
    initialize_boundary_projectors!(semi)
    return invoke(Trixi.semidiscretize,
                  Tuple{Trixi.AbstractSemidiscretization, typeof(tspan)},
                  semi, tspan; kwargs...)
end

# ------------------------------------------------------------------------------------------
# Internal boundary state templates
# ------------------------------------------------------------------------------------------

_template(bc::AbstractBoundaryCondition, u_inner, normal, equations) =
    _template(bc, u_inner, equations)

@inline function _specular_reflection_template(u_inner, normal,
                                               equations::IsotropicFermiHarmonics2D)
    T = eltype(u_inner)
    M = equations.n_harmonics

    nrm = norm(normal)
    nx = normal[1] / nrm
    ny = normal[2] / nrm

    # Reflection angle is A = 2*theta_n + pi where n = (cos(theta_n), sin(theta_n)).
    # Use trig recurrence for cos(mA), sin(mA) without calling atan for each m.
    cA = ny^2 - nx^2
    sA = -2 * nx * ny

    reflected = MVector{length(u_inner), T}(undef)
    reflected[1] = u_inner[1]

    c_m = cA
    s_m = sA
    for m in 1:M
        cidx = 2m
        sidx = 2m + 1
        a_m = u_inner[cidx]
        b_m = u_inner[sidx]

        reflected[cidx] = a_m * c_m + b_m * s_m
        reflected[sidx] = a_m * s_m - b_m * c_m

        if m < M
            c_next = c_m * cA - s_m * sA
            s_next = s_m * cA + c_m * sA
            c_m, s_m = c_next, s_next
        end
    end

    return SVector(reflected)
end

@inline function _diffuse_isotropic_template(u_inner)
    T = eltype(u_inner)
    return SVector{length(u_inner), T}(ntuple(i -> i == 1 ? u_inner[1] : zero(T),
                                              Val(length(u_inner))))
end

@inline function _template(bc::MaxwellWallBC, u_inner, normal,
                           equations::IsotropicFermiHarmonics2D)
    α = convert(eltype(u_inner), bc.accommodation)
    specular = _specular_reflection_template(u_inner, normal, equations)
    diffuse = _diffuse_isotropic_template(u_inner)
    return (one(α) - α) * specular + α * diffuse
end

function _template(::MaxwellWallBC, u_inner, equations)
    return _diffuse_isotropic_template(u_inner)
end

function _template(bc::OhmicContactBC, u_inner, equations)
    return SVector{length(u_inner)}(ntuple(i -> i == 1 ? bc.bias : zero(eltype(u_inner)),
                                          Val(length(u_inner))))
end

function _template(::FloatingProbeBC, u_inner, equations)
    return zero(u_inner)
end

_target_flux(::MaxwellWallBC, u_inner, normal, equations) = zero(eltype(u_inner))

@inline function _current_contact_base_state(cache::ProjectorCache, u_inner)
    return u_inner + _apply_P_in(cache, zero(u_inner) - u_inner)
end

@inline function _current_contact_state(cache::ProjectorCache, u_inner, potential)
    return _current_contact_base_state(cache, u_inner) +
           convert(eltype(u_inner), potential) * cache.p_in_e1
end

function assemble_ghost_state(bc::AbstractBoundaryCondition, u_inner, normal, equations)
    cache       = _build_uncached_projectors(normal, equations)
    template    = _template(bc, u_inner, normal, equations)
    target_flux = _target_flux(bc, u_inner, normal, equations)
    u_boundary, _ = solve_bc_constant(cache, u_inner;
                                      u_bc_template=template,
                                      target_flux=target_flux)
    return u_boundary
end

function assemble_ghost_state(bc::OhmicContactBC, u_inner, normal, equations)
    cache = _build_uncached_projectors(normal, equations)
    template = _template(bc, u_inner, normal, equations)
    return u_inner + _apply_P_in(cache, template - u_inner)
end

function assemble_ghost_state(bc::CurrentContactBC, u_inner, normal, equations)
    throw(ArgumentError("CurrentContactBC enforces a total current over a full " *
                        "boundary, so assemble_ghost_state requires a semidiscretization " *
                        "boundary-flux context. Use OhmicContactBC for a prescribed local " *
                        "potential template."))
end

function assemble_ghost_state(bc::FloatingProbeBC, u_inner, normal, equations)
    throw(ArgumentError("FloatingProbeBC enforces zero total current over a full " *
                        "boundary, so assemble_ghost_state requires a semidiscretization " *
                        "boundary-flux context. Use OhmicContactBC for a prescribed local " *
                        "potential template."))
end

# ------------------------------------------------------------------------------------------
# Trixi boundary-flux integration
# ------------------------------------------------------------------------------------------

# Exact characteristic boundary flux. Bypasses the numerical surface flux entirely:
# reconstruct the boundary state from outgoing characteristics of u_inner and incoming
# characteristics of the boundary template, then apply the physical flux.
@inline function _exact_boundary_flux(u_inner, normal, bc, equations)
    u_boundary = assemble_ghost_state(bc, u_inner, normal, equations)
    return flux(u_boundary, normal, equations)
end

@inline function (bc::AbstractBoundaryCondition)(u_inner, normal_direction::AbstractVector,
                                                 x, t, surface_flux_function,
                                                 equations::IsotropicFermiHarmonics2D)
    return _exact_boundary_flux(u_inner, normal_direction, bc, equations)
end

@inline function (bc::AbstractBoundaryCondition)(u_inner, orientation_or_normal,
                                                 direction::Integer, x, t,
                                                 surface_flux_function,
                                                 equations::IsotropicFermiHarmonics2D)
    normal = orientation_or_normal isa Integer ?
             (orientation_or_normal == 1 ?
              SVector(isodd(direction) ? -one(eltype(u_inner)) : one(eltype(u_inner)),
                      zero(eltype(u_inner))) :
              SVector(zero(eltype(u_inner)),
                      isodd(direction) ? -one(eltype(u_inner)) : one(eltype(u_inner)))) :
             orientation_or_normal
    return _exact_boundary_flux(u_inner, normal, bc, equations)
end

function assemble_ghost_state(cache::ProjectorCache, bc::AbstractBoundaryCondition,
                              u_inner, normal, equations)
    template    = _template(bc, u_inner, normal, equations)
    target_flux = _target_flux(bc, u_inner, normal, equations)
    u_boundary, _ = solve_bc_constant(cache, u_inner;
                                      u_bc_template=template,
                                      target_flux=target_flux)
    return u_boundary
end

function assemble_ghost_state(cache::ProjectorCache, bc::OhmicContactBC,
                              u_inner, normal, equations)
    template = _template(bc, u_inner, normal, equations)
    return u_inner + _apply_P_in(cache, template - u_inner)
end

_total_current_target(bc::CurrentContactBC) = bc.current
_total_current_target(::FloatingProbeBC) = 0.0

function _current_contact_potential(bc::Union{CurrentContactBC, FloatingProbeBC},
                                    equations, dg, cache)
    storage = _boundary_projector_cache(cache, bc)
    isempty(storage.boundary_indices) &&
        throw(ArgumentError("contact boundary condition has no initialized boundary indices. " *
                            "Call initialize_boundary_projectors!(semi) before solve(...)."))

    boundaries = cache.boundaries
    contravariant_vectors = cache.elements.contravariant_vectors
    weights = dg.basis.weights
    index_range = eachnode(dg)

    base_current = zero(eltype(boundaries.u))
    potential_response = zero(eltype(boundaries.u))

    for boundary in storage.boundary_indices
        element      = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction    = Trixi.indices2direction(node_indices)

        i_start, i_step = Trixi.index_to_start_step_2d(node_indices[1], index_range)
        j_start, j_step = Trixi.index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_start
        j_node = j_start
        for node in index_range
            normal_direction = Trixi.get_normal_direction(direction,
                                                          contravariant_vectors,
                                                          i_node, j_node, element)
            projector = _lookup_projectors(storage, boundary, node, normal_direction,
                                           equations)
            u_inner = Trixi.get_node_vars(boundaries.u, equations, dg, node, boundary)
            base_state = _current_contact_base_state(projector, u_inner)
            rho_row = normal_flux_row(equations, normal_direction)

            weight = weights[node]
            base_current += weight * dot(rho_row, base_state)
            potential_response += weight * dot(rho_row, projector.p_in_e1)

            i_node += i_step
            j_node += j_step
        end
    end

    abs(potential_response) > sqrt(eps(real(potential_response))) ||
        throw(ArgumentError("CurrentContactBC has singular contact-current response"))

    return (convert(typeof(base_current), _total_current_target(bc)) - base_current) /
           potential_response
end

function assemble_ghost_state(cache::ProjectorCache,
                              bc::Union{CurrentContactBC, FloatingProbeBC},
                              u_inner, normal, equations, potential)
    return _current_contact_state(cache, u_inner, potential)
end

@inline function _exact_boundary_flux(cache::ProjectorCache, u_inner, normal, bc,
                                      equations)
    u_boundary = assemble_ghost_state(cache, bc, u_inner, normal, equations)
    return flux(u_boundary, normal, equations)
end

@inline function _exact_boundary_flux_at_node(boundary, node, u_inner, normal, bc,
                                              equations)
    cache = _build_uncached_projectors(normal, equations)
    return _exact_boundary_flux(cache, u_inner, normal, bc, equations)
end

@inline function _exact_boundary_flux_at_node(boundary, node, u_inner, normal, bc,
                                              equations, cache_data)
    cache = _lookup_projectors(cache_data, bc, boundary, node, normal, equations)
    return _exact_boundary_flux(cache, u_inner, normal, bc, equations)
end

@inline function _exact_boundary_flux_at_node(boundary, node, u_inner, normal,
                                              bc::Union{CurrentContactBC, FloatingProbeBC},
                                              equations, dg, cache_data)
    cache = _lookup_projectors(cache_data, bc, boundary, node, normal, equations)
    potential = _current_contact_potential(bc, equations, dg, cache_data)
    u_boundary = assemble_ghost_state(cache, bc, u_inner, normal, equations,
                                      potential)
    return flux(u_boundary, normal, equations)
end

function Trixi.calc_boundary_flux!(surface_flux_values, t,
                                   boundary_condition::AbstractBoundaryCondition,
                                   mesh::Trixi.P4estMesh{2},
                                   have_nonconservative_terms::Trixi.False,
                                   equations::IsotropicFermiHarmonics2D,
                                   surface_integral, dg::Trixi.DG, cache,
                                   i_index, j_index,
                                   node_index, direction_index, element_index,
                                   boundary_index)
    boundaries = cache.boundaries
    contravariant_vectors = cache.elements.contravariant_vectors

    u_inner = Trixi.get_node_vars(boundaries.u, equations, dg, node_index,
                                  boundary_index)
    normal_direction = Trixi.get_normal_direction(direction_index,
                                                  contravariant_vectors,
                                                  i_index, j_index,
                                                  element_index)
    flux_ = if boundary_condition isa Union{CurrentContactBC, FloatingProbeBC}
        _exact_boundary_flux_at_node(boundary_index, node_index, u_inner,
                                     normal_direction, boundary_condition,
                                     equations, dg, cache)
    else
        _exact_boundary_flux_at_node(boundary_index, node_index, u_inner,
                                     normal_direction, boundary_condition,
                                     equations, cache)
    end

    for v in Trixi.eachvariable(equations)
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
    end

    return nothing
end
