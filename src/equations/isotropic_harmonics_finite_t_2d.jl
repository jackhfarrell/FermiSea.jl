# This file implements a finite-temperature linear-response harmonic-moment model in
# 2D for a parabolic band. It extends the angular-only zero-temperature model by adding
# an orthonormal radial basis in each angular sector.

# ------------------------------------------------------------------------------------------
# Equation type and constructor
# ------------------------------------------------------------------------------------------

@doc raw"""
    IsotropicHarmonicsFiniteT2D(n_harmonics, n_radial;
        mass=1.0, mu0=1.0, temperature=0.05, zmax=20.0, n_quad=256,
        conserve_energy=false)

Two-dimensional finite-temperature, linear-response harmonic-moment model for a
parabolic band ``epsilon = p^2/(2m)``.

The perturbation is expanded in a real harmonic basis with radial functions,

```math
\delta f = -\partial_\epsilon f_0
           \sum_{\ell,a} c_{\ell a} B_{\ell a}(\epsilon)
           \{1, \cos(\ell\theta), \sin(\ell\theta)\}.
```

For each angular sector, radial seed functions ``p^\ell z^n`` are orthonormalized
numerically with the finite-temperature radial weight. The state ordering is

```math
a0_r0, a0_r1, \ldots, a1_r0, \ldots, b1_r0, \ldots, a2_r0, b2_r0, \ldots
```

This first implementation is linear-response only. A future nonlinear local
equilibrium/BGK target can be obtained by projecting the isothermal expansion
``Psi - tanh((epsilon - mu0)/(2T)) Psi^2 / (2T) + ...`` into the same basis.
"""
struct IsotropicHarmonicsFiniteT2D{NVARS} <: AbstractEquations{2, NVARS}
    n_harmonics::Int
    n_radial::Int
    mass::Float64
    mu0::Float64
    temperature::Float64
    zmax::Float64
    n_quad::Int
    conserve_energy::Bool
    z_nodes::Vector{Float64}
    eps_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
    fermi_window::Vector{Float64}
    radial_basis::Vector{Matrix{Float64}}
    radial_grams::Vector{Matrix{Float64}}
    Ax::Matrix{Float64}
    Ay::Matrix{Float64}
    moment_matrix::Matrix{Float64}
    hydro_embedding::Matrix{Float64}
    hydro_projector::Matrix{Float64}
    density_projector::Matrix{Float64}
    gram_sqrt::Vector{Float64}
    vmax::Float64
end

function IsotropicHarmonicsFiniteT2D(n_harmonics::Integer, n_radial::Integer;
                                     mass::Real=1.0,
                                     mu0::Real=1.0,
                                     temperature::Real=0.05,
                                     zmax::Real=20.0,
                                     n_quad::Integer=256,
                                     conserve_energy::Bool=false)
    n_harmonics = Int(n_harmonics)
    n_radial = Int(n_radial)
    n_quad = Int(n_quad)
    mass = Float64(mass)
    mu0 = Float64(mu0)
    temperature = Float64(temperature)
    zmax = Float64(zmax)

    n_harmonics >= 1 || throw(ArgumentError("n_harmonics must be at least 1"))
    n_radial >= 1 || throw(ArgumentError("n_radial must be at least 1"))
    mass > 0 || throw(ArgumentError("mass must be positive"))
    mu0 > 0 || throw(ArgumentError("mu0 must be positive"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    zmax > 0 || throw(ArgumentError("zmax must be positive"))
    n_quad >= 32 || throw(ArgumentError("n_quad must be at least 32"))
    if conserve_energy && n_radial < 2
        throw(ArgumentError("conserve_energy=true requires n_radial >= 2"))
    end

    NVARS = n_radial * (1 + 2 * n_harmonics)
    z_nodes, eps_nodes, quad_weights, fermi_window =
        _finite_t_quadrature(mu0, temperature, zmax, n_quad)
    p_nodes = sqrt.(2 .* mass .* eps_nodes)
    radial_basis, radial_grams =
        _finite_t_radial_bases(n_harmonics, n_radial, p_nodes, z_nodes, quad_weights)
    gram_sqrt = _finite_t_gram_sqrt(n_harmonics, n_radial)
    Ax, Ay = _finite_t_streaming_matrices(n_harmonics, n_radial, mass, p_nodes,
                                          quad_weights, radial_basis)
    moment_matrix, hydro_embedding, hydro_projector, density_projector =
        _finite_t_moment_data(n_harmonics, n_radial, eps_nodes, p_nodes, quad_weights,
                              radial_basis, conserve_energy)
    vmax = sqrt(2 * maximum(eps_nodes) / mass)

    return IsotropicHarmonicsFiniteT2D{NVARS}(
        n_harmonics, n_radial, mass, mu0, temperature, zmax, n_quad,
        conserve_energy, z_nodes, eps_nodes, quad_weights, fermi_window,
        radial_basis, radial_grams, Ax, Ay, moment_matrix, hydro_embedding,
        hydro_projector, density_projector, gram_sqrt, vmax)
end

# ------------------------------------------------------------------------------------------
# Basis indexing and quadrature
# ------------------------------------------------------------------------------------------

@inline _finite_scalar_index(n_radial::Integer, a::Integer) = Int(a)

@inline function _finite_cos_index(n_radial::Integer, ell::Integer, a::Integer)
    return n_radial + 2 * (Int(ell) - 1) * n_radial + Int(a)
end

@inline function _finite_sin_index(n_radial::Integer, ell::Integer, a::Integer)
    return n_radial + (2 * (Int(ell) - 1) + 1) * n_radial + Int(a)
end

@inline _finite_scalar_index(equations::IsotropicHarmonicsFiniteT2D, a::Integer) =
    _finite_scalar_index(equations.n_radial, a)

@inline _finite_cos_index(equations::IsotropicHarmonicsFiniteT2D, ell::Integer,
                          a::Integer) =
    _finite_cos_index(equations.n_radial, ell, a)

@inline _finite_sin_index(equations::IsotropicHarmonicsFiniteT2D, ell::Integer,
                          a::Integer) =
    _finite_sin_index(equations.n_radial, ell, a)

function _finite_mode(index::Integer, n_radial::Integer)
    index = Int(index)
    n_radial = Int(n_radial)
    if index <= n_radial
        return 0, :cos, index
    end
    shifted = index - n_radial - 1
    block = shifted ÷ n_radial
    a = shifted % n_radial + 1
    ell = block ÷ 2 + 1
    kind = iseven(block) ? :cos : :sin
    return ell, kind, a
end

function _finite_t_gausslegendre(n::Integer)
    n = Int(n)
    n >= 1 || throw(ArgumentError("number of quadrature nodes must be positive"))
    diagonal = zeros(Float64, n)
    offdiag = [k / sqrt(4 * k^2 - 1) for k in 1:(n - 1)]
    F = eigen(SymTridiagonal(diagonal, offdiag))
    nodes = F.values
    weights = 2 .* abs2.(F.vectors[1, :])
    return nodes, weights
end

function _finite_t_quadrature(mu0, temperature, zmax, n_quad)
    z_left = max(-zmax, -mu0 / temperature)
    z_right = zmax
    z_left < z_right || throw(ArgumentError("empty finite-temperature quadrature window"))

    nodes, weights = _finite_t_gausslegendre(n_quad)
    center = (z_right + z_left) / 2
    half_width = (z_right - z_left) / 2
    z_nodes = center .+ half_width .* nodes
    z_weights = half_width .* weights
    eps_nodes = mu0 .+ temperature .* z_nodes

    # w(epsilon) = -df0/depsilon. The radial quadrature weights include d epsilon.
    fermi_window = @. inv(4 * temperature * cosh(z_nodes / 2)^2)
    quad_weights = @. z_weights * temperature * fermi_window
    return z_nodes, eps_nodes, quad_weights, fermi_window
end

function _finite_weighted_dot(weights, a, b)
    result = 0.0
    @inbounds for q in eachindex(weights)
        result += weights[q] * a[q] * b[q]
    end
    return result
end

function _finite_t_radial_bases(n_harmonics, n_radial, p_nodes, z_nodes, weights)
    radial_basis = Vector{Matrix{Float64}}(undef, n_harmonics + 1)
    radial_grams = Vector{Matrix{Float64}}(undef, n_harmonics + 1)
    n_quad = length(z_nodes)

    for ell in 0:n_harmonics
        basis = zeros(Float64, n_radial, n_quad)
        for a in 1:n_radial
            v = @. p_nodes^ell * z_nodes^(a - 1)
            for j in 1:(a - 1)
                coeff = _finite_weighted_dot(weights, view(basis, j, :), v)
                @inbounds for q in 1:n_quad
                    v[q] -= coeff * basis[j, q]
                end
            end
            norm_v = sqrt(_finite_weighted_dot(weights, v, v))
            if !(norm_v > 100 * eps(Float64))
                throw(ArgumentError("radial seed basis is numerically dependent; " *
                                    "increase n_quad or reduce n_radial"))
            end
            @inbounds for q in 1:n_quad
                basis[a, q] = v[q] / norm_v
            end
        end

        gram = zeros(Float64, n_radial, n_radial)
        for b in 1:n_radial, a in 1:n_radial
            gram[a, b] = _finite_weighted_dot(weights, view(basis, a, :),
                                              view(basis, b, :))
        end
        radial_basis[ell + 1] = basis
        radial_grams[ell + 1] = gram
    end

    return radial_basis, radial_grams
end

function _finite_t_gram_sqrt(n_harmonics, n_radial)
    nvars = n_radial * (1 + 2 * n_harmonics)
    gram_sqrt = ones(Float64, nvars)
    @inbounds for a in 1:n_radial
        gram_sqrt[_finite_scalar_index(n_radial, a)] = sqrt(2.0)
    end
    return gram_sqrt
end

# ------------------------------------------------------------------------------------------
# Streaming and moments
# ------------------------------------------------------------------------------------------

function _finite_t_radial_velocity_matrix(basis_out, basis_in, p_over_m, weights)
    n_out = size(basis_out, 1)
    n_in = size(basis_in, 1)
    V = zeros(Float64, n_out, n_in)
    @inbounds for b in 1:n_in, a in 1:n_out
        value = 0.0
        for q in eachindex(weights)
            value += weights[q] * basis_out[a, q] * p_over_m[q] * basis_in[b, q]
        end
        V[a, b] = value
    end
    return V
end

function _finite_add_block!(A, rows, cols, block, factor)
    factor == 0 && return nothing
    @inbounds for b in eachindex(cols), a in eachindex(rows)
        A[rows[a], cols[b]] += factor * block[a, b]
    end
    return nothing
end

function _finite_sector_indices(n_radial, ell, kind)
    if ell == 0
        return [_finite_scalar_index(n_radial, a) for a in 1:n_radial]
    elseif kind === :cos
        return [_finite_cos_index(n_radial, ell, a) for a in 1:n_radial]
    else
        return [_finite_sin_index(n_radial, ell, a) for a in 1:n_radial]
    end
end

function _finite_t_streaming_matrices(n_harmonics, n_radial, mass, p_nodes, weights,
                                      radial_basis)
    nvars = n_radial * (1 + 2 * n_harmonics)
    Ax = zeros(Float64, nvars, nvars)
    Ay = zeros(Float64, nvars, nvars)
    p_over_m = p_nodes ./ mass

    radial_velocity = [begin
        _finite_t_radial_velocity_matrix(radial_basis[ell_out + 1],
                                         radial_basis[ell_in + 1],
                                         p_over_m, weights)
    end for ell_out in 0:n_harmonics, ell_in in 0:n_harmonics]

    if n_harmonics >= 1
        cols0 = _finite_sector_indices(n_radial, 0, :cos)
        rows_c1 = _finite_sector_indices(n_radial, 1, :cos)
        rows_s1 = _finite_sector_indices(n_radial, 1, :sin)
        _finite_add_block!(Ax, rows_c1, cols0, radial_velocity[2, 1], 1.0)
        _finite_add_block!(Ay, rows_s1, cols0, radial_velocity[2, 1], 1.0)
    end

    for ell in 1:n_harmonics
        cols_c = _finite_sector_indices(n_radial, ell, :cos)
        cols_s = _finite_sector_indices(n_radial, ell, :sin)

        ell_down = ell - 1
        if ell_down == 0
            rows0 = _finite_sector_indices(n_radial, 0, :cos)
            block = radial_velocity[1, ell + 1]
            _finite_add_block!(Ax, rows0, cols_c, block, 0.5)
            _finite_add_block!(Ay, rows0, cols_s, block, 0.5)
        elseif ell_down >= 1
            rows_cd = _finite_sector_indices(n_radial, ell_down, :cos)
            rows_sd = _finite_sector_indices(n_radial, ell_down, :sin)
            block = radial_velocity[ell_down + 1, ell + 1]
            _finite_add_block!(Ax, rows_cd, cols_c, block, 0.5)
            _finite_add_block!(Ax, rows_sd, cols_s, block, 0.5)
            _finite_add_block!(Ay, rows_sd, cols_c, block, -0.5)
            _finite_add_block!(Ay, rows_cd, cols_s, block, 0.5)
        end

        ell_up = ell + 1
        if ell_up <= n_harmonics
            rows_cu = _finite_sector_indices(n_radial, ell_up, :cos)
            rows_su = _finite_sector_indices(n_radial, ell_up, :sin)
            block = radial_velocity[ell_up + 1, ell + 1]
            _finite_add_block!(Ax, rows_cu, cols_c, block, 0.5)
            _finite_add_block!(Ax, rows_su, cols_s, block, 0.5)
            _finite_add_block!(Ay, rows_su, cols_c, block, 0.5)
            _finite_add_block!(Ay, rows_cu, cols_s, block, -0.5)
        end
    end

    return Ax, Ay
end

function _finite_project_radial(basis, weights, values)
    coeffs = zeros(Float64, size(basis, 1))
    @inbounds for a in axes(basis, 1)
        value = 0.0
        for q in eachindex(weights)
            value += weights[q] * basis[a, q] * values[q]
        end
        coeffs[a] = value
    end
    return coeffs
end

function _finite_t_moment_data(n_harmonics, n_radial, eps_nodes, p_nodes, weights,
                               radial_basis, conserve_energy)
    nvars = n_radial * (1 + 2 * n_harmonics)
    n_moments = conserve_energy ? 4 : 3
    moment_matrix = zeros(Float64, n_moments, nvars)

    density_coeffs = _finite_project_radial(radial_basis[1], weights,
                                            ones(Float64, length(weights)))
    px_coeffs = _finite_project_radial(radial_basis[2], weights, p_nodes)
    py_coeffs = px_coeffs
    energy_coeffs = _finite_project_radial(radial_basis[1], weights, eps_nodes)

    @inbounds for a in 1:n_radial
        moment_matrix[1, _finite_scalar_index(n_radial, a)] = 2 * density_coeffs[a]
        moment_matrix[2, _finite_cos_index(n_radial, 1, a)] = px_coeffs[a]
        moment_matrix[3, _finite_sin_index(n_radial, 1, a)] = py_coeffs[a]
        if conserve_energy
            moment_matrix[4, _finite_scalar_index(n_radial, a)] = 2 * energy_coeffs[a]
        end
    end

    hydro_embedding = zeros(Float64, nvars, n_moments)
    @inbounds for a in 1:n_radial
        hydro_embedding[_finite_scalar_index(n_radial, a), 1] = density_coeffs[a]
        hydro_embedding[_finite_cos_index(n_radial, 1, a), 2] = px_coeffs[a]
        hydro_embedding[_finite_sin_index(n_radial, 1, a), 3] = py_coeffs[a]
        if conserve_energy
            hydro_embedding[_finite_scalar_index(n_radial, a), 4] = energy_coeffs[a]
        end
    end

    hydro_projector = hydro_embedding *
                      ((moment_matrix * hydro_embedding) \ moment_matrix)

    density_moment = reshape(moment_matrix[1, :], 1, :)
    density_embedding = reshape(hydro_embedding[:, 1], :, 1)
    density_projector = density_embedding *
                        ((density_moment * density_embedding) \ density_moment)

    return moment_matrix, hydro_embedding, hydro_projector, density_projector
end

# ------------------------------------------------------------------------------------------
# Trixi equation interface
# ------------------------------------------------------------------------------------------

function _finite_matrix_times_vector(A, u, ::Val{NVARS}) where {NVARS}
    T = eltype(u)
    result = MVector{NVARS, T}(undef)
    @inbounds for i in 1:NVARS
        value = zero(T)
        for j in 1:NVARS
            value += convert(T, A[i, j]) * u[j]
        end
        result[i] = value
    end
    return SVector(result)
end

@inline function flux(u, orientation::Integer,
                      equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    if orientation == 1
        return _finite_matrix_times_vector(equations.Ax, u, Val(NVARS))
    elseif orientation == 2
        return _finite_matrix_times_vector(equations.Ay, u, Val(NVARS))
    else
        throw(ArgumentError("orientation must be 1 or 2"))
    end
end

@inline function flux(u, normal_direction::AbstractVector,
                      equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    T = eltype(u)
    result = MVector{NVARS, T}(undef)
    nx = convert(T, normal_direction[1])
    ny = convert(T, normal_direction[2])
    @inbounds for i in 1:NVARS
        value = zero(T)
        for j in 1:NVARS
            Aij = nx * convert(T, equations.Ax[i, j]) +
                  ny * convert(T, equations.Ay[i, j])
            value += Aij * u[j]
        end
        result[i] = value
    end
    return SVector(result)
end

@inline cons2prim(u, ::IsotropicHarmonicsFiniteT2D) = u
@inline cons2cons(u, ::IsotropicHarmonicsFiniteT2D) = u
@inline cons2entropy(u, ::IsotropicHarmonicsFiniteT2D) = u
@inline entropy(u, equations::IsotropicHarmonicsFiniteT2D) =
    dot(equations.gram_sqrt .* u, equations.gram_sqrt .* u) / 2

function varnames(::Any, equations::IsotropicHarmonicsFiniteT2D)
    names = Vector{String}(undef, nvariables(equations))
    n_radial = equations.n_radial
    for a in 1:n_radial
        names[_finite_scalar_index(n_radial, a)] = "a0_r$(a - 1)"
    end
    for ell in 1:equations.n_harmonics
        for a in 1:n_radial
            names[_finite_cos_index(n_radial, ell, a)] = "a$(ell)_r$(a - 1)"
            names[_finite_sin_index(n_radial, ell, a)] = "b$(ell)_r$(a - 1)"
        end
    end
    return Tuple(names)
end

@inline max_abs_speeds(u, equations::IsotropicHarmonicsFiniteT2D) =
    (equations.vmax, equations.vmax)

@inline max_abs_speeds(equations::IsotropicHarmonicsFiniteT2D) =
    (equations.vmax, equations.vmax)

@inline have_constant_speed(::IsotropicHarmonicsFiniteT2D) = Trixi.True()

@inline max_abs_speed_naive(u_ll, u_rr, ::Integer,
                            equations::IsotropicHarmonicsFiniteT2D) = equations.vmax

@inline max_abs_speed_naive(u_ll, u_rr, normal::AbstractVector,
                            equations::IsotropicHarmonicsFiniteT2D) =
    equations.vmax * norm(normal)

@inline residual_steady_state(du, ::IsotropicHarmonicsFiniteT2D) = maximum(abs, du)

@inline _transformed_speed(equations::IsotropicHarmonicsFiniteT2D, Ja1, Ja2) =
    equations.vmax * sqrt(Ja1^2 + Ja2^2)

function Trixi.max_dt(u, t,
                      mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                  Trixi.T8codeMesh{2}, Trixi.StructuredMesh{2},
                                  Trixi.StructuredMeshView{2}, Trixi.UnstructuredMesh2D},
                      constant_speed::Trixi.True,
                      equations::IsotropicHarmonicsFiniteT2D, dg::Trixi.DG, cache)
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

function LinearCollisionMatrix(equations::IsotropicHarmonicsFiniteT2D, W::AbstractMatrix)
    nvars = nvariables(equations)
    size(W) == (nvars, nvars) || throw(ArgumentError("W must be $(nvars)x$(nvars)"))
    return LinearCollisionMatrix(Matrix{Float64}(W))
end

function LinearCollisionMatrix(equations::IsotropicHarmonicsFiniteT2D;
                               gamma_mr::Real,
                               gamma_mc::Real)
    gamma_mr = Float64(gamma_mr)
    gamma_mc = Float64(gamma_mc)
    gamma_mr >= 0 || throw(ArgumentError("gamma_mr must be non-negative"))
    gamma_mc >= 0 || throw(ArgumentError("gamma_mc must be non-negative"))

    nvars = nvariables(equations)
    I_n = Matrix{Float64}(I, nvars, nvars)
    W = gamma_mr .* (I_n .- equations.density_projector) .+
        gamma_mc .* (I_n .- equations.hydro_projector)
    return LinearCollisionMatrix(W)
end

@inline function (source_terms::SourceTerms)(u, x, t,
                                             equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    result = zero(SVector{NVARS, eltype(u)})
    @inbounds for source in source_terms.sources
        result += source(u, x, t, equations)
    end
    return result
end

# ------------------------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------------------------

normal_flux_row(equations::IsotropicHarmonicsFiniteT2D, normal) =
    vec((normal[1] .* equations.Ax .+ normal[2] .* equations.Ay)[1, :])

function build_projector_cache(equations::IsotropicHarmonicsFiniteT2D,
                               normal::AbstractVector{T},
                               rho_row::AbstractVector{T}) where {T<:Real}
    nrm = norm(normal)
    nrm > zero(T) || throw(ArgumentError("normal must be nonzero"))
    nx = normal[1] / nrm
    ny = normal[2] / nrm
    A = nx .* equations.Ax .+ ny .* equations.Ay
    nvars = nvariables(equations)
    S = T.(equations.gram_sqrt)

    A_orth = Matrix{T}(undef, nvars, nvars)
    @inbounds for j in 1:nvars, i in 1:nvars
        A_orth[i, j] = S[i] * convert(T, A[i, j]) / S[j]
    end

    F = eigen(Symmetric(A_orth))
    tol = sqrt(eps(T)) * max(one(T), maximum(abs, F.values))
    incoming = F.values .< -tol
    V_in = F.vectors[:, incoming]
    kin = size(V_in, 2)
    kin > 0 || throw(ErrorException("empty incoming eigenspace for normal $normal"))

    e1 = zeros(T, nvars)
    e1[1] = one(T)
    p_in_e1 = (V_in * (V_in' * (S .* e1))) ./ S

    return ProjectorCache{T, nvars, kin, nvars * kin}(
        SVector{2, T}(normal),
        SMatrix{nvars, kin, T}(V_in),
        SVector{nvars, T}(S),
        SVector{nvars, T}(p_in_e1),
        SVector{nvars, T}(rho_row),
    )
end

@inline function _specular_reflection_template(u_inner, normal,
                                               equations::IsotropicHarmonicsFiniteT2D)
    T = eltype(u_inner)
    M = equations.n_harmonics
    R = equations.n_radial

    nrm = norm(normal)
    nx = normal[1] / nrm
    ny = normal[2] / nrm

    cA = ny^2 - nx^2
    sA = -2 * nx * ny

    reflected = MVector{length(u_inner), T}(undef)
    for a in 1:R
        reflected[_finite_scalar_index(R, a)] = u_inner[_finite_scalar_index(R, a)]
    end

    c_m = cA
    s_m = sA
    for ell in 1:M
        for a in 1:R
            cidx = _finite_cos_index(R, ell, a)
            sidx = _finite_sin_index(R, ell, a)
            a_m = u_inner[cidx]
            b_m = u_inner[sidx]
            reflected[cidx] = a_m * c_m + b_m * s_m
            reflected[sidx] = a_m * s_m - b_m * c_m
        end

        if ell < M
            c_next = c_m * cA - s_m * sA
            s_next = s_m * cA + c_m * sA
            c_m, s_m = c_next, s_next
        end
    end

    return SVector(reflected)
end

@inline function _template(bc::MaxwellWallBC, u_inner, normal,
                           equations::IsotropicHarmonicsFiniteT2D)
    alpha = convert(eltype(u_inner), bc.accommodation)
    specular = _specular_reflection_template(u_inner, normal, equations)
    diffuse = _diffuse_isotropic_template(u_inner)
    return (one(alpha) - alpha) * specular + alpha * diffuse
end

@inline function (bc::AbstractBoundaryCondition)(u_inner, normal_direction::AbstractVector,
                                                 x, t, surface_flux_function,
                                                 equations::IsotropicHarmonicsFiniteT2D)
    return _exact_boundary_flux(u_inner, normal_direction, bc, equations)
end

@inline function (bc::AbstractBoundaryCondition)(u_inner, orientation_or_normal,
                                                 direction::Integer, x, t,
                                                 surface_flux_function,
                                                 equations::IsotropicHarmonicsFiniteT2D)
    normal = orientation_or_normal isa Integer ?
             (orientation_or_normal == 1 ?
              SVector(isodd(direction) ? -one(eltype(u_inner)) : one(eltype(u_inner)),
                      zero(eltype(u_inner))) :
              SVector(zero(eltype(u_inner)),
                      isodd(direction) ? -one(eltype(u_inner)) : one(eltype(u_inner)))) :
             orientation_or_normal
    return _exact_boundary_flux(u_inner, normal, bc, equations)
end

function Trixi.semidiscretize(semi::Trixi.SemidiscretizationHyperbolic{<:Any,
                                  <:IsotropicHarmonicsFiniteT2D}, tspan; kwargs...)
    initialize_boundary_projectors!(semi)
    return invoke(Trixi.semidiscretize,
                  Tuple{Trixi.AbstractSemidiscretization, typeof(tspan)},
                  semi, tspan; kwargs...)
end

function Trixi.calc_boundary_flux!(surface_flux_values, t,
                                   boundary_condition::AbstractBoundaryCondition,
                                   mesh::Trixi.P4estMesh{2},
                                   have_nonconservative_terms::Trixi.False,
                                   equations::IsotropicHarmonicsFiniteT2D,
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
