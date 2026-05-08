# This file implements a finite-temperature linear-response harmonic-moment model in
# 2D for a parabolic band. It extends the angular-only zero-temperature model by adding
# an orthonormal radial basis in each angular sector.

# ------------------------------------------------------------------------------------------
# Equation type and constructor
# ------------------------------------------------------------------------------------------

struct MatrixTriplets
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{Float64}
end

@doc raw"""
    IsotropicHarmonicsFiniteT2D(n_harmonics, n_radial;
        mass=1.0, mu0=1.0, temperature=0.05, zmax=20.0, n_quad=256,
        conserve_energy=false, electrostatic_chi=0.0)

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

`LinearCollisionMatrix` provides the linear-response collision model.
`NonlinearBGKCollision` provides an isothermal local-equilibrium BGK closure
using a quadratic finite-temperature expansion in density and drift velocity.

If `electrostatic_chi != 0`, the model also includes the self-consistent gradual
channel force from ``U = \chi n``. The rank-one current contribution is folded
into the conservative flux matrices, while the nonlinear force completion is
applied through `GradualChannelForce2D` and `GradualChannelForceSource`. Trixi
computes the spatial gradients, the parabolic flux is zero, and the volume
source supplies ``-\chi \nabla n \cdot \nabla_p f`` for the perturbation part.
"""
struct IsotropicHarmonicsFiniteT2D{NVARS, ELECTROSTATIC} <: AbstractEquations{2, NVARS}
    n_harmonics::Int
    n_radial::Int
    mass::Float64
    mu0::Float64
    temperature::Float64
    zmax::Float64
    n_quad::Int
    conserve_energy::Bool
    electrostatic_chi::Float64
    z_nodes::Vector{Float64}
    eps_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
    fermi_window::Vector{Float64}
    radial_basis::Vector{Matrix{Float64}}
    radial_basis_z_derivative::Vector{Matrix{Float64}}
    radial_grams::Vector{Matrix{Float64}}
    Ax::Matrix{Float64}
    Ay::Matrix{Float64}
    Ax_triplets::MatrixTriplets
    Ay_triplets::MatrixTriplets
    velocity_embedding_x::Vector{Float64}
    velocity_embedding_y::Vector{Float64}
    Dx_force::Matrix{Float64}
    Dy_force::Matrix{Float64}
    Dx_force_triplets::MatrixTriplets
    Dy_force_triplets::MatrixTriplets
    moment_matrix::Matrix{Float64}
    hydro_embedding::Matrix{Float64}
    hydro_correction::Matrix{Float64}
    hydro_projector::Matrix{Float64}
    density_projector::Matrix{Float64}
    density_row::Vector{Float64}
    density_row_l1::Float64
    momentum_row_x::Vector{Float64}
    momentum_row_y::Vector{Float64}
    equilibrium_density::Float64
    local_equilibrium_linear::Matrix{Float64}
    local_equilibrium_quadratic::Matrix{Float64}
    local_equilibrium_density_linear::Vector{Float64}
    local_equilibrium_density_quadratic::Vector{Float64}
    local_equilibrium_moments_linear::Matrix{Float64}
    local_equilibrium_moments_quadratic::Matrix{Float64}
    gram_sqrt::Vector{Float64}
    force_vmax::Float64
    vmax::Float64
    full_characteristic_projectors::Bool
end

function IsotropicHarmonicsFiniteT2D(n_harmonics::Integer, n_radial::Integer;
                                     mass::Real=1.0,
                                     mu0::Real=1.0,
                                     temperature::Real=0.05,
                                     zmax::Real=20.0,
                                     n_quad::Integer=256,
                                     conserve_energy::Bool=false,
                                     electrostatic_chi::Real=0.0,
                                     full_characteristic_projectors::Bool=false)
    n_harmonics = Int(n_harmonics)
    n_radial = Int(n_radial)
    n_quad = Int(n_quad)
    mass = Float64(mass)
    mu0 = Float64(mu0)
    temperature = Float64(temperature)
    zmax = Float64(zmax)
    electrostatic_chi = Float64(electrostatic_chi)

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
    radial_basis, radial_basis_z_derivative, radial_grams =
        _finite_t_radial_bases(n_harmonics, n_radial, mass, temperature, p_nodes,
                               z_nodes, quad_weights)
    gram_sqrt = _finite_t_gram_sqrt(n_harmonics, n_radial)
    Ax, Ay = _finite_t_streaming_matrices(n_harmonics, n_radial, mass, p_nodes,
                                          quad_weights, radial_basis)
    moment_matrix, hydro_embedding, hydro_projector, density_projector =
        _finite_t_moment_data(n_harmonics, n_radial, eps_nodes, p_nodes, quad_weights,
                              radial_basis, conserve_energy)
    density_row = vec(moment_matrix[1, :])
    density_row_l1 = sum(abs, density_row)
    momentum_row_x = vec(moment_matrix[2, :])
    momentum_row_y = vec(moment_matrix[3, :])
    hydro_correction = hydro_embedding / (moment_matrix * hydro_embedding)
    velocity_embedding_x, velocity_embedding_y =
        _finite_t_velocity_embeddings(n_harmonics, n_radial, mass, p_nodes,
                                      quad_weights, radial_basis)
    local_equilibrium_linear, local_equilibrium_quadratic =
        _finite_t_local_equilibrium_embeddings(n_harmonics, n_radial, mass,
                                               temperature, p_nodes, z_nodes,
                                               quad_weights, radial_basis)
    Dx_force, Dy_force =
        _finite_t_force_derivative_matrices(n_harmonics, n_radial, mass,
                                            temperature, p_nodes, z_nodes,
                                            quad_weights, radial_basis,
                                            radial_basis_z_derivative)
    kinetic_vmax = sqrt(2 * maximum(eps_nodes) / mass)
    if electrostatic_chi != 0
        Ax .+= electrostatic_chi .* (velocity_embedding_x * density_row')
        Ay .+= electrostatic_chi .* (velocity_embedding_y * density_row')
    end
    Ax_triplets = _finite_matrix_triplets(Ax)
    Ay_triplets = _finite_matrix_triplets(Ay)
    Dx_force_triplets = _finite_matrix_triplets(Dx_force)
    Dy_force_triplets = _finite_matrix_triplets(Dy_force)
    force_vmax = electrostatic_chi == 0 ? 0.0 :
                 hypot(opnorm(Dx_force, 2), opnorm(Dy_force, 2))
    vmax = electrostatic_chi == 0 ? kinetic_vmax :
           _finite_t_characteristic_speed(Ax, Ay, kinetic_vmax)
    equilibrium_density = 2 * temperature * _finite_log1pexp(mu0 / temperature)
    local_equilibrium_density_linear = local_equilibrium_linear[:, 1]
    local_equilibrium_density_quadratic = local_equilibrium_quadratic[:, 1]
    local_equilibrium_moments_linear = moment_matrix[1:3, :] * local_equilibrium_linear
    local_equilibrium_moments_quadratic = moment_matrix[1:3, :] *
                                          local_equilibrium_quadratic

    ELECTROSTATIC = electrostatic_chi != 0
    return IsotropicHarmonicsFiniteT2D{NVARS, ELECTROSTATIC}(
        n_harmonics, n_radial, mass, mu0, temperature, zmax, n_quad,
        conserve_energy, electrostatic_chi, z_nodes, eps_nodes, quad_weights,
        fermi_window, radial_basis, radial_basis_z_derivative, radial_grams, Ax, Ay,
        Ax_triplets, Ay_triplets, velocity_embedding_x, velocity_embedding_y,
        Dx_force, Dy_force, Dx_force_triplets, Dy_force_triplets,
        moment_matrix, hydro_embedding, hydro_correction, hydro_projector,
        density_projector,
        density_row, density_row_l1, momentum_row_x, momentum_row_y,
        equilibrium_density, local_equilibrium_linear,
        local_equilibrium_quadratic, local_equilibrium_density_linear,
        local_equilibrium_density_quadratic, local_equilibrium_moments_linear,
        local_equilibrium_moments_quadratic, gram_sqrt, force_vmax, vmax,
        full_characteristic_projectors)
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

@inline function _finite_log1pexp(x::Real)
    x = Float64(x)
    return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
end

@inline function _finite_logexpm1(x::Real)
    x = Float64(x)
    x > 0 || return NaN
    return x > log(2.0) ? x + log1p(-exp(-x)) : log(expm1(x))
end

function _finite_weighted_dot(weights, a, b)
    result = 0.0
    @inbounds for q in eachindex(weights)
        result += weights[q] * a[q] * b[q]
    end
    return result
end

function _finite_matrix_triplets(A::AbstractMatrix)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        value = A[i, j]
        value == 0 && continue
        push!(rows, i)
        push!(cols, j)
        push!(vals, Float64(value))
    end
    return MatrixTriplets(rows, cols, vals)
end

function _finite_t_radial_bases(n_harmonics, n_radial, mass, temperature, p_nodes,
                                z_nodes, weights)
    radial_basis = Vector{Matrix{Float64}}(undef, n_harmonics + 1)
    radial_basis_z_derivative = Vector{Matrix{Float64}}(undef, n_harmonics + 1)
    radial_grams = Vector{Matrix{Float64}}(undef, n_harmonics + 1)
    n_quad = length(z_nodes)

    for ell in 0:n_harmonics
        basis = zeros(Float64, n_radial, n_quad)
        basis_z_derivative = zeros(Float64, n_radial, n_quad)
        for a in 1:n_radial
            seed_power = a - 1
            v = @. p_nodes^ell * z_nodes^seed_power
            dv = zeros(Float64, n_quad)
            if ell > 0 || seed_power > 0
                mt = mass * temperature
                for q in 1:n_quad
                    p = p_nodes[q]
                    z = z_nodes[q]
                    dpell_dz = zero(Float64)
                    if ell > 0
                        dpell_dz = ell * mt * p^(ell - 2)
                    end
                    zpart = seed_power == 0 ? 0.0 :
                            seed_power * p^ell * z^(seed_power - 1)
                    dv[q] = dpell_dz * z^seed_power + zpart
                end
            end
            for j in 1:(a - 1)
                coeff = _finite_weighted_dot(weights, view(basis, j, :), v)
                @inbounds for q in 1:n_quad
                    v[q] -= coeff * basis[j, q]
                    dv[q] -= coeff * basis_z_derivative[j, q]
                end
            end
            norm_v = sqrt(_finite_weighted_dot(weights, v, v))
            if !(norm_v > 100 * eps(Float64))
                throw(ArgumentError("radial seed basis is numerically dependent; " *
                                    "increase n_quad or reduce n_radial"))
            end
            @inbounds for q in 1:n_quad
                basis[a, q] = v[q] / norm_v
                basis_z_derivative[a, q] = dv[q] / norm_v
            end
        end

        gram = zeros(Float64, n_radial, n_radial)
        for b in 1:n_radial, a in 1:n_radial
            gram[a, b] = _finite_weighted_dot(weights, view(basis, a, :),
                                              view(basis, b, :))
        end
        radial_basis[ell + 1] = basis
        radial_basis_z_derivative[ell + 1] = basis_z_derivative
        radial_grams[ell + 1] = gram
    end

    return radial_basis, radial_basis_z_derivative, radial_grams
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

function _finite_t_velocity_embeddings(n_harmonics, n_radial, mass, p_nodes, weights,
                                       radial_basis)
    nvars = n_radial * (1 + 2 * n_harmonics)
    vx = zeros(Float64, nvars)
    vy = zeros(Float64, nvars)
    n_harmonics >= 1 || return vx, vy

    coeffs = _finite_project_radial(radial_basis[2], weights, p_nodes ./ mass)
    @inbounds for a in 1:n_radial
        vx[_finite_cos_index(n_radial, 1, a)] = coeffs[a]
        vy[_finite_sin_index(n_radial, 1, a)] = coeffs[a]
    end
    return vx, vy
end

function _finite_t_characteristic_speed(Ax, Ay, fallback_speed; n_angles=64)
    vmax = Float64(fallback_speed)
    for k in 0:(n_angles - 1)
        theta = 2pi * k / n_angles
        A = cos(theta) .* Ax .+ sin(theta) .* Ay
        values = eigvals(A)
        imag_part = maximum(abs, imag.(values))
        real_speed = maximum(abs, real.(values))
        tol = 100 * sqrt(eps(Float64)) * max(1.0, real_speed)
        imag_part <= tol ||
            throw(ErrorException("finite-temperature normal flux has complex " *
                                 "characteristic speeds"))
        vmax = max(vmax, real_speed)
    end
    return nextfloat(vmax)
end

function _finite_embedding(n_harmonics, n_radial, radial_basis, weights, ell, kind,
                           values; factor=1.0)
    nvars = n_radial * (1 + 2 * n_harmonics)
    embedding = zeros(Float64, nvars)
    ell <= n_harmonics || return embedding

    coeffs = _finite_project_radial(radial_basis[ell + 1], weights, values)
    @inbounds for a in 1:n_radial
        idx = if ell == 0
            _finite_scalar_index(n_radial, a)
        elseif kind === :cos
            _finite_cos_index(n_radial, ell, a)
        else
            _finite_sin_index(n_radial, ell, a)
        end
        embedding[idx] += factor * coeffs[a]
    end
    return embedding
end

function _finite_t_local_equilibrium_embeddings(n_harmonics, n_radial, mass,
                                                temperature, p_nodes, z_nodes, weights,
                                                radial_basis)
    nvars = n_radial * (1 + 2 * n_harmonics)
    linear = zeros(Float64, nvars, 3)
    quadratic = zeros(Float64, nvars, 6)

    ones_q = ones(Float64, length(weights))
    q = @. tanh(z_nodes / 2) / (2 * temperature)
    qp = q .* p_nodes
    qp2 = q .* abs2.(p_nodes)

    linear[:, 1] .= _finite_embedding(n_harmonics, n_radial, radial_basis, weights,
                                      0, :cos, ones_q)
    linear[:, 2] .= _finite_embedding(n_harmonics, n_radial, radial_basis, weights,
                                      1, :cos, p_nodes)
    linear[:, 3] .= _finite_embedding(n_harmonics, n_radial, radial_basis, weights,
                                      1, :sin, p_nodes)

    quadratic[:, 1] .= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                         weights, 0, :cos, q)
    quadratic[:, 2] .= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                         weights, 1, :cos, qp; factor=2.0)
    quadratic[:, 3] .= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                         weights, 1, :sin, qp; factor=2.0)

    # The quadratic drift coefficients come from
    # q(z) * (p_x v_x + p_y v_y + delta_mu)^2 - (m/2) * (v_x^2 + v_y^2),
    # where p_x^2 = p^2 * (1 + cos(2theta)) / 2, p_y^2 = p^2 * (1 - cos(2theta)) / 2,
    # and 2 p_x p_y = p^2 * sin(2theta). If ell=2 is truncated away, only the
    # scalar parts of p_x^2 and p_y^2 remain.
    quadratic[:, 4] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 0, :cos, ones_q; factor=-0.5 * mass)
    quadratic[:, 4] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 0, :cos, qp2; factor=0.5)
    quadratic[:, 4] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 2, :cos, qp2; factor=0.5)
    quadratic[:, 5] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 0, :cos, ones_q; factor=-0.5 * mass)
    quadratic[:, 5] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 0, :cos, qp2; factor=0.5)
    quadratic[:, 5] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 2, :cos, qp2; factor=-0.5)
    quadratic[:, 6] .+= _finite_embedding(n_harmonics, n_radial, radial_basis,
                                          weights, 2, :sin, qp2; factor=1.0)

    return linear, quadratic
end

function _finite_t_radial_force_matrices(basis_out, basis_in, basis_in_z_derivative,
                                         ell_in, mass, temperature, p_nodes, z_nodes,
                                         weights)
    n_out = size(basis_out, 1)
    n_in = size(basis_in, 1)
    radial = zeros(Float64, n_out, n_in)
    angular = zeros(Float64, n_out, n_in)
    @inbounds for b in 1:n_in, a in 1:n_out
        radial_value = 0.0
        angular_value = 0.0
        for q in eachindex(weights)
            p = p_nodes[q]
            basis_value = basis_in[b, q]
            radial_operator = (p / (mass * temperature)) *
                              (basis_in_z_derivative[b, q] -
                               tanh(z_nodes[q] / 2) * basis_value)
            angular_operator = ell_in == 0 ? 0.0 : ell_in * basis_value / p
            radial_value += weights[q] * basis_out[a, q] * radial_operator
            angular_value += weights[q] * basis_out[a, q] * angular_operator
        end
        radial[a, b] = radial_value
        angular[a, b] = angular_value
    end
    return radial, angular
end

function _finite_t_force_derivative_matrices(n_harmonics, n_radial, mass, temperature,
                                             p_nodes, z_nodes, weights, radial_basis,
                                             radial_basis_z_derivative)
    nvars = n_radial * (1 + 2 * n_harmonics)
    Dx = zeros(Float64, nvars, nvars)
    Dy = zeros(Float64, nvars, nvars)

    # Momentum derivatives are represented spectrally: we project
    # (1 / w) * d_{p_i}(w * Phi_in) onto the truncated radial-harmonic basis once
    # in the constructor. Runtime application is then a dense coefficient-space
    # matrix multiply, not pointwise differencing in momentum space.
    radial_force = Matrix{Matrix{Float64}}(undef, n_harmonics + 1, n_harmonics + 1)
    angular_force = Matrix{Matrix{Float64}}(undef, n_harmonics + 1, n_harmonics + 1)
    for ell_out in 0:n_harmonics, ell_in in 0:n_harmonics
        radial_force[ell_out + 1, ell_in + 1],
        angular_force[ell_out + 1, ell_in + 1] =
            _finite_t_radial_force_matrices(radial_basis[ell_out + 1],
                                            radial_basis[ell_in + 1],
                                            radial_basis_z_derivative[ell_in + 1],
                                            ell_in, mass, temperature, p_nodes,
                                            z_nodes, weights)
    end

    if n_harmonics >= 1
        cols0 = _finite_sector_indices(n_radial, 0, :cos)
        rows_c1 = _finite_sector_indices(n_radial, 1, :cos)
        rows_s1 = _finite_sector_indices(n_radial, 1, :sin)
        _finite_add_block!(Dx, rows_c1, cols0, radial_force[2, 1], 1.0)
        _finite_add_block!(Dy, rows_s1, cols0, radial_force[2, 1], 1.0)
    end

    for ell in 1:n_harmonics
        cols_c = _finite_sector_indices(n_radial, ell, :cos)
        cols_s = _finite_sector_indices(n_radial, ell, :sin)

        ell_down = ell - 1
        if ell_down == 0
            rows0 = _finite_sector_indices(n_radial, 0, :cos)
            radial_down = radial_force[1, ell + 1]
            angular_down = angular_force[1, ell + 1]
            _finite_add_block!(Dx, rows0, cols_c, radial_down .+ angular_down, 0.5)
            _finite_add_block!(Dy, rows0, cols_s, radial_down .+ angular_down, 0.5)
        elseif ell_down >= 1
            rows_cd = _finite_sector_indices(n_radial, ell_down, :cos)
            rows_sd = _finite_sector_indices(n_radial, ell_down, :sin)
            radial_down = radial_force[ell_down + 1, ell + 1]
            angular_down = angular_force[ell_down + 1, ell + 1]
            plus_down = radial_down .+ angular_down
            _finite_add_block!(Dx, rows_cd, cols_c, plus_down, 0.5)
            _finite_add_block!(Dx, rows_sd, cols_s, plus_down, 0.5)
            _finite_add_block!(Dy, rows_sd, cols_c, plus_down, -0.5)
            _finite_add_block!(Dy, rows_cd, cols_s, plus_down, 0.5)
        end

        ell_up = ell + 1
        if ell_up <= n_harmonics
            rows_cu = _finite_sector_indices(n_radial, ell_up, :cos)
            rows_su = _finite_sector_indices(n_radial, ell_up, :sin)
            radial_up = radial_force[ell_up + 1, ell + 1]
            angular_up = angular_force[ell_up + 1, ell + 1]
            minus_up = radial_up .- angular_up
            _finite_add_block!(Dx, rows_cu, cols_c, minus_up, 0.5)
            _finite_add_block!(Dx, rows_su, cols_s, minus_up, 0.5)
            _finite_add_block!(Dy, rows_su, cols_c, minus_up, 0.5)
            _finite_add_block!(Dy, rows_cu, cols_s, minus_up, -0.5)
        end
    end

    return Dx, Dy
end

# ------------------------------------------------------------------------------------------
# Trixi equation interface
# ------------------------------------------------------------------------------------------

function _finite_triplets_times_vector(A::MatrixTriplets, u, ::Val{NVARS}) where {NVARS}
    T = eltype(u)
    result = zero(MVector{NVARS, T})
    @inbounds for k in eachindex(A.vals)
        result[A.rows[k]] += convert(T, A.vals[k]) * u[A.cols[k]]
    end
    return SVector(result)
end

function _finite_triplets_times_vector(A::MatrixTriplets, scale, u,
                                      result::MVector{NVARS, T}) where {NVARS, T}
    scale = convert(T, scale)
    @inbounds for k in eachindex(A.vals)
        result[A.rows[k]] += scale * convert(T, A.vals[k]) * u[A.cols[k]]
    end
    return result
end

function _finite_normal_triplets_times_vector(Ax::MatrixTriplets, Ay::MatrixTriplets,
                                             normal_direction::AbstractVector, u,
                                             ::Val{NVARS}) where {NVARS}
    T = eltype(u)
    result = zero(MVector{NVARS, T})
    _finite_triplets_times_vector(Ax, normal_direction[1], u, result)
    _finite_triplets_times_vector(Ay, normal_direction[2], u, result)
    return SVector(result)
end

function _finite_matrix_times_vector(A, u, ::Val{NVARS}) where {NVARS}
    T = eltype(u)
    result = zero(MVector{NVARS, T})
    @inbounds for j in 1:NVARS
        u_j = u[j]
        for i in 1:NVARS
            result[i] += convert(T, A[i, j]) * u_j
        end
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
    result = zero(MVector{NVARS, T})
    @inbounds for j in 1:NVARS
        u_j = normal_direction[1] * u[j]
        for i in 1:NVARS
            result[i] += convert(T, equations.Ax[i, j]) * u_j
        end
    end
    @inbounds for j in 1:NVARS
        u_j = normal_direction[2] * u[j]
        for i in 1:NVARS
            result[i] += convert(T, equations.Ay[i, j]) * u_j
        end
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

Trixi.have_nonconservative_terms(::IsotropicHarmonicsFiniteT2D) = Trixi.False()

@inline max_abs_speed_naive(u_ll, u_rr, ::Integer,
                            equations::IsotropicHarmonicsFiniteT2D) = equations.vmax

@inline max_abs_speed_naive(u_ll, u_rr, normal::AbstractVector,
                            equations::IsotropicHarmonicsFiniteT2D) =
    equations.vmax * norm(normal)

@inline residual_steady_state(du, ::IsotropicHarmonicsFiniteT2D) = maximum(abs, du)

@inline _transformed_speed(speed, Ja1, Ja2) = speed * sqrt(Ja1^2 + Ja2^2)

const _FINITE_T_MESH_UNION = Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                   Trixi.T8codeMesh{2}, Trixi.StructuredMesh{2},
                                   Trixi.StructuredMeshView{2}, Trixi.UnstructuredMesh2D}

# Cache the purely geometric part of the CFL denominator: max over all nodes of
# (‖J₁‖ + ‖J₂‖) × inv_jac, keyed on the Trixi cache object (unique per semi).
# The mesh geometry is static for non-AMR runs, so this only needs computing once.
const _max_dt_geometric_cache = IdDict{Any, Float64}()

function _max_dt_geometric_factor(mesh::_FINITE_T_MESH_UNION, dg::Trixi.DG, cache)
    cached = get(_max_dt_geometric_cache, cache, 0.0)
    cached > 0 && return cached

    contravariant_vectors = cache.elements.contravariant_vectors
    inverse_jacobian = cache.elements.inverse_jacobian
    geo_max = nextfloat(0.0)

    for element in Trixi.eachelement(dg, cache)
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors,
                                                        i, j, element)
            Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors,
                                                        i, j, element)
            inv_jac = abs(inverse_jacobian[i, j, element])
            g1 = sqrt(Ja11^2 + Ja12^2) * inv_jac
            g2 = sqrt(Ja21^2 + Ja22^2) * inv_jac
            geo_max = Base.max(geo_max, g1 + g2)
        end
    end

    _max_dt_geometric_cache[cache] = geo_max
    return geo_max
end

function Trixi.max_dt(u, t, mesh::_FINITE_T_MESH_UNION, constant_speed::Trixi.True,
                      equations::IsotropicHarmonicsFiniteT2D{NVARS, false},
                      dg::Trixi.DG, cache) where {NVARS}
    geo = _max_dt_geometric_factor(mesh, dg, cache)
    return 2 / (Trixi.nnodes(dg) * equations.vmax * geo)
end

function Trixi.max_dt(u, t, mesh::_FINITE_T_MESH_UNION, constant_speed::Trixi.True,
                      equations::IsotropicHarmonicsFiniteT2D{NVARS, true},
                      dg::Trixi.DG, cache) where {NVARS}
    geo = _max_dt_geometric_factor(mesh, dg, cache)
    return 2 / (Trixi.nnodes(dg) * equations.vmax * geo)
end

function _finite_force_matrix(equations::IsotropicHarmonicsFiniteT2D,
                              orientation::Integer)
    orientation == 1 && return equations.Dx_force
    orientation == 2 && return equations.Dy_force
    throw(ArgumentError("orientation must be 1 or 2"))
end

function _finite_force_triplets(equations::IsotropicHarmonicsFiniteT2D,
                                orientation::Integer)
    orientation == 1 && return equations.Dx_force_triplets
    orientation == 2 && return equations.Dy_force_triplets
    throw(ArgumentError("orientation must be 1 or 2"))
end

function _finite_apply_force_matrix(equations::IsotropicHarmonicsFiniteT2D,
                                    normal_direction::AbstractVector, u,
                                    ::Val{NVARS}) where {NVARS}
    T = eltype(u)
    result = zero(MVector{NVARS, T})
    _finite_triplets_times_vector(equations.Dx_force_triplets, normal_direction[1],
                                  u, result)
    _finite_triplets_times_vector(equations.Dy_force_triplets, normal_direction[2],
                                  u, result)
    return SVector(result)
end

@inline function flux_no_electrostatic_nonconservative(u_mine, u_other,
                                                       orientation::Integer,
                                                       equations::IsotropicHarmonicsFiniteT2D)
    return zero(u_mine)
end

@inline function flux_no_electrostatic_nonconservative(u_mine, u_other,
                                                       normal_direction::AbstractVector,
                                                       equations::IsotropicHarmonicsFiniteT2D)
    return zero(u_mine)
end

@doc raw"""
    flux_gradual_channel_volume(u_mine, u_other, direction, equations)

Interior two-point flux for the nonlinear gradual-channel completion
``-\chi \nabla\delta n \cdot \nabla_p \delta f``. The conservative rank-one
piece from ``-\chi \nabla\delta n \cdot \nabla_p f_0`` is already folded into
`Ax, Ay`; this helper represents only the perturbative ``\delta f`` piece. The
momentum derivatives are
the precomputed spectral matrices `Dx_force` and `Dy_force`; `u_other` supplies
the density field differentiated by Trixi's volume flux-differencing operator.

This helper is retained for physics tests and diagnostic checks. The live
hyperbolic discretization keeps this force out of Trixi's nonconservative path;
the actual simulation applies it through `GradualChannelForceSource`.
"""
@inline function flux_gradual_channel_volume(u_mine, u_other,
                                             orientation::Integer,
                                             equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    D = _finite_force_triplets(equations, orientation)
    force_state = _finite_triplets_times_vector(D, u_mine, Val(NVARS))
    density_other = dot(equations.density_row, u_other)
    return -convert(eltype(u_mine), equations.electrostatic_chi) *
           density_other * force_state
end

@inline function flux_gradual_channel_volume(u_mine, u_other,
                                             normal_direction::AbstractVector,
                                             equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    force_state = _finite_apply_force_matrix(equations, normal_direction, u_mine,
                                             Val(NVARS))
    density_other = dot(equations.density_row, u_other)
    return -convert(eltype(u_mine), equations.electrostatic_chi) *
           density_other * force_state
end

# Backwards-compatible name for the volume force.
@inline flux_electrostatic_nonconservative(u_mine, u_other, orientation::Integer,
                                           equations::IsotropicHarmonicsFiniteT2D) =
    flux_gradual_channel_volume(u_mine, u_other, orientation, equations)

@inline flux_electrostatic_nonconservative(u_mine, u_other,
                                           normal_direction::AbstractVector,
                                           equations::IsotropicHarmonicsFiniteT2D) =
    flux_gradual_channel_volume(u_mine, u_other, normal_direction, equations)

# ------------------------------------------------------------------------------------------
# Source terms
# ------------------------------------------------------------------------------------------

"""
    GradualChannelForce2D(equations_hyperbolic)

Parabolic-side equation wrapper for the nonlinear gradual-channel completion.
Trixi's
hyperbolic-parabolic workflow computes gradients of the conservative variables
for this equation; `GradualChannelForceSource` then applies
`-χ ∇δn ⋅ ∇p δf` as a gradient-dependent source term. The rank-one conservative
piece already lives in the hyperbolic flux matrices, so the parabolic flux
itself is zero and this wrapper carries only the nonlinear completion.
"""
struct GradualChannelForce2D{E, NVARS} <:
       Trixi.AbstractEquationsParabolic{2, NVARS,
                                        Trixi.GradientVariablesConservative}
    equations_hyperbolic::E
end

function GradualChannelForce2D(equations_hyperbolic::IsotropicHarmonicsFiniteT2D)
    return GradualChannelForce2D{typeof(equations_hyperbolic),
                                 nvariables(equations_hyperbolic)}(equations_hyperbolic)
end

struct GradualChannelBoundaryCondition{BC} <: AbstractBoundaryCondition
    hyperbolic_bc::BC
end

GradualChannelBoundaryCondition(hyperbolic_bc::AbstractBoundaryCondition) =
    GradualChannelBoundaryCondition{typeof(hyperbolic_bc)}(hyperbolic_bc)

@inline Base.getproperty(equations::GradualChannelForce2D, field::Symbol) =
    field === :equations_hyperbolic ? getfield(equations, field) :
    getproperty(getfield(equations, :equations_hyperbolic), field)

@inline Base.propertynames(equations::GradualChannelForce2D,
                           private::Bool=false) =
    (fieldnames(typeof(equations))...,
     propertynames(getfield(equations, :equations_hyperbolic), private)...)

@inline varnames(variable_mapping, equations::GradualChannelForce2D) =
    varnames(variable_mapping, equations.equations_hyperbolic)

@inline Trixi.have_constant_diffusivity(::GradualChannelForce2D) = Trixi.True()

@inline Trixi.max_diffusivity(::GradualChannelForce2D) = 0.0

@inline function flux(u, gradients, orientation::Integer,
                      equations::GradualChannelForce2D{E, NVARS}) where {E, NVARS}
    return zero(SVector{NVARS, eltype(u)})
end

@inline function (boundary_condition::GradualChannelBoundaryCondition)(flux_inner, u_inner,
                                                                       normal::AbstractVector,
                                                                       x, t,
                                                                       operator_type::Trixi.Gradient,
                                                                       equations::GradualChannelForce2D)
    return assemble_ghost_state(boundary_condition.hyperbolic_bc, u_inner, normal,
                                equations.equations_hyperbolic)
end

@inline function (boundary_condition::GradualChannelBoundaryCondition)(flux_inner, u_inner,
                                                                       normal::AbstractVector,
                                                                       x, t,
                                                                       operator_type::Trixi.Divergence,
                                                                       equations::GradualChannelForce2D)
    return flux_inner
end

struct GradualChannelForceSource end

@inline function (::GradualChannelForceSource)(u, gradients, x, t,
                                               equations::GradualChannelForce2D{E, NVARS}) where {E, NVARS}
    grad_x, grad_y = gradients
    density_x = dot(equations.density_row, grad_x)
    density_y = dot(equations.density_row, grad_y)
    force_state = _finite_apply_force_matrix(equations.equations_hyperbolic,
                                             SVector(density_x, density_y),
                                             u, Val(NVARS))
    return -convert(eltype(u), equations.electrostatic_chi) * force_state
end

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

"""
    NonlinearBGKCollision(equations::IsotropicHarmonicsFiniteT2D; gamma_mr, gamma_mc)

Isothermal finite-temperature BGK source for `IsotropicHarmonicsFiniteT2D`
using the precomputed quadratic local-equilibrium expansion. The
momentum-relaxing rate `gamma_mr` relaxes toward the density-only local
equilibrium, while the momentum-conserving rate `gamma_mc` relaxes toward the
density-and-momentum local equilibrium.
"""
struct NonlinearBGKCollision
    gamma_mr::Float64
    gamma_mc::Float64
end

function NonlinearBGKCollision(equations::IsotropicHarmonicsFiniteT2D;
                               gamma_mr::Real,
                               gamma_mc::Real)
    equations.conserve_energy &&
        throw(ArgumentError("NonlinearBGKCollision currently supports only " *
                            "conserve_energy=false"))
    gamma_mr = Float64(gamma_mr)
    gamma_mc = Float64(gamma_mc)
    gamma_mr >= 0 || throw(ArgumentError("gamma_mr must be non-negative"))
    gamma_mc >= 0 || throw(ArgumentError("gamma_mc must be non-negative"))
    if equations.n_harmonics < 2 && gamma_mc != 0
        @warn "NonlinearBGKCollision with n_harmonics=1 truncates the ell=2 " *
              "and higher harmonic content of the drifting local equilibrium"
    end
    return NonlinearBGKCollision(gamma_mr, gamma_mc)
end

"""
    hydrodynamic_density(equations::IsotropicHarmonicsFiniteT2D, u)

Total isothermal parabolic-band density `n0 + delta_n` represented by coefficient
state `u`. The evolved scalar moment `dot(equations.density_row, u)` is the
perturbation density `delta_n`; this helper adds the equilibrium density because
the nonlinear BGK formulas need the positive total density. If the reconstructed
density is non-positive, this returns `NaN` so time integration can fail through
the residual instead of an exception in the hot path.
"""
function hydrodynamic_density(equations::IsotropicHarmonicsFiniteT2D, u)
    n_total = equations.equilibrium_density + dot(equations.density_row, u)
    return n_total > 0 ? n_total : oftype(n_total, NaN)
end

"""
    hydrodynamic_momentum(equations::IsotropicHarmonicsFiniteT2D, u)

Momentum moments `(Px, Py)` in the finite-temperature internal normalization.
"""
function hydrodynamic_momentum(equations::IsotropicHarmonicsFiniteT2D, u)
    px = dot(equations.momentum_row_x, u)
    py = dot(equations.momentum_row_y, u)
    return SVector(px, py)
end

"""
    hydrodynamic_velocity(equations::IsotropicHarmonicsFiniteT2D, u)

Exact isothermal parabolic-band fluid velocity `P / (m n)` recovered from the
conserved density and momentum moments.
"""
function hydrodynamic_velocity(equations::IsotropicHarmonicsFiniteT2D, u)
    n_total = hydrodynamic_density(equations, u)
    momentum = hydrodynamic_momentum(equations, u)
    return momentum ./ (equations.mass * n_total)
end

"""
    hydrodynamic_chemical_potential_shift(equations::IsotropicHarmonicsFiniteT2D, u)

Exact chemical-potential shift `delta_mu` corresponding to the total density of
state `u` in the isothermal 2D parabolic band.
"""
function hydrodynamic_chemical_potential_shift(equations::IsotropicHarmonicsFiniteT2D,
                                               u)
    n_total = hydrodynamic_density(equations, u)
    return _finite_chemical_potential_shift_from_density(equations, n_total)
end

"""
    hydrodynamic_fields(equations::IsotropicHarmonicsFiniteT2D, u)

Return hydrodynamic diagnostics using the exact finite-temperature isothermal
formulas. `density_delta` is the perturbation density `delta_n`, while `density`
is the total density `n0 + delta_n`. If that density becomes non-positive, the
returned diagnostics are non-finite sentinels.
"""
function hydrodynamic_fields(equations::IsotropicHarmonicsFiniteT2D, u)
    density = hydrodynamic_density(equations, u)
    density_delta = density - equations.equilibrium_density
    delta_mu = _finite_chemical_potential_shift_from_density(equations, density)
    momentum = hydrodynamic_momentum(equations, u)
    velocity = momentum ./ (equations.mass * density)
    speed = hypot(velocity[1], velocity[2])
    electrochemical_potential = delta_mu + equations.electrostatic_chi * density_delta
    return (; density, density_delta, delta_mu, electrochemical_potential, momentum,
            velocity, speed)
end

@inline function _finite_chemical_potential_shift_from_density(equations, density)
    return equations.temperature *
           _finite_logexpm1(density / (2 * equations.temperature)) -
           equations.mu0
end

@inline function _finite_chemical_potential_shift_from_density_delta(equations,
                                                                     density_delta)
    return _finite_chemical_potential_shift_from_density(equations,
                                                         equations.equilibrium_density +
                                                         density_delta)
end

function _finite_density_delta_from_electrochemical_bias(equations, bias)
    chi = equations.electrostatic_chi
    chi >= 0 || throw(ArgumentError("electrochemical Ohmic contact requires " *
                                    "electrostatic_chi >= 0"))
    iszero(chi) && begin
        contact_density = 2 * equations.temperature *
                          _finite_log1pexp((equations.mu0 + bias) /
                                           equations.temperature)
        return contact_density - equations.equilibrium_density
    end

    f(density_delta) =
        _finite_chemical_potential_shift_from_density_delta(equations,
                                                            density_delta) +
        chi * density_delta - bias

    eps_density = max(eps(Float64) * equations.equilibrium_density, eps(Float64))
    lower = -equations.equilibrium_density + eps_density
    upper = max(abs(bias) / max(abs(chi), eps(Float64)) + 1.0, 1.0)
    while f(upper) < 0
        upper *= 2
    end

    # The function is strictly increasing for positive compressibility and chi >= 0,
    # which is the screened electrostatic case used by the gradual-channel model.
    for _ in 1:100
        midpoint = (lower + upper) / 2
        if f(midpoint) < 0
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return (lower + upper) / 2
end

function _finite_bgk_parameters(equations::IsotropicHarmonicsFiniteT2D, u)
    fields = hydrodynamic_fields(equations, u)
    return fields.density, fields.delta_mu, fields.velocity[1], fields.velocity[2]
end

function _finite_quadratic_local_equilibrium(equations::IsotropicHarmonicsFiniteT2D{NVARS},
                                             delta_mu, vx, vy) where {NVARS}
    T = promote_type(typeof(delta_mu), typeof(vx), typeof(vy))
    result = zero(MVector{NVARS, T})

    linear = equations.local_equilibrium_linear
    quadratic = equations.local_equilibrium_quadratic
    delta_mu2 = delta_mu * delta_mu
    delta_mu_vx = delta_mu * vx
    delta_mu_vy = delta_mu * vy
    vx2 = vx * vx
    vy2 = vy * vy
    vx_vy = vx * vy

    @inbounds for i in 1:NVARS
        result[i] = delta_mu * convert(T, linear[i, 1]) +
                    vx * convert(T, linear[i, 2]) +
                    vy * convert(T, linear[i, 3]) +
                    delta_mu2 * convert(T, quadratic[i, 1]) +
                    delta_mu_vx * convert(T, quadratic[i, 2]) +
                    delta_mu_vy * convert(T, quadratic[i, 3]) +
                    vx2 * convert(T, quadratic[i, 4]) +
                    vy2 * convert(T, quadratic[i, 5]) +
                    vx_vy * convert(T, quadratic[i, 6])
    end
    return SVector(result)
end

@inline function _finite_quadratic_local_equilibrium_isotropic(equations, delta_mu)
    return _finite_quadratic_local_equilibrium(equations, delta_mu, zero(delta_mu),
                                               zero(delta_mu))
end

@inline function _finite_exact_response(z, energy_shift, temperature)
    x = energy_shift / temperature
    if abs(x) <= sqrt(eps(Float64))
        return energy_shift +
               energy_shift^2 * tanh(z / 2) / (2 * temperature)
    elseif x > 0
        a = exp(z)
        emx = exp(-x)
        return temperature * (1 - emx) * (1 + a) / (1 + a * emx)
    else
        a = exp(z)
        ex = exp(x)
        return temperature * expm1(x) * (1 + a) / (a + ex)
    end
end

function _finite_local_equilibrium_isotropic(equations::IsotropicHarmonicsFiniteT2D{NVARS},
                                              delta_mu) where {NVARS}
    T = typeof(delta_mu)
    result = zero(MVector{NVARS, T})
    radial_basis = equations.radial_basis[1]
    @inbounds for q in eachindex(equations.quad_weights)
        value = convert(T,
                        equations.quad_weights[q] *
                        _finite_exact_response(equations.z_nodes[q], delta_mu,
                                               equations.temperature))
        for a in 1:equations.n_radial
            result[_finite_scalar_index(equations.n_radial, a)] +=
                value * convert(T, radial_basis[a, q])
        end
    end
    return SVector(result)
end

function _finite_local_equilibrium(equations::IsotropicHarmonicsFiniteT2D{NVARS},
                                   delta_mu, vx, vy) where {NVARS}
    T = promote_type(typeof(delta_mu), typeof(vx), typeof(vy))
    speed2 = vx^2 + vy^2
    iszero(speed2) && return _finite_local_equilibrium_isotropic(equations, delta_mu)

    result = zero(MVector{NVARS, T})
    mass = convert(T, equations.mass)
    drift_speed = sqrt(speed2)
    drift_cos = vx / drift_speed
    drift_sin = vy / drift_speed
    A = delta_mu - T(0.5) * mass * speed2

    ntheta = max(64, 16 * equations.n_harmonics)
    angular_weight = convert(T, 2 / ntheta)
    ctheta_start = cospi(1 / ntheta)
    stheta_start = sinpi(1 / ntheta)
    ctheta_step = cospi(2 / ntheta)
    stheta_step = sinpi(2 / ntheta)

    @inbounds for q in eachindex(equations.quad_weights)
        p = sqrt(2 * mass * convert(T, equations.eps_nodes[q]))
        scalar_average = zero(T)
        harmonic_average = zeros(T, equations.n_harmonics)

        ctheta = ctheta_start
        stheta = stheta_start
        for _ in 1:ntheta
            energy_shift = A + p * drift_speed * ctheta
            value = _finite_exact_response(equations.z_nodes[q], energy_shift,
                                           equations.temperature)
            scalar_average += value

            c_ell = ctheta
            s_ell = stheta
            for ell in 1:equations.n_harmonics
                harmonic_average[ell] += value * c_ell
                if ell < equations.n_harmonics
                    c_next = c_ell * ctheta - s_ell * stheta
                    s_next = s_ell * ctheta + c_ell * stheta
                    c_ell, s_ell = c_next, s_next
                end
            end

            c_next = ctheta * ctheta_step - stheta * stheta_step
            s_next = stheta * ctheta_step + ctheta * stheta_step
            ctheta, stheta = c_next, s_next
        end

        scalar_average *= angular_weight / 2
        for ell in eachindex(harmonic_average)
            harmonic_average[ell] *= angular_weight
        end

        weight = convert(T, equations.quad_weights[q])
        for a in 1:equations.n_radial
            result[_finite_scalar_index(equations.n_radial, a)] +=
                weight * scalar_average *
                convert(T, equations.radial_basis[1][a, q])
        end

        cphi = drift_cos
        sphi = drift_sin
        for ell in 1:equations.n_harmonics
            aligned = weight * harmonic_average[ell]
            basis = equations.radial_basis[ell + 1]
            for a in 1:equations.n_radial
                coeff = aligned * convert(T, basis[a, q])
                result[_finite_cos_index(equations.n_radial, ell, a)] += coeff * cphi
                result[_finite_sin_index(equations.n_radial, ell, a)] += coeff * sphi
            end
            if ell < equations.n_harmonics
                c_next = cphi * drift_cos - sphi * drift_sin
                s_next = sphi * drift_cos + cphi * drift_sin
                cphi, sphi = c_next, s_next
            end
        end
    end
    return SVector(result)
end

@inline function _finite_quadratic_equilibrium_moments(equations, delta_mu, vx, vy)
    T = promote_type(typeof(delta_mu), typeof(vx), typeof(vy))
    linear = equations.local_equilibrium_moments_linear
    quadratic = equations.local_equilibrium_moments_quadratic
    delta_mu2 = delta_mu * delta_mu
    delta_mu_vx = delta_mu * vx
    delta_mu_vy = delta_mu * vy
    vx2 = vx * vx
    vy2 = vy * vy
    vx_vy = vx * vy

    density = delta_mu * convert(T, linear[1, 1]) +
              vx * convert(T, linear[1, 2]) +
              vy * convert(T, linear[1, 3]) +
              delta_mu2 * convert(T, quadratic[1, 1]) +
              delta_mu_vx * convert(T, quadratic[1, 2]) +
              delta_mu_vy * convert(T, quadratic[1, 3]) +
              vx2 * convert(T, quadratic[1, 4]) +
              vy2 * convert(T, quadratic[1, 5]) +
              vx_vy * convert(T, quadratic[1, 6])
    px = delta_mu * convert(T, linear[2, 1]) +
         vx * convert(T, linear[2, 2]) +
         vy * convert(T, linear[2, 3]) +
         delta_mu2 * convert(T, quadratic[2, 1]) +
         delta_mu_vx * convert(T, quadratic[2, 2]) +
         delta_mu_vy * convert(T, quadratic[2, 3]) +
         vx2 * convert(T, quadratic[2, 4]) +
         vy2 * convert(T, quadratic[2, 5]) +
         vx_vy * convert(T, quadratic[2, 6])
    py = delta_mu * convert(T, linear[3, 1]) +
         vx * convert(T, linear[3, 2]) +
         vy * convert(T, linear[3, 3]) +
         delta_mu2 * convert(T, quadratic[3, 1]) +
         delta_mu_vx * convert(T, quadratic[3, 2]) +
         delta_mu_vy * convert(T, quadratic[3, 3]) +
         vx2 * convert(T, quadratic[3, 4]) +
         vy2 * convert(T, quadratic[3, 5]) +
         vx_vy * convert(T, quadratic[3, 6])
    return SVector(density, px, py)
end

@inline function _finite_quadratic_equilibrium_jacobian(equations, delta_mu, vx, vy)
    T = promote_type(typeof(delta_mu), typeof(vx), typeof(vy))
    linear = equations.local_equilibrium_moments_linear
    quadratic = equations.local_equilibrium_moments_quadratic

    d_delta_mu = SVector(
        convert(T, linear[1, 1]) + 2 * delta_mu * convert(T, quadratic[1, 1]) +
        vx * convert(T, quadratic[1, 2]) + vy * convert(T, quadratic[1, 3]),
        convert(T, linear[2, 1]) + 2 * delta_mu * convert(T, quadratic[2, 1]) +
        vx * convert(T, quadratic[2, 2]) + vy * convert(T, quadratic[2, 3]),
        convert(T, linear[3, 1]) + 2 * delta_mu * convert(T, quadratic[3, 1]) +
        vx * convert(T, quadratic[3, 2]) + vy * convert(T, quadratic[3, 3]))
    d_vx = SVector(
        convert(T, linear[1, 2]) + delta_mu * convert(T, quadratic[1, 2]) +
        2 * vx * convert(T, quadratic[1, 4]) + vy * convert(T, quadratic[1, 6]),
        convert(T, linear[2, 2]) + delta_mu * convert(T, quadratic[2, 2]) +
        2 * vx * convert(T, quadratic[2, 4]) + vy * convert(T, quadratic[2, 6]),
        convert(T, linear[3, 2]) + delta_mu * convert(T, quadratic[3, 2]) +
        2 * vx * convert(T, quadratic[3, 4]) + vy * convert(T, quadratic[3, 6]))
    d_vy = SVector(
        convert(T, linear[1, 3]) + delta_mu * convert(T, quadratic[1, 3]) +
        vx * convert(T, quadratic[1, 6]) + 2 * vy * convert(T, quadratic[1, 5]),
        convert(T, linear[2, 3]) + delta_mu * convert(T, quadratic[2, 3]) +
        vx * convert(T, quadratic[2, 6]) + 2 * vy * convert(T, quadratic[2, 5]),
        convert(T, linear[3, 3]) + delta_mu * convert(T, quadratic[3, 3]) +
        vx * convert(T, quadratic[3, 6]) + 2 * vy * convert(T, quadratic[3, 5]))
    return SMatrix{3, 3, T}(d_delta_mu[1], d_vx[1], d_vy[1],
                            d_delta_mu[2], d_vx[2], d_vy[2],
                            d_delta_mu[3], d_vx[3], d_vy[3])
end

function _finite_match_quadratic_density_parameter(equations, density_delta)
    T = typeof(density_delta)
    c1 = convert(T, equations.local_equilibrium_moments_linear[1, 1])
    c2 = convert(T, equations.local_equilibrium_moments_quadratic[1, 1])
    if !isfinite(density_delta)
        return oftype(density_delta, NaN)
    elseif abs(c2) <= sqrt(eps(T))
        return density_delta / c1
    end

    discriminant = c1^2 + 4 * c2 * density_delta
    discriminant >= zero(T) || return oftype(density_delta, NaN)

    root_disc = sqrt(discriminant)
    delta_mu_exact = _finite_chemical_potential_shift_from_density_delta(equations,
                                                                         density_delta)
    root1 = (-c1 + root_disc) / (2 * c2)
    root2 = (-c1 - root_disc) / (2 * c2)
    return abs(root1 - delta_mu_exact) <= abs(root2 - delta_mu_exact) ? root1 : root2
end

function _finite_match_quadratic_hydro_parameters(equations, u)
    density_delta = dot(equations.density_row, u)
    px = dot(equations.momentum_row_x, u)
    py = dot(equations.momentum_row_y, u)
    target = SVector(density_delta, px, py)

    _, delta_mu0, vx0, vy0 = _finite_bgk_parameters(equations, u)
    T = promote_type(typeof(delta_mu0), typeof(vx0), typeof(vy0), eltype(u))
    params = MVector{3, T}(delta_mu0, vx0, vy0)
    tolerance = T(256) * eps(T)

    for _ in 1:20
        moments = _finite_quadratic_equilibrium_moments(equations, params[1], params[2],
                                                        params[3])
        residual = moments - target
        max_residual = maximum(abs, residual)
        max_residual <= tolerance * max(one(T), maximum(abs, target)) &&
            return SVector(params)

        jacobian = _finite_quadratic_equilibrium_jacobian(equations, params[1], params[2],
                                                          params[3])
        if !all(isfinite, jacobian)
            return SVector(oftype(params[1], NaN), oftype(params[2], NaN),
                           oftype(params[3], NaN))
        end

        delta = jacobian \ residual
        all(isfinite, delta) || return SVector(oftype(params[1], NaN),
                                               oftype(params[2], NaN),
                                               oftype(params[3], NaN))
        params .-= delta
        if maximum(abs, delta) <= tolerance * max(one(T), maximum(abs, params))
            moments = _finite_quadratic_equilibrium_moments(equations, params[1], params[2],
                                                            params[3])
            residual = moments - target
            maximum(abs, residual) <= tolerance * max(one(T), maximum(abs, target)) &&
                return SVector(params)
            return SVector(oftype(params[1], NaN), oftype(params[2], NaN),
                           oftype(params[3], NaN))
        end
    end

    return SVector(oftype(params[1], NaN), oftype(params[2], NaN),
                   oftype(params[3], NaN))
end

function _finite_correct_density(equations::IsotropicHarmonicsFiniteT2D{NVARS}, target,
                                 u) where {NVARS}
    T = promote_type(eltype(target), eltype(u))
    residual = dot(equations.density_row, u) - dot(equations.density_row, target)
    embedding = view(equations.hydro_embedding, :, 1)
    denom = dot(equations.density_row, embedding)
    correction = residual / denom
    result = MVector{NVARS, T}(undef)
    @inbounds for i in 1:NVARS
        result[i] = target[i] + correction * convert(T, embedding[i])
    end
    return SVector(result)
end

function _finite_correct_hydro(equations::IsotropicHarmonicsFiniteT2D{NVARS}, target,
                               u) where {NVARS}
    return _finite_correct_hydro(equations, target, u,
                                 Val(size(equations.hydro_correction, 2)))
end

function _finite_correct_hydro(equations::IsotropicHarmonicsFiniteT2D{NVARS}, target,
                               u, ::Val{N_MOMENTS}) where {NVARS, N_MOMENTS}
    T = promote_type(eltype(target), eltype(u))
    result = MVector{NVARS, T}(undef)
    residual1 = zero(T)
    residual2 = zero(T)
    residual3 = zero(T)
    residual4 = zero(T)
    @inbounds for j in 1:NVARS
        delta = u[j] - target[j]
        residual1 += convert(T, equations.moment_matrix[1, j]) * delta
        residual2 += convert(T, equations.moment_matrix[2, j]) * delta
        residual3 += convert(T, equations.moment_matrix[3, j]) * delta
        if N_MOMENTS == 4
            residual4 += convert(T, equations.moment_matrix[4, j]) * delta
        end
    end
    @inbounds for i in 1:NVARS
        correction = convert(T, equations.hydro_correction[i, 1]) * residual1 +
                     convert(T, equations.hydro_correction[i, 2]) * residual2 +
                     convert(T, equations.hydro_correction[i, 3]) * residual3
        if N_MOMENTS == 4
            correction += convert(T, equations.hydro_correction[i, 4]) * residual4
        end
        result[i] = target[i] + correction
    end
    return SVector(result)
end

function _finite_density_local_equilibrium(equations::IsotropicHarmonicsFiniteT2D,
                                           u)
    delta_mu = _finite_match_quadratic_density_parameter(equations,
                                                         dot(equations.density_row, u))
    target = _finite_quadratic_local_equilibrium_isotropic(equations, delta_mu)
    return target
end

function _finite_hydro_local_equilibrium(equations::IsotropicHarmonicsFiniteT2D,
                                         u)
    delta_mu, vx, vy = _finite_match_quadratic_hydro_parameters(equations, u)
    target = _finite_quadratic_local_equilibrium(equations, delta_mu, vx, vy)
    return target
end

@inline function (source::NonlinearBGKCollision)(u, x, t,
                                                 equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    equations.conserve_energy &&
        throw(ArgumentError("NonlinearBGKCollision currently supports only " *
                            "conserve_energy=false"))
    density_target = _finite_density_local_equilibrium(equations, u)
    hydro_target = _finite_hydro_local_equilibrium(equations, u)

    T = eltype(u)
    result = MVector{NVARS, T}(undef)
    gamma_mr = convert(T, source.gamma_mr)
    gamma_mc = convert(T, source.gamma_mc)
    @inbounds for i in 1:NVARS
        result[i] = -gamma_mr * (u[i] - density_target[i]) -
                    gamma_mc * (u[i] - hydro_target[i])
    end
    return SVector(result)
end

@inline function (source::LinearCollisionMatrix)(u, x, t,
                                                  equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    T = eltype(u)
    result = zero(MVector{NVARS, T})
    W = source.W
    @inbounds for j in 1:NVARS
        u_j = u[j]
        for i in 1:NVARS
            result[i] -= convert(T, W[i, j]) * u_j
        end
    end
    return SVector(result)
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
    vec(transpose(equations.density_row) *
        (normal[1] .* equations.Ax .+ normal[2] .* equations.Ay))

function _finite_projector_matrix(equations::IsotropicHarmonicsFiniteT2D,
                                  nx, ny; full_operator::Bool)
    A = nx .* equations.Ax .+ ny .* equations.Ay
    if !full_operator && equations.electrostatic_chi != 0
        A .-= equations.electrostatic_chi .*
              ((nx .* equations.velocity_embedding_x .+
                ny .* equations.velocity_embedding_y) * equations.density_row')
    end
    return A
end

function _real_matrix_from_complex(M::AbstractMatrix{Complex{T}},
                                   label::AbstractString) where {T<:Real}
    imag_norm = maximum(abs, imag.(M))
    imag_norm <= sqrt(eps(T)) * max(one(T), maximum(abs, real.(M))) ||
        throw(ArgumentError("$label has non-negligible imaginary part $imag_norm"))
    return real.(M)
end

function _biorthogonal_projector_from_operator(A_orth::AbstractMatrix{T},
                                               gram_sqrt::AbstractVector{T},
                                               tol::T) where {T<:Real}
    nvars = length(gram_sqrt)
    right = eigen(ComplexF64.(A_orth))
    left = eigen(adjoint(ComplexF64.(A_orth)))
    incoming_right = real.(right.values) .< -tol
    incoming_left = real.(left.values) .< -tol
    kin = count(incoming_right)
    kin == count(incoming_left) || throw(ErrorException(
        "left/right incoming eigenspace dimension mismatch ($kin vs " *
        "$(count(incoming_left)))"))
    kin > 0 || throw(ErrorException("empty incoming eigenspace for full projector"))

    R = right.vectors[:, incoming_right]
    L = left.vectors[:, incoming_left]
    overlap = adjoint(L) * R
    inv(cond(overlap)) > sqrt(eps(T)) || throw(ErrorException(
        "incoming full-characteristic eigenspace is ill-conditioned"))
    P_orth_complex = R * (overlap \ adjoint(L))
    P_orth = _real_matrix_from_complex(P_orth_complex, "full-characteristic projector")

    P = Matrix{T}(undef, nvars, nvars)
    @inbounds for j in 1:nvars, i in 1:nvars
        P[i, j] = P_orth[i, j] * gram_sqrt[j] / gram_sqrt[i]
    end
    return P, kin
end

function build_projector_cache(equations::IsotropicHarmonicsFiniteT2D,
                               normal::AbstractVector{T},
                               rho_row::AbstractVector{T};
                               full_operator::Bool=equations.full_characteristic_projectors) where {T<:Real}
    nrm = norm(normal)
    nrm > zero(T) || throw(ArgumentError("normal must be nonzero"))
    nx = normal[1] / nrm
    ny = normal[2] / nrm
    A = _finite_projector_matrix(equations, nx, ny; full_operator=full_operator)
    nvars = nvariables(equations)
    S = T.(equations.gram_sqrt)

    A_orth = Matrix{T}(undef, nvars, nvars)
    @inbounds for j in 1:nvars, i in 1:nvars
        A_orth[i, j] = S[i] * convert(T, A[i, j]) / S[j]
    end

    F = eigen(Symmetric(A_orth))
    tol = sqrt(eps(T)) * max(one(T), maximum(abs, F.values))
    P_in, kin = if full_operator && equations.electrostatic_chi != 0
        _biorthogonal_projector_from_operator(A_orth, S, tol)
    else
        incoming = F.values .< -tol
        V_in = F.vectors[:, incoming]
        (_coeff_projector_from_orthogonal_basis(V_in, S), size(V_in, 2))
    end
    kin > 0 || throw(ErrorException("empty incoming eigenspace for normal $normal"))

    e1 = zeros(T, nvars)
    e1[1] = one(T)
    p_in_e1 = P_in * e1

    return ProjectorCache{T, nvars, kin, nvars * nvars}(
        SVector{2, T}(normal),
        SMatrix{nvars, nvars, T}(P_in),
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
    iszero(alpha) && return _specular_reflection_template(u_inner, normal, equations)
    alpha == one(alpha) && return _finite_diffuse_thermal_template(u_inner, equations)

    specular = _specular_reflection_template(u_inner, normal, equations)
    diffuse = _finite_diffuse_thermal_template(u_inner, equations)
    return (one(alpha) - alpha) * specular + alpha * diffuse
end

function _finite_diffuse_thermal_template(u_inner,
                                          equations::IsotropicHarmonicsFiniteT2D)
    density_delta = dot(equations.density_row, u_inner)
    n_total = equations.equilibrium_density + density_delta
    delta_mu = _finite_chemical_potential_shift_from_density(equations, n_total)
    target = _finite_local_equilibrium(equations, delta_mu, zero(delta_mu),
                                       zero(delta_mu))
    return _finite_correct_to_density(equations, target, density_delta)
end

@doc raw"""
    ChemicalPotentialContactBC(delta_mu)

Finite-temperature contact that prescribes the incoming reservoir's kinetic
chemical-potential shift ``\delta\mu``. The target reservoir density is computed
from the exact 2D parabolic-band relation

```math
\delta n = 2T\left[\log(1 + e^{(\mu_0 + \delta\mu)/T}) -
                  \log(1 + e^{\mu_0/T})\right],
```

projected to the finite-temperature local-equilibrium basis, and then applied
only through incoming kinetic characteristics. If `electrostatic_chi != 0`, the
resulting density gradient drives the gradual-channel force through the PDE term
``U = \chi \delta n``; the contact itself does not screen the imposed
``\delta\mu`` into an electrochemical voltage.
"""
struct ChemicalPotentialContactBC{T<:Real} <: AbstractBoundaryCondition
    delta_mu::T
    ChemicalPotentialContactBC(delta_mu::Real) = new{Float64}(Float64(delta_mu))
end

@doc raw"""
    DensityContactBC(relative_density)

Finite-temperature contact that prescribes the incoming reservoir density as a
relative perturbation ``\delta n / n_0``. The target density is converted to the
corresponding exact isothermal chemical-potential shift before projection into
the finite-temperature local-equilibrium basis. As with other finite-T contacts,
the target is imposed only on incoming kinetic half-space characteristics.
"""
struct DensityContactBC{T<:Real} <: AbstractBoundaryCondition
    relative_density::T

    function DensityContactBC(relative_density::Real)
        relative_density = Float64(relative_density)
        relative_density > -1 ||
            throw(ArgumentError("relative_density must be greater than -1"))
        return new{Float64}(relative_density)
    end
end

function _finite_correct_to_density(equations::IsotropicHarmonicsFiniteT2D{NVARS},
                                    target, density_delta) where {NVARS}
    T = promote_type(eltype(target), typeof(density_delta))
    residual = density_delta - dot(equations.density_row, target)
    embedding = view(equations.hydro_embedding, :, 1)
    denom = dot(equations.density_row, embedding)
    correction = residual / denom
    result = MVector{NVARS, T}(undef)
    @inbounds for i in 1:NVARS
        result[i] = target[i] + correction * convert(T, embedding[i])
    end
    return SVector(result)
end

function _finite_template_from_chemical_potential(equations::IsotropicHarmonicsFiniteT2D,
                                                  delta_mu)
    target = _finite_local_equilibrium(equations, delta_mu, zero(delta_mu),
                                       zero(delta_mu))
    contact_density = 2 * equations.temperature *
                      _finite_log1pexp((equations.mu0 + delta_mu) /
                                       equations.temperature)
    density_delta = contact_density - equations.equilibrium_density
    return _finite_correct_to_density(equations, target, density_delta)
end

function _template(bc::ChemicalPotentialContactBC, u_inner,
                   equations::IsotropicHarmonicsFiniteT2D)
    delta_mu = convert(eltype(u_inner), bc.delta_mu)
    return _finite_template_from_chemical_potential(equations, delta_mu)
end

function _template(bc::DensityContactBC, u_inner,
                   equations::IsotropicHarmonicsFiniteT2D)
    density_delta = convert(eltype(u_inner),
                            bc.relative_density * equations.equilibrium_density)
    delta_mu = _finite_chemical_potential_shift_from_density_delta(equations,
                                                                   density_delta)
    target = _finite_local_equilibrium(equations, delta_mu, zero(delta_mu),
                                       zero(delta_mu))
    return _finite_correct_to_density(equations, target, density_delta)
end

function _template(bc::OhmicContactBC, u_inner,
                   equations::IsotropicHarmonicsFiniteT2D)
    bias = convert(eltype(u_inner), bc.bias)
    density_delta = _finite_density_delta_from_electrochemical_bias(equations, bias)
    delta_mu = _finite_chemical_potential_shift_from_density_delta(equations,
                                                                   density_delta)
    target = _finite_local_equilibrium(equations, delta_mu, zero(delta_mu),
                                       zero(delta_mu))
    return _finite_correct_to_density(equations, target, density_delta)
end

function _finite_add_boundary_density_shift(base, direction, coefficient,
                                            ::Val{NVARS}) where {NVARS}
    T = promote_type(eltype(base), eltype(direction), typeof(coefficient))
    result = MVector{NVARS, T}(undef)
    @inbounds for i in 1:NVARS
        result[i] = base[i] + coefficient * direction[i]
    end
    return SVector(result)
end

function _finite_enforce_contact_electrochemical_potential(cache::ProjectorCache,
                                                           base,
                                                           bc::OhmicContactBC,
                                                           equations::IsotropicHarmonicsFiniteT2D{NVARS}) where {NVARS}
    iszero(equations.electrostatic_chi) && return base

    direction = cache.p_in_e1
    density_slope = dot(equations.density_row, direction)
    abs(density_slope) > sqrt(eps(eltype(base))) ||
        throw(ArgumentError("incoming density correction is singular for this contact"))

    target_density_delta =
        _finite_density_delta_from_electrochemical_bias(equations,
                                                        convert(eltype(base),
                                                                bc.bias))
    density_base = dot(equations.density_row, base)
    coefficient = (target_density_delta - density_base) / density_slope
    return _finite_add_boundary_density_shift(base, direction, coefficient, Val(NVARS))
end

function assemble_ghost_state(cache::ProjectorCache, bc::OhmicContactBC,
                              u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    template = _template(bc, u_inner, normal, equations)
    base = u_inner + _apply_P_in(cache, template - u_inner)
    return _finite_enforce_contact_electrochemical_potential(cache, base, bc,
                                                             equations)
end

function assemble_ghost_state(bc::OhmicContactBC, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    cache = _build_uncached_projectors(normal, equations)
    return assemble_ghost_state(cache, bc, u_inner, normal, equations)
end

function assemble_ghost_state(cache::ProjectorCache, bc::ChemicalPotentialContactBC,
                              u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    template = _template(bc, u_inner, normal, equations)
    return u_inner + _apply_P_in(cache, template - u_inner)
end

function assemble_ghost_state(bc::ChemicalPotentialContactBC, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    cache = _build_uncached_projectors(normal, equations)
    return assemble_ghost_state(cache, bc, u_inner, normal, equations)
end

function assemble_ghost_state(cache::ProjectorCache, bc::DensityContactBC,
                              u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    template = _template(bc, u_inner, normal, equations)
    return u_inner + _apply_P_in(cache, template - u_inner)
end

function assemble_ghost_state(bc::DensityContactBC, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    cache = _build_uncached_projectors(normal, equations)
    return assemble_ghost_state(cache, bc, u_inner, normal, equations)
end

_target_flux(::MaxwellWallBC, u_inner, normal,
             equations::IsotropicHarmonicsFiniteT2D) = zero(eltype(u_inner))

function assemble_ghost_state(bc::AbstractBoundaryCondition, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    cache = _build_uncached_projectors(normal, equations)
    template = _template(bc, u_inner, normal, equations)
    target_flux = _target_flux(bc, u_inner, normal, equations)
    u_boundary, _ = solve_bc_constant(cache, u_inner;
                                      u_bc_template=template,
                                      target_flux=target_flux)
    return u_boundary
end

function assemble_ghost_state(bc::CurrentContactBC, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    throw(ArgumentError("CurrentContactBC enforces a total current over a full " *
                        "boundary, so assemble_ghost_state requires a semidiscretization " *
                        "boundary-flux context. Use OhmicContactBC for a prescribed local " *
                        "potential template."))
end

function assemble_ghost_state(bc::FloatingProbeBC, u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    throw(ArgumentError("FloatingProbeBC enforces zero total current over a full " *
                        "boundary, so assemble_ghost_state requires a semidiscretization " *
                        "boundary-flux context. Use OhmicContactBC for a prescribed local " *
                        "potential template."))
end

function assemble_ghost_state(cache::ProjectorCache, bc::AbstractBoundaryCondition,
                              u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D)
    template = _template(bc, u_inner, normal, equations)
    target_flux = _target_flux(bc, u_inner, normal, equations)
    u_boundary, _ = solve_bc_constant(cache, u_inner;
                                      u_bc_template=template,
                                      target_flux=target_flux)
    return u_boundary
end

function _current_contact_potential(bc::Union{CurrentContactBC, FloatingProbeBC},
                                    equations::IsotropicHarmonicsFiniteT2D,
                                    dg, cache)
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
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = Trixi.indices2direction(node_indices)

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
                              u_inner, normal,
                              equations::IsotropicHarmonicsFiniteT2D, potential)
    return _current_contact_state(cache, u_inner, potential)
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

@inline function _initialize_finite_t_boundary_projectors!(semi)
    initialize_boundary_projectors!(semi)
    _, _, _, trixi_cache = Trixi.mesh_equations_solver_cache(semi)
    delete!(_max_dt_geometric_cache, trixi_cache)
    return semi
end

function Trixi.semidiscretize(semi::Trixi.SemidiscretizationHyperbolic{<:Any,
                                  <:IsotropicHarmonicsFiniteT2D}, tspan; kwargs...)
    _initialize_finite_t_boundary_projectors!(semi)
    return invoke(Trixi.semidiscretize,
                  Tuple{Trixi.AbstractSemidiscretization, typeof(tspan)},
                  semi, tspan; kwargs...)
end

function Trixi.semidiscretize(semi::Trixi.SemidiscretizationHyperbolicParabolic{<:Any,
                                  <:Tuple{<:IsotropicHarmonicsFiniteT2D, <:Any}},
                              tspan; kwargs...)
    _initialize_finite_t_boundary_projectors!(semi)
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
