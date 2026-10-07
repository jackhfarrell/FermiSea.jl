# Store modes and weighted duals so the response projection needs no dense matrix
struct WeightedProjector{T<:AbstractFloat}
    modes::Matrix{T}
    weighted_modes::Matrix{T}
end

function weighted_projector(columns::AbstractMatrix, weights::AbstractVector;
                            rank_tolerance = nothing)
    size(columns, 1) == length(weights) || throw(DimensionMismatch(
        "invariant columns and weights disagree"))
    # Weighting the columns turns the response inner product into the Euclidean SVD
    # metric
    # The resulting factors also drop dependent invariants without forming a dense
    # projector
    scaled = sqrt.(weights) .* columns
    factor = svd(scaled)
    isempty(factor.S) && return WeightedProjector(zeros(length(weights), 0),
                                                   zeros(length(weights), 0))
    tolerance = something(rank_tolerance,
                          max(size(scaled)...) * eps(eltype(factor.S)) * factor.S[1])
    rank = count(>(tolerance), factor.S)
    modes = factor.U[:, 1:rank] ./ sqrt.(weights)
    return WeightedProjector(Matrix(modes), Matrix(weights .* modes))
end

@inline function project(projector::WeightedProjector, u)
    return projector.modes * (transpose(projector.weighted_modes) * u)
end

"""
    Callaway(model; gamma_mr, gamma_mc)

Use the positive collision operator
`C = gamma_mr (I - Pn) + gamma_mc (I - Pnp)`. `Pn` keeps the constant population
mode. `Pnp` also keeps the two momentum modes. The projectors are orthogonal in
the response inner product. Trixi uses `-C * phi` as its source.
"""
struct Callaway{T<:AbstractFloat,P1,P2}
    gamma_mr::T
    gamma_mc::T
    population::P1
    momentum::P2
end

"""
    collision_workspace(collision, ncolumns=1)

Get scratch for applying a collision. Callaway needs none. Tomographic collisions use
`ncolumns` to size their FFT scratch, and custom collisions call the supplied workspace
factory. Use separate scratch for simultaneous calls.
"""
collision_workspace(::Callaway, ncolumns::Integer = 1) = nothing

function Callaway(model::BoltzmannEquation; gamma_mr::Real, gamma_mc::Real)
    gamma_mr >= 0 && isfinite(gamma_mr) ||
        throw(ArgumentError("gamma_mr must be finite and nonnegative"))
    gamma_mc >= 0 && isfinite(gamma_mc) ||
        throw(ArgumentError("gamma_mc must be finite and nonnegative"))
    surface = model.surface
    population_columns = ones(length(surface), 1)
    full_columns = hcat(population_columns, surface.momenta)
    population_projector = weighted_projector(population_columns, surface.weights)
    momentum_projector = weighted_projector(full_columns, surface.weights)
    T = promote_type(Float64, typeof(gamma_mr), typeof(gamma_mc))
    return Callaway{T,typeof(population_projector),typeof(momentum_projector)}(
        gamma_mr, gamma_mc, population_projector, momentum_projector)
end

"""
    CustomCollision(model, action!; workspace=()->nothing, diagonal=nothing,
                    rate_bound=nothing)
    CustomCollision(model, matrix; diagonal=diag(matrix), rate_bound=nothing)

Supply a collision law as a matrix or `action!(out, input, workspace)`.
The callback writes the positive collision action `C * input` to `out` for one
unweighted momentum state. It must leave `input` unchanged, and the arrays cannot
alias. Conservation is up to the supplied law.

`workspace` makes scratch for each caller. If known, supply `diagonal` and a nonnegative
`rate_bound` for the transport preconditioner. Trixi uses `-C * phi` as its source.
"""
struct CustomCollision{F,W,D}
    n::Int
    action!::F
    workspace_factory::W
    diagonal::D
    rate_bound::Union{Nothing,Float64}
end


function CustomCollision(model::BoltzmannEquation, action!;
                         workspace = () -> nothing, diagonal = nothing,
                         rate_bound = nothing)
    n = length(model.surface)
    diagonal === nothing || length(diagonal) == n || throw(DimensionMismatch(
        "collision diagonal must have $n entries"))
    rate_bound === nothing || (isfinite(rate_bound) && rate_bound >= 0) ||
        throw(ArgumentError("rate_bound must be finite and nonnegative"))
    return CustomCollision(n, action!, workspace,
                           diagonal === nothing ? nothing : Vector{Float64}(diagonal),
                           rate_bound === nothing ? nothing : Float64(rate_bound))
end

function CustomCollision(model::BoltzmannEquation, matrix::AbstractMatrix;
                         diagonal = diag(matrix), rate_bound = nothing)
    n = length(model.surface)
    size(matrix) == (n, n) ||
        throw(DimensionMismatch("collision matrix must be $n by $n"))
    action! = (out, input, workspace) -> mul!(out, matrix, input)
    return CustomCollision(model, action!; diagonal, rate_bound)
end

collision_workspace(collision::CustomCollision) = collision.workspace_factory()

"""
    apply_collision!(out, collision, phi[, workspace])

Write `C * phi` into `out` and return it. `phi` can be a momentum vector or a matrix
with one spatial state per column. If no workspace is passed, one is made as needed.
Callaway allows `out` to alias `phi`. CustomCollision does not. Trixi uses the negative
of this action as its source.
"""
function apply_collision!(out, collision::CustomCollision, input, workspace)
    size(out) == size(input) || throw(DimensionMismatch("collision arrays must agree"))
    size(input, 1) == collision.n || throw(DimensionMismatch(
        "the first collision axis must have $(collision.n) nodes"))
    if input isa AbstractVector
        Base.mightalias(out, input) && throw(ArgumentError(
            "custom collision input and output must not alias"))
        collision.action!(out, input, workspace)
    else
        for column in axes(input, 2)
            Base.mightalias(view(out, :, column), view(input, :, column)) &&
                throw(ArgumentError("custom collision input and output must not alias"))
            collision.action!(view(out, :, column), view(input, :, column), workspace)
        end
    end
    return out
end

function apply_collision!(out, collision::CustomCollision, input)
    return apply_collision!(out, collision, input, collision_workspace(collision))
end

"""
    positive_collision(collision, phi)

Return a new array containing the positive collision action `C * phi`. Use
`apply_collision!` to reuse the output and scratch across calls.
"""
function positive_collision(collision::CustomCollision, input)
    output = similar(input, promote_type(Float64, eltype(input)))
    return apply_collision!(output, collision, input)
end

function (collision::CustomCollision)(u, x, t,
                                      equations::BoltzmannEquation{N}) where {N}
    throw(ArgumentError("prepare_source(custom_collision) before using it in Trixi"))
end

function positive_collision(collision::Callaway, u)
    out = similar(u, promote_type(Float64, eltype(u)))
    return apply_collision!(out, collision, u)
end

function apply_collision!(out, collision::Callaway, u,
                          ::Nothing = nothing)
    # In-place callers still need the original state for both weighted projections
    Base.mightalias(out, u) && return apply_collision!(out, collision, copy(u))
    # Each conserved column keeps its own accumulator. One running sum over the
    # whole shell cannot start the next multiply until the previous result returns
    apply_callaway!(out, collision, u,
                    Val(size(collision.population.modes, 2)),
                    Val(size(collision.momentum.modes, 2)))
    return out
end

@inline function apply_callaway!(out, collision::Callaway, u, ::Val{Kp},
                                 ::Val{Km}) where {Kp, Km}
    input = reshape(u, size(collision.population.modes, 1), :)
    output = reshape(out, size(input))
    population_weight = collision.population.weighted_modes
    population_mode = collision.population.modes
    momentum_weight = collision.momentum.weighted_modes
    momentum_mode = collision.momentum.modes
    gamma_mr = collision.gamma_mr
    gamma_mc = collision.gamma_mc
    rate = gamma_mr + gamma_mc
    @inbounds for column in axes(input, 2)
        population = zero(MVector{Kp, Float64})
        momentum = zero(MVector{Km, Float64})
        for node in axes(input, 1)
            value = input[node, column]
            for mode in 1:Kp
                population[mode] = muladd(population_weight[node, mode], value,
                                          population[mode])
            end
            for mode in 1:Km
                momentum[mode] =
                    muladd(momentum_weight[node, mode], value, momentum[mode])
            end
        end
        for node in axes(input, 1)
            acc = rate * input[node, column]
            for mode in 1:Kp
                acc = muladd(-(gamma_mr * population_mode[node, mode]),
                             population[mode], acc)
            end
            for mode in 1:Km
                acc =
                    muladd(-(gamma_mc * momentum_mode[node, mode]), momentum[mode], acc)
            end
            output[node, column] = acc
        end
    end
    return out
end

@inline function (collision::Callaway)(u, x, t,
                                       equations::BoltzmannEquation{N}) where {N}
    output = MVector{N,promote_type(Float64,eltype(u))}(undef)
    apply_collision!(output, collision, u)
    return -SVector{N}(output)
end
