# The moment sweep solves the same sparse system many times. We factor its matrix
# once and reuse the factors. Float32 uses less memory, but we keep Float64 when
# a small pivot makes the lower precision risky

struct MomentLU32
    L::SparseMatrixCSC{Float32,Int}
    U::SparseMatrixCSC{Float32,Int}
    p::Vector{Int}
    q::Vector{Int}
    scaling::Vector{Float64}
    work::Vector{Float32}
end

function moment_factorization(operator, storage)
    factorization = lu(operator)
    storage === Float64 && return factorization
    upper = factorization.U
    smallest, largest = extrema(i -> abs(upper[i, i]), axes(upper, 1))
    # A tiny pivot may lose too much accuracy when U is stored in Float32
    smallest <= eps(Float32) * largest && return factorization
    return MomentLU32(SparseMatrixCSC{Float32,Int}(factorization.L),
                      SparseMatrixCSC{Float32,Int}(upper), factorization.p,
                      factorization.q, factorization.Rs,
                      Vector{Float32}(undef, size(operator, 1)))
end

function LinearAlgebra.ldiv!(solution::AbstractVector{Float64}, factor::MomentLU32,
                             rhs::AbstractVector{Float64})
    # Apply the sparse factorization's row scaling and ordering before solving
    for i in eachindex(factor.work)
        row = factor.p[i]
        factor.work[i] = Float32(factor.scaling[row] * rhs[row])
    end
    ldiv!(LowerTriangular(factor.L), factor.work)
    ldiv!(UpperTriangular(factor.U), factor.work)
    # Put the answer back in the original column order
    for i in eachindex(factor.work)
        solution[factor.q[i]] = Float64(factor.work[i])
    end
    return solution
end

release_factorization!(::Any) = nothing
function release_factorization!(factor::SparseArrays.UMFPACK.UmfpackLU)
    # Free the symbolic data even if releasing the numeric data fails
    try
        finalize(factor.numeric)
    finally
        finalize(factor.symbolic)
    end
    return nothing
end
