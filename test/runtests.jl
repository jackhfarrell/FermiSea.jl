using FermiSea
using CairoMakie
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Test
using Trixi

include("helpers.jl")
include("test_surface_collision.jl")
include("test_local_operators.jl")
include("test_equations_boundaries.jl")
include("test_steady.jl")
include("test_optimized.jl")
include("test_boundaries_contacts.jl")
include("test_moment_sweep.jl")
include("test_workflow.jl")
include("test_plot_data.jl")
include("test_io.jl")
include("test_makie.jl")
include("test_contracts.jl")
