using FermiSea
using LinearAlgebra
using StaticArrays
using Test
using Trixi

const TEST_GROUP = get(ENV, "TRIXI_TEST", "all")

function include_test_group(group)
    if group == "all"
        include("test_unit.jl")
        include("test_type.jl")
        include("test_upstream.jl")
        include("test_p4est_2d.jl")
    elseif group == "unit"
        include("test_unit.jl")
    elseif group == "type"
        include("test_type.jl")
    elseif group == "upstream"
        include("test_upstream.jl")
    elseif group == "p4est_2d"
        include("test_p4est_2d.jl")
    else
        error("Unknown TRIXI_TEST group: $group")
    end
end

@testset "FermiSea" begin
    include_test_group(TEST_GROUP)
end
