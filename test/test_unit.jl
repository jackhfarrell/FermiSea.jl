@testset "unit" begin
    include("test_rate_profiles.jl")
    include("test_equation.jl")
    include("test_source_terms.jl")
    include("test_characteristic.jl")
    include("test_boundary_conditions.jl")
end
