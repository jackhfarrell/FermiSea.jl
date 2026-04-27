@testset "p4est_2d" begin
    include("test_convergence.jl")
    include("test_analysis.jl")

    @testset "square bells mesh" begin
        mesh_dir = joinpath(@__DIR__, "..", "assets", "square_bells")
        mesh_candidates = [
            joinpath(mesh_dir, "square_bells.inp"),
            joinpath(mesh_dir, "square_bells.mesh"),
        ]
        mesh_index = findfirst(isfile, mesh_candidates)

        @test mesh_index !== nothing

        if mesh_index !== nothing
            mesh = P4estMesh{2}(mesh_candidates[mesh_index]; polydeg=4)
            @test ndims(mesh) == 2
            @test !isempty(mesh.boundary_names)
        end
    end
end
