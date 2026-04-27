@testset "types" begin
    equations = IsotropicFermiHarmonics2D(2; v_fermi=1.0f0)
    u = SVector(1.0f0, 0.2f0, -0.3f0, 0.4f0, -0.5f0)

    @test equations isa Trixi.AbstractEquations{2, 5}
    @test eltype(Trixi.flux(u, 1, equations)) == Float32
    @test eltype(Trixi.flux(u, SVector(1.0f0, 0.0f0), equations)) == Float32
    @test eltype(Trixi.cons2entropy(u, equations)) == Float32

    callback = MonitorCallback(interval=2)
    @test callback isa Trixi.DiscreteCallback
end
