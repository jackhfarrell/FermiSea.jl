@testset "steady damping" begin
    equations = IsotropicFermiHarmonics2D(2)
    source = LinearCollisionMatrix(equations, SVector(0.0, 2.0, 2.0, 2.0, 2.0))
    u0 = SVector(1.0, 1.0, -0.5, 0.25, -0.125)
    residual0 = norm(source(u0, nothing, 0.0, equations))
    u1 = SVector(u0[1], 0.005, -0.0025, 0.00125, -0.000625)
    residual1 = norm(source(u1, nothing, 0.0, equations))

    @test residual1 / residual0 < 1.0e-2
end
