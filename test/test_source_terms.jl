@testset "source terms" begin
    equations = IsotropicFermiHarmonics2D(2)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0)
    # rates: mode 0 undamped, modes 1-2 damped at 0.75
    collision = LinearCollisionMatrix(equations, SVector(0.0, 0.75, 0.75, 0.75, 0.75))
    magnetic  = MagneticFieldSource(equations, 2.0)
    sources   = SourceTerms(collision, magnetic)

    @test collision(u, nothing, 0.0, equations) == SVector(0.0, -1.5, -2.25, -3.0, -3.75)
    @test magnetic(u, nothing, 0.0, equations)  == SVector(0.0, 6.0, -4.0, 20.0, -16.0)
    @test sources(u, nothing, 0.0, equations)   == collision(u, nothing, 0.0, equations) +
          magnetic(u, nothing, 0.0, equations)
end
