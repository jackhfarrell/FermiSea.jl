@testset "linear collision matrix constructors" begin
    equations = IsotropicFermiHarmonics2D(3)
    rates = SVector(0.0, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3)
    W = Diagonal(rates)

    @test LinearCollisionMatrix(equations, Matrix(W)).W == Matrix(W)
    @test diag(LinearCollisionMatrix(equations, rates).W) == collect(rates)

    @test_throws ArgumentError LinearCollisionMatrix(equations, zeros(6, 6))
    @test_throws ArgumentError LinearCollisionMatrix(equations, zeros(6))

    @test diag(LinearCollisionMatrix(equations; gamma_mr=0.1, gamma_mc=0.2).W) ≈
          collect(rates)
    @test diag(LinearCollisionMatrix(equations; gamma_mr=0.1, gamma_mc=0.2,
                                     gamma_3=0.2).W) ≈ collect(rates)

    third_channel_rates = SVector(0.0, 0.1, 0.1, 0.3, 0.3,
                                  0.1 + min(0.05 * 3^4 / 81, 0.2),
                                  0.1 + min(0.05 * 3^4 / 81, 0.2))
    @test diag(LinearCollisionMatrix(equations; gamma_mr=0.1, gamma_mc=0.2,
                                     gamma_3=0.05).W) ≈ collect(third_channel_rates)

    @test_throws ArgumentError LinearCollisionMatrix(equations; gamma_mr=-0.1,
                                                     gamma_mc=0.2)
    @test_throws ArgumentError LinearCollisionMatrix(equations; gamma_mr=0.1,
                                                     gamma_mc=-0.2)
    @test_throws ArgumentError LinearCollisionMatrix(equations; gamma_mr=0.1,
                                                     gamma_mc=0.2,
                                                     gamma_3=-0.3)
end
