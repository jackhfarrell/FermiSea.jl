@testset "equation" begin
    equations = IsotropicFermiHarmonics2D(2; v_fermi=3.0)
    u = SVector(1.0, 0.2, -0.3, 0.4, -0.5)

    @test Trixi.nvariables(equations) == 5
    @test Trixi.varnames(Trixi.cons2cons, equations) == ("a0", "a1", "b1", "a2", "b2")
    @test FermiSea._default_output_var_indices(Trixi.nvariables(equations)) == 1:3

    # flux path is the fast tridiagonal loop; matrices are the reference
    @test Trixi.flux(u, 1, equations) ≈ equations.Ax * u
    @test Trixi.flux(u, 2, equations) ≈ equations.Ay * u
    @test Trixi.flux(u, SVector(0.6, 0.8), equations) ≈
          0.6 * equations.Ax * u + 0.8 * equations.Ay * u

    @test Trixi.have_constant_speed(equations) == Trixi.True()
    @test Trixi.max_abs_speeds(u, equations) == (3.0, 3.0)
    @test Trixi.max_abs_speed_naive(u, -u, SVector(3.0, 4.0), equations) == 15.0
    @test FermiSea._transformed_speed(equations, 3.0, 4.0) == 15.0
    @test Trixi.residual_steady_state(SVector(1.0, -2.0, 0.5, 0.0, -1.5), equations) == 2.0

    @test equations.Ax[2, 1] == 3.0
    @test equations.Ax[1, 2] == 1.5
    @test equations.Ay[3, 1] == 3.0
    @test equations.Ay[1, 3] == 1.5
end
