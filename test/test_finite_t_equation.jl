@testset "finite-temperature harmonic equation" begin
    equations = IsotropicHarmonicsFiniteT2D(3, 2; mass=1.0, mu0=1.0,
                                            temperature=0.05, n_quad=128)
    nvars = nvariables(equations)

    @test equations isa Trixi.AbstractEquations{2, 14}
    @test nvars == equations.n_radial * (1 + 2 * equations.n_harmonics)
    @test size(equations.Ax) == (nvars, nvars)
    @test size(equations.Ay) == (nvars, nvars)
    @test Trixi.varnames(Trixi.cons2cons, equations)[1:6] ==
          ("a0_r0", "a0_r1", "a1_r0", "a1_r1", "b1_r0", "b1_r1")

    u = SVector{14, Float64}(randn(14))
    @test Trixi.flux(u, 1, equations) ≈ equations.Ax * u
    @test Trixi.flux(u, 2, equations) ≈ equations.Ay * u
    @test Trixi.flux(u, SVector(0.6, 0.8), equations) ≈
          0.6 * equations.Ax * u + 0.8 * equations.Ay * u
    @test Trixi.have_constant_speed(equations) == Trixi.True()
    @test Trixi.max_abs_speeds(equations) == (equations.vmax, equations.vmax)
    @test Trixi.max_abs_speed_naive(u, -u, SVector(3.0, 4.0), equations) ≈
          5.0 * equations.vmax

    for gram in equations.radial_grams
        @test gram ≈ I atol=5.0e-11 rtol=5.0e-11
    end

    S = Diagonal(equations.gram_sqrt)
    Sinv = Diagonal(1 ./ equations.gram_sqrt)
    @test S * equations.Ax * Sinv ≈ (S * equations.Ax * Sinv)' atol=1.0e-10
    @test S * equations.Ay * Sinv ≈ (S * equations.Ay * Sinv)' atol=1.0e-10

    for A in (equations.Ax, equations.Ay)
        for j in 1:nvars, i in 1:nvars
            abs(A[i, j]) <= 1.0e-11 && continue
            ell_i, _, _ = FermiSea._finite_mode(i, equations.n_radial)
            ell_j, _, _ = FermiSea._finite_mode(j, equations.n_radial)
            @test abs(ell_i - ell_j) == 1
        end
    end

    c = randn(nvars)
    @test equations.moment_matrix * (equations.hydro_projector * c) ≈
          equations.moment_matrix * c atol=1.0e-10

    density_moment = reshape(equations.moment_matrix[1, :], 1, :)
    @test density_moment * (equations.density_projector * c) ≈
          density_moment * c atol=1.0e-10

    gamma_mr = 0.2
    gamma_mc = 0.7
    collision = LinearCollisionMatrix(equations; gamma_mr, gamma_mc)
    @test density_moment * (collision.W * c) ≈ zeros(1) atol=1.0e-10
    mc_part = gamma_mc .* (Matrix{Float64}(I, nvars, nvars) .-
                           equations.hydro_projector)
    @test equations.moment_matrix * (mc_part * c) ≈
          zeros(size(equations.moment_matrix, 1)) atol=1.0e-10

    normal = SVector(1.0, 0.0)
    wall = MaxwellWallBC(0.0)
    boundary = FermiSea.assemble_ghost_state(wall, u, normal, equations)
    @test dot(FermiSea.normal_flux_row(equations, normal), boundary) ≈ 0.0 atol=1.0e-10

    contact = OhmicContactBC(0.5)
    contact_boundary = FermiSea.assemble_ghost_state(contact, u, normal, equations)
    contact_cache = FermiSea.build_projector_cache(equations, normal,
                                                   FermiSea.normal_flux_row(equations,
                                                                            normal))
    contact_template = SVector{14, Float64}(ntuple(i -> i == 1 ? 0.5 : 0.0,
                                                   Val(14)))
    @test contact_boundary ≈
          u + FermiSea._apply_P_in(contact_cache, contact_template - u)
end
