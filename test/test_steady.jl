function small_problem(left_bias, right_bias; refinement = 1)
    model = BoltzmannEquation(circular_fermi_surface(;
        p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    collision = Callaway(model; gamma_mr = 0.6, gamma_mc = 1.0)
    mesh = TreeMesh((-0.5, -0.5), (0.5, 0.5);
                    initial_refinement_level = refinement, n_cells_max = 100)
    boundaries = (; x_neg = FixedReservoir(left_bias),
                    x_pos = FixedReservoir(right_bias),
                    y_neg = DiffuseWall(), y_pos = DiffuseWall())
    return BoltzmannProblem(model, mesh, boundaries; source_terms = collision, degree = 1)
end

@testset "magnetic steady solve" begin
    magnetic_model = BoltzmannEquation(circular_fermi_surface(;
        p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, charge = -1.0, nangle = 8))
    gamma, field = 0.7, 0.2
    magnetic_collision = Callaway(magnetic_model; gamma_mr = gamma, gamma_mc = 1.1)
    magnetic = MagneticField(magnetic_model, field)
    theta = magnetic_model.surface.theta
    drive = (u, x, t, equations) -> SVector{8}(cos.(theta))
    periodic_mesh = TreeMesh((-0.5, -0.5), (0.5, 0.5); periodicity = true,
                             initial_refinement_level = 0, n_cells_max = 10)
    magnetic_problem = BoltzmannProblem(magnetic_model, periodic_mesh,
        Trixi.boundary_condition_periodic;
        source_terms = (magnetic_collision, magnetic, drive), degree = 1)
    magnetic_solution = solve(magnetic_problem, GMRES(; restart = 40, maxiters = 500);
                              tolerance = 1e-10)
    expected = @. (gamma * cos(theta) + field * sin(theta)) / (gamma^2 + field^2)
    @test magnetic_solution.converged
    @test magnetic_solution.residual < 1e-10
    @test state(magnetic_solution) ≈ repeat(expected, 1, size(state(magnetic_solution), 2))
          atol = 4e-13
end

@testset "steady vertical slice" begin
    driven = small_problem(0.15, -0.05)
    solution = solve(driven, GMRES(; restart = 30, maxiters = 500); tolerance = 1e-9)
    @test solution.converged
    @test solution.residual <= 1e-9
    @test size(state(solution), 1) == length(driven.semi.equations.surface)
    @test maximum(state(solution)) - minimum(state(solution)) > 1e-3

    left_flux = contact_particle_flux(solution, :x_neg)
    right_flux = contact_particle_flux(solution, :x_pos)
    @test abs(left_flux + right_flux) < 2e-8
    @test left_flux * right_flux < 0

    # Wrapper A is exactly the sign-reversed homogeneous part of Trixi's RHS
    Random.seed!(104)
    probe = randn(length(driven.b))
    direct = similar(probe)
    zero_rhs = similar(probe)
    Trixi.rhs!(direct, probe, driven.semi, 0.0)
    Trixi.rhs!(zero_rhs, zero(probe), driven.semi, 0.0)
    wrapper = driven.A * probe
    @test wrapper ≈ -(direct - zero_rhs) rtol = 2e-13 atol = 2e-13

    equilibrium = small_problem(0.2, 0.2)
    constant_phi = fill(0.2, length(equilibrium.b))
    @test weighted_norm(equilibrium.A * constant_phi - equilibrium.b, equilibrium) < 2e-13
    equilibrium_solution = solve(equilibrium; tolerance = 1e-10)
    @test equilibrium_solution.converged
    @test state(equilibrium_solution) ≈ fill(0.2, size(state(equilibrium_solution))) atol = 2e-10

    failed = solve(driven, GMRES(; restart = 1, maxiters = 1);
                   tolerance = 1e-14, throw_on_failure = false)
    @test !failed.converged
    @test failed.status == :maxiters
    @test_throws ErrorException solve(driven, GMRES(; restart = 1, maxiters = 1);
                                      tolerance = 1e-14, throw_on_failure = true)
end

@testset "physical measures and ballistic current" begin
    model = BoltzmannEquation(circular_fermi_surface(;
        p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    height = 2.0
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0); periodicity = (false, true),
                    initial_refinement_level = 1, n_cells_max = 100)
    left_bias, right_bias = 0.3, -0.1
    boundaries = (; x_neg = FixedReservoir(left_bias),
                    x_pos = FixedReservoir(right_bias))
    ballistic = BoltzmannProblem(model, mesh, boundaries; degree = 1)
    @test sum(ballistic.spatial_mass) ≈ 4.0 atol = 2e-14

    solution = solve(ballistic, GMRES(; restart = 30, maxiters = 300);
                     tolerance = 1e-10)
    @test solution.converged
    surface = model.surface
    positive_flux_measure = sum(surface.weights[i] * surface.velocities[i, 1]
                                for i in eachindex(surface.weights)
                                if surface.velocities[i, 1] > 0)
    expected = height * positive_flux_measure * (left_bias - right_bias)
    @test contact_particle_flux(solution, :x_pos) ≈ expected rtol = 2e-10 atol = 2e-12
    @test contact_particle_flux(solution, :x_neg) ≈ -expected rtol = 2e-10 atol = 2e-12
end
