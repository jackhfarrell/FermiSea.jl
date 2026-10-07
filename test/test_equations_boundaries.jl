@testset "direct Trixi equations and boundaries" begin
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 2.0, nangle = 12)
    model = BoltzmannEquation(surface)
    Random.seed!(103)
    phi = SVector{length(surface)}(randn(length(surface)))
    normal = SVector(0.3, -0.8)
    expected = @. (normal[1] * surface.velocities[:, 1] +
                   normal[2] * surface.velocities[:, 2]) * phi
    @test Trixi.flux(phi, normal, model) ≈ expected

    left = SVector{length(surface)}(randn(length(surface)))
    right = SVector{length(surface)}(randn(length(surface)))
    flux = flux_upwind(left, right, normal, model)
    speed = @. normal[1] * surface.velocities[:, 1] + normal[2] * surface.velocities[:, 2]
    @test flux ≈ speed .* ifelse.(speed .>= 0, left, right)

    wall_state = FermiSea.diffuse_state(DiffuseWall(), phi, normal, model)
    @test dot(surface.weights, speed .* wall_state) ≈ 0 atol = 3e-16
    @test FermiSea.diffuse_state(DiffuseWall(), ones(length(surface)), normal, model) ≈
          ones(length(surface))
end


@testset "fresh direct Trixi construction" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    collision = Callaway(model; gamma_mr = 0.3, gamma_mc = 0.8)
    mesh = TreeMesh((-0.5, -0.5), (0.5, 0.5); initial_refinement_level = 0,
                    n_cells_max = 10)
    boundaries = (; x_neg = FixedReservoir(0.1), x_pos = FixedReservoir(-0.1),
                    y_neg = DiffuseWall(), y_pos = DiffuseWall())
    problem = BoltzmannProblem(model, mesh, boundaries; source_terms = collision, degree = 1)

    direct_solver = DGSEM(; polydeg = 1, surface_flux = flux_upwind,
                          volume_integral = VolumeIntegralWeakForm())
    direct_semi = SemidiscretizationHyperbolic(
        mesh, model, Trixi.initial_condition_constant, direct_solver;
        boundary_conditions = boundaries, source_terms = collision)
    direct_time_operator, direct_time_drive = Trixi.linear_structure(direct_semi)
    Random.seed!(105)
    probe = randn(length(problem.b))
    @test problem.A * probe ≈ -(direct_time_operator * probe) rtol = 2e-13 atol = 2e-13
    @test problem.b ≈ -direct_time_drive rtol = 2e-13 atol = 2e-13
end
