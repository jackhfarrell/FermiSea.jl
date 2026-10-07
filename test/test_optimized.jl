@testset "packed unstructured parity and retained preconditioners" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    collision = Callaway(model; gamma_mr = 0.3, gamma_mc = 0.7)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    native_solver = DGSEM(; polydeg = 1, surface_flux = flux_upwind,
                          volume_integral = VolumeIntegralWeakForm())
    native = SemidiscretizationHyperbolic(UnstructuredMesh2D(mesh_file), model,
        Trixi.initial_condition_constant, native_solver;
        boundary_conditions = boundaries, source_terms = collision)
    native_A, native_b = Trixi.linear_structure(native)
    rng = MersenneTwister(901)
    trial = randn(rng, length(problem.b))
    @test problem.b ≈ -native_b rtol = 3e-14 atol = 3e-14
    @test problem.A * trial ≈ -(native_A * trial) rtol = 3e-14 atol = 3e-14

    reference = solve(problem, GMRES(; precond = nothing, restart = 80, maxiters = 800);
                      tolerance = 2e-10)
    @test reference.converged
    for config in (TransportSweep(; factor_storage = Float64),
                   MomentSweep(; factor_storage = Float64))
        prepared = prepare_preconditioner(problem, config)
        solution = solve(problem, GMRES(; precond = prepared, restart = 40,
                                        maxiters = 400); tolerance = 2e-9)
        @test solution.converged
        @test solution.residual < 2e-9
        @test weighted_norm(solution.phi - reference.phi, problem) <
              2e-7 * weighted_norm(reference.phi, problem)
    end
    for config in (TransportSweep(), MomentSweep())
        solution = solve(problem, GMRES(; precond = config, restart = 40,
                                        maxiters = 500); tolerance = 1e-9)
        @test solution.converged
        @test solution.residual < 1e-9
    end
    old_velocity = surface.velocities[1, 1]
    surface.velocities[1, 1] += 0.01
    @test_throws ArgumentError solve(problem, GMRES(; precond = nothing, maxiters = 0))
    surface.velocities[1, 1] = old_velocity
end

@testset "affine source sign and action-only custom fallback" begin
    mesh_file = small_unstructured_mesh(; nx = 1, ny = 1)
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.0), right = FixedReservoir(0.0),
                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0))
    drive = (u, x, t, equations) -> SVector{8}(ntuple(i ->
        (1 + 0.2x[1]) * equations.surface.velocities[i, 1], 8))
    custom = CustomCollision(model,
        (out, input, workspace) -> (@. out = 0.25 * (input - sum(input) / length(input))))
    sources = SourceTerms(drive, custom)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = sources, degree = 1)
    @test norm(problem.b) > 0
    sweep = prepare_preconditioner(problem, TransportSweep(; factor_storage = Float64))
    @test sweep.collision_passes == 0
    solution = solve(problem, GMRES(; precond = sweep, restart = 30, maxiters = 300);
                     tolerance = 2e-9)
    @test solution.converged
    @test solution.residual < 2e-9
    @test_throws ArgumentError prepare_preconditioner(problem, MomentSweep())
end
