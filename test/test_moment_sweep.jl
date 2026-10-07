function prolonged_kinetic(beta, preconditioner)
    phi = zeros(preconditioner.momentum_node_count * preconditioner.spatial_node_count)
    FermiSea.prolong_add_moments!(phi, beta, preconditioner)
    return phi
end

function restricted_moments(phi, preconditioner)
    beta = zeros(preconditioner.moment_count * preconditioner.spatial_node_count)
    FermiSea.restrict_moments!(beta, phi, preconditioner)
    return beta
end

@testset "batched moment assembly" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0,
                                                    v_fermi = 1.0, nangle = 8))
    mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 4, ny = 3))
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = MaxwellWall())
    sources = SourceTerms(Callaway(model; gamma_mr = 0.0, gamma_mc = 100.0),
                          MagneticField(model, 0.3), ElectricDrive(model, (0.1, 0.0)))
    problem = BoltzmannProblem(model, mesh, boundaries; source_terms = sources, degree = 3)
    precond = prepare_preconditioner(problem, MomentSweep())
    groups, neighborhoods = FermiSea.moment_element_groups(problem.semi)
    @test any(group -> length(group) > 1, groups)
    for group in groups
        touched = vcat(neighborhoods[group]...)
        @test length(unique(touched)) == length(touched)
    end
    beta = randn(MersenneTwister(1204), precond.moment_count * precond.spatial_node_count)
    restricted = restricted_moments(problem.A * prolonged_kinetic(beta, precond), precond)
    @test precond.factorization \ restricted ≈ beta rtol = 2e-10 atol = 2e-10
    release_preconditioner!(precond)
end

@testset "degree-three moment sweep across collision regimes" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0,
                                                    v_fermi = 1.0, nangle = 16))
    mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 2, ny = 2))
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    for (mr, mc) in ((0.0, 0.0), (0.01, 0.01), (0.0, 1.0),
                    (1.0, 0.0), (1.0, 1.0), (1000.0, 0.0),
                    (0.0, 100.0), (0.0, 10000.0), (100.0, 100.0))
        problem = BoltzmannProblem(model, mesh, boundaries;
            source_terms = Callaway(model; gamma_mr = mr, gamma_mc = mc), degree = 3)
        # Dense kinetic solves check that the reduced correction changes only convergence
        reference = Matrix(problem.A) \ problem.b
        solution = solve(problem; tolerance = 1e-9)
        @test solution.converged
        @test solution.iterations <= 80
        @test weighted_norm(problem.A * solution.phi - problem.b, problem) /
              weighted_norm(problem.b, problem) < 1e-9
        @test weighted_norm(solution.phi - reference, problem) /
              weighted_norm(reference, problem) < 2e-6
        left = contact_particle_flux(solution, :left)
        right = contact_particle_flux(solution, :right)
        @test abs(left + right) < 2e-6 * max(abs(left), abs(right))
    end
end

@testset "stress correction across spatial degrees" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.7,
                                                    v_fermi = 0.8, nangle = 16))
    mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 2, ny = 2))
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    for degree in (1, 2, 4)
        problem = BoltzmannProblem(model, mesh, boundaries;
            source_terms = Callaway(model; gamma_mr = 0.0, gamma_mc = 1000.0), degree)
        reference = Matrix(problem.A) \ problem.b
        solution = solve(problem; tolerance = 1e-9)
        @test solution.converged
        @test solution.iterations <= 80
        @test weighted_norm(solution.phi - reference, problem) /
              weighted_norm(reference, problem) < 2e-6
    end
end

function moment_operator_action(preconditioner, beta)
    work = preconditioner.workspace
    copyto!(work.moment_rhs, beta)
    mul!(work.moment_solution, preconditioner.factorization, work.moment_rhs)
    return work.moment_solution
end

@testset "moment restriction identity" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.5)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file),
                               (; left = FixedReservoir(0.0), right = FixedReservoir(0.0),
                                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0));
                               source_terms = collision, degree = 1)
    precond = prepare_preconditioner(problem, MomentSweep(; factor_storage = Float64))
    rng = MersenneTwister(1201)
    beta = randn(rng, precond.moment_count * precond.spatial_node_count)
    phi = prolonged_kinetic(beta, precond)
    recovered = restricted_moments(phi, precond)
    @test recovered ≈ beta rtol = 1e-13 atol = 1e-13
end

@testset "moment operator matches fine restriction" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = FixedReservoir(0.0))
    collision = Callaway(model; gamma_mr = 0.0, gamma_mc = 0.4)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    precond = prepare_preconditioner(problem, MomentSweep(; factor_storage = Float64))
    modes = precond.modes
    weighted = precond.weighted_modes
    S_matrix = FermiSea.moment_restricted_operator(problem, modes, weighted, precond.momentum_node_count,
                                                   precond.moment_count, precond.spatial_node_count)
    rng = MersenneTwister(1202)
    for _ in 1:4
        beta = randn(rng, precond.moment_count * precond.spatial_node_count)
        phi = prolonged_kinetic(beta, precond)
        restricted = restricted_moments(problem.A * phi, precond)
        @test S_matrix * beta ≈ restricted rtol = 5e-11 atol = 5e-11
        @test precond.factorization \ restricted ≈ beta rtol = 5e-11 atol = 5e-11
    end
    for alpha in 1:precond.moment_count
        beta = zeros(precond.moment_count * precond.spatial_node_count)
        for node in 1:precond.spatial_node_count
            beta[(node - 1) * precond.moment_count + alpha] = 1.0
        end
        phi = prolonged_kinetic(beta, precond)
        @test S_matrix * beta ≈ restricted_moments(problem.A * phi, precond) rtol = 5e-11
    end
end

@testset "collisionless moment sweep matches transport sweep" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0))
    collision = Callaway(model; gamma_mr = 0.0, gamma_mc = 0.0)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    transport = prepare_preconditioner(problem, TransportSweep(;
        factor_storage = Float64, collision_passes = 0))
    moment = prepare_preconditioner(problem, MomentSweep(; factor_storage = Float64))
    rng = MersenneTwister(1203)
    rhs = randn(rng, length(problem.b))
    y_transport = similar(rhs)
    y_moment = similar(rhs)
    ldiv!(y_transport, transport, rhs)
    ldiv!(y_moment, moment, rhs)
    @test y_moment ≈ y_transport rtol = 1e-12 atol = 1e-12
    reference = solve(problem, GMRES(; precond = nothing, restart = 60, maxiters = 600);
                      tolerance = 1e-10)
    @test reference.converged
    for precond in (transport, moment)
        solution = solve(problem, GMRES(; precond, restart = 40, maxiters = 400);
                         tolerance = 1e-9)
        @test solution.converged
        @test weighted_norm(solution.phi - reference.phi, problem) <
              2e-7 * max(weighted_norm(reference.phi, problem), 1.0)
    end
end

@testset "moment sweep gmres accuracy" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    collision = Callaway(model; gamma_mr = 0.3, gamma_mc = 0.7)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    reference = solve(problem, GMRES(; precond = nothing, restart = 80, maxiters = 800);
                      tolerance = 2e-10)
    @test reference.converged
    for config in (MomentSweep(; factor_storage = Float64),
                   TransportSweep(; factor_storage = Float64, collision_passes = 1))
        prepared = prepare_preconditioner(problem, config)
        solution = solve(problem, GMRES(; precond = prepared, restart = 40, maxiters = 400);
                         tolerance = 2e-9)
        @test solution.converged
        @test weighted_norm(solution.phi - reference.phi, problem) <
              2e-7 * max(weighted_norm(reference.phi, problem), 1.0)
    end
end

@testset "hydrodynamic moment sweep convergence" begin
    mesh_file = small_unstructured_mesh(; nx = 2, ny = 2)
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.15), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = FixedReservoir(0.05))
    collision = Callaway(model; gamma_mr = 50.0, gamma_mc = 50.0)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    reference = solve(problem, GMRES(; precond = nothing, restart = 80, maxiters = 1200);
                      tolerance = 2e-10)
    @test reference.converged
    unprepared = solve(problem, GMRES(; precond = nothing, restart = 40, maxiters = 400);
                       tolerance = 2e-9)
    moment = solve(problem, GMRES(; precond = MomentSweep(; factor_storage = Float64),
                                  restart = 40, maxiters = 400); tolerance = 2e-9)
    @test moment.converged
    @test weighted_norm(moment.phi - reference.phi, problem) <
          2e-7 * max(weighted_norm(reference.phi, problem), 1.0)
    @test moment.iterations < unprepared.iterations
    @test moment.iterations < 0.5 * unprepared.iterations + 5
    modest = Callaway(model; gamma_mr = 5.0, gamma_mc = 5.0)
    modest_problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                                      source_terms = modest, degree = 1)
    modest_moment = solve(modest_problem,
                          GMRES(; precond = MomentSweep(), restart = 40, maxiters = 400);
                          tolerance = 2e-9)
    modest_unprepared = solve(modest_problem, GMRES(; precond = nothing, restart = 40,
                                                    maxiters = 400); tolerance = 2e-9)
    @test modest_moment.converged
    @test modest_moment.iterations <= modest_unprepared.iterations
end

@testset "moment sweep without momentum relaxation" begin
    mesh_file = small_unstructured_mesh()
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(0.0),
                  wall_bottom = DiffuseWall(), wall_top = FixedReservoir(0.0))
    collision = Callaway(model; gamma_mr = 0.0, gamma_mc = 80.0)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    reference = solve(problem, GMRES(; precond = nothing, restart = 80, maxiters = 800);
                      tolerance = 2e-10)
    @test reference.converged
    solution = solve(problem, GMRES(; precond = MomentSweep(; factor_storage = Float64),
                                    restart = 40, maxiters = 500); tolerance = 2e-9)
    @test solution.converged
    @test weighted_norm(solution.phi - reference.phi, problem) <
          2e-7 * max(weighted_norm(reference.phi, problem), 1.0)
    @test contact_particle_flux(solution, :left) ≈
          contact_particle_flux(reference, :left) rtol = 2e-7 atol = 2e-10
end

@testset "zero temperature moment columns" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    mesh_file = small_unstructured_mesh(; nx = 1, ny = 1)
    collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.6)
    @test size(collision.momentum.modes, 2) == 3
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file),
                               (; left = FixedReservoir(0.0), right = FixedReservoir(0.0),
                                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0));
                               source_terms = collision, degree = 1)
    @test_nowarn prepare_preconditioner(problem, MomentSweep())
end

@testset "moment sweep source and stale guards" begin
    mesh_file = small_unstructured_mesh(; nx = 1, ny = 1)
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.0), right = FixedReservoir(0.0),
                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0))
    custom = CustomCollision(model,
        (out, input, workspace) -> (@. out = 0.2 * input))
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = custom, degree = 1)
    @test_throws ArgumentError prepare_preconditioner(problem, MomentSweep())
    tomographic = TomographicCollision(model; gamma_mr = 0.2, gamma_mc = 0.5, gamma_3 = 0.1)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = tomographic, degree = 1)
    @test_throws ArgumentError prepare_preconditioner(problem, MomentSweep())
    collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.5)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    prepared = prepare_preconditioner(problem, MomentSweep())
    surface.velocities[1, 1] += 0.01
    @test_throws ArgumentError prepare_preconditioner(problem, prepared)
end

@testset "automatic preconditioner selects moment sweep" begin
    mesh_file = small_unstructured_mesh(; nx = 1, ny = 1)
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    boundaries = (; left = FixedReservoir(0.0), right = FixedReservoir(0.0),
                  wall_bottom = FixedReservoir(0.0), wall_top = FixedReservoir(0.0))
    collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.5)
    problem = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                               source_terms = collision, degree = 1)
    automatic = prepare_preconditioner(problem, AutomaticPreconditioner())
    @test typeof(automatic) <: FermiSea.MomentSweepPreconditioner
    @test automatic.moment_count == 5
    weak = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
        source_terms = Callaway(model; gamma_mr = 0.01, gamma_mc = 0.01), degree = 3)
    ballistic = prepare_preconditioner(weak, AutomaticPreconditioner())
    @test typeof(ballistic) <: FermiSea.TransportSweepPreconditioner
    drive = ElectricDrive(model, (0.1, 0.0))
    custom = CustomCollision(model, (out, input, workspace) -> (@. out = 0.1 * input))
    affine = BoltzmannProblem(model, UnstructuredMesh2D(mesh_file), boundaries;
                              source_terms = SourceTerms(drive, custom), degree = 1)
    fallback = prepare_preconditioner(affine, AutomaticPreconditioner())
    @test typeof(fallback) <: FermiSea.TransportSweepPreconditioner
end
