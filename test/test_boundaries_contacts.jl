@testset "Maxwell wall and developed lead" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 16))
    surface = model.surface
    normal = SVector(cos(0.173), sin(0.173))
    Random.seed!(401)
    phi = randn(length(surface))
    reflected = FermiSea.boundary_state(MaxwellWall(), phi, normal, model)
    speed = surface.velocities * normal
    @test dot(surface.weights, speed .* reflected) ≈ 0 atol = 2e-14
    @test FermiSea.boundary_state(MaxwellWall(; p_scatter = 0.63), ones(16),
                                  normal, model) ≈ ones(16) atol = 2e-13
    @test MaxwellWall().p_scatter == 0
    for p_scatter in (-0.1, 1.1, Inf, NaN)
        @test_throws ArgumentError MaxwellWall(; p_scatter)
    end
    diffuse = FermiSea.boundary_state(DiffuseWall(), phi, normal, model)
    for p_scatter in (0.0, 0.63, 1.0)
        wall = MaxwellWall(; p_scatter)
        for boundary in (wall, FermiSea.prepare_boundary(wall))
            @test FermiSea.boundary_state(boundary, phi, normal, model) ≈
                  (1 - p_scatter) .* reflected .+ p_scatter .* diffuse atol = 2e-13
        end
    end

    collision = Callaway(model; gamma_mr = 0.5, gamma_mc = 1.3)
    profile = solve(developed_channel(model; width = 0.23,
        axis = (0.0, 1.0), walls = (MaxwellWall(), MaxwellWall()),
        source_terms = collision, electric_field = 0.6 / model.surface.charge); ncells = 8)
    expected = @. (0.6 / 0.5) * surface.velocities[:, 2]
    @test profile.states ≈ repeat(expected, 1, size(profile.states, 2)) rtol = 2e-12 atol = 2e-12
    @test maximum(abs, profile.wall_particle_flux) < 2e-13
    @test abs(channel_current(profile; component = :transverse)) < 2e-13
end

@testset "circular wall flux maps" begin
    for nangle in (4, 8, 16), angle in (0.0, 0.173, pi / 4, pi / 2)
        surface = circular_fermi_surface(; p_fermi = 1.2, v_fermi = 0.8, nangle)
        model = BoltzmannEquation(surface)
        normal = SVector(cos(angle), sin(angle))
        phi = randn(MersenneTwister(nangle), nangle)
        for wall in (DiffuseWall(), MaxwellWall(), MaxwellWall(; p_scatter = 0.63),
                     MaxwellWall(; p_scatter = 1))
            for boundary in (wall, FermiSea.prepare_boundary(wall))
                reflected = FermiSea.boundary_state(boundary, phi, normal, model)
                @test dot(surface.weights, (surface.velocities * normal) .* reflected) ≈
                      0 atol = 3e-14
                @test FermiSea.boundary_state(boundary, ones(nangle), normal, model) ≈
                      ones(nangle) atol = 3e-13
                @test FermiSea.boundary_state(boundary, phi, 2 .* normal, model) ≈
                      reflected atol = 3e-13
            end
        end
    end
end

@testset "floating contact and prepared moment walls" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0); initial_refinement_level = 0,
                    n_cells_max = 10)
    boundaries = (; x_neg = FixedReservoir(0.2), x_pos = FloatingContact(),
                    y_neg = DiffuseWall(), y_pos = DiffuseWall())
    solution = solve(BoltzmannProblem(model, mesh, boundaries; degree = 1);
                     tolerance = 1e-10, current_tolerance = 1e-11)
    @test solution.converged
    @test solution.contact_offsets.x_pos ≈ 0.2 atol = 2e-12
    @test abs(solution.contact_currents.x_pos) < 1e-11

    all_floating = (; x_neg = FloatingContact(), x_pos = FloatingContact(),
                      y_neg = DiffuseWall(), y_pos = DiffuseWall())
    problem = BoltzmannProblem(model, mesh, all_floating; degree = 1)
    @test_throws ArgumentError solve(problem; tolerance = 1e-9)

    single_floating = (; x_neg = FloatingContact(), x_pos = DiffuseWall(),
                         y_neg = DiffuseWall(), y_pos = DiffuseWall())
    single_problem = BoltzmannProblem(model, mesh, single_floating; degree = 1)
    single_solution = solve(single_problem; reference = :x_neg)
    @test single_solution.converged
    @test single_solution.iterations == 0
    @test single_solution.contact_offsets.x_neg == 0
    @test single_solution.contact_currents.x_neg == 0

    unstructured = UnstructuredMesh2D(small_unstructured_mesh(; nx = 1, ny = 1);
                                      periodicity = false)
    uboundaries = Dict(:left => FixedReservoir(0.1), :right => FixedReservoir(-0.1),
                       :wall_bottom => MaxwellWall(), :wall_top => MaxwellWall())
    uproblem = BoltzmannProblem(model, unstructured, uboundaries; degree = 1)
    usolution = solve(uproblem, GMRES(; restart = 30, maxiters = 300,
        precond = TransportSweep(; factor_storage = Float64));
        tolerance = 1e-9)
    @test usolution.converged
    @test contact_particle_flux(usolution, :left) +
          contact_particle_flux(usolution, :right) ≈ 0 atol = 2e-9
    @test abs(contact_particle_flux(usolution, :wall_bottom)) < 2e-9
    @test abs(contact_particle_flux(usolution, :wall_top)) < 2e-9
end

@testset "developed profile packed and native parity" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    collision = Callaway(model; gamma_mr = 0.4, gamma_mc = 1.1)
    magnetic = MagneticField(model, 0.12)
    qE = 0.18
    lead = solve(developed_channel(model; width = 2.0, axis = (1.0, 0.0),
        walls = (DiffuseWall(), DiffuseWall()), source_terms = (collision, magnetic),
        electric_field = qE / model.surface.charge); ncells = 8)
    @test maximum(abs, lead.states[:, 1] .- lead.states[:, 4]) > 1e-5
    profile = IncomingProfile(lead)

    surface = model.surface
    incompatible_surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0,
                                               nangle = 8, mu = 0.5, spin_degeneracy = 2)
    incompatible_model = BoltzmannEquation(incompatible_surface)
    @test_throws ArgumentError FermiSea.fill_boundary_state!(
        similar(surface.weights), profile, zeros(length(surface)), (-1.0, 0.0),
        incompatible_model, (0.0, 0.0))
    endpoint = (0.0, 1.0 + 4eps(Float64))
    @test_nowarn FermiSea.fill_boundary_state!(similar(surface.weights), profile,
        zeros(length(surface)), (1.0, 0.0), model, endpoint)
    @test_nowarn FermiSea.fill_boundary_state!(similar(surface.weights), profile,
        zeros(length(surface)), (-1.0, 0.0), model, endpoint)
    @test_throws ArgumentError FermiSea.fill_boundary_state!(similar(surface.weights),
        profile, zeros(length(surface)), (1.0, 1.0), model, (0.0, 0.0))
    mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 1, ny = 1);
                              periodicity = false)
    boundaries = Dict(:left => profile, :right => profile,
                       :wall_bottom => DiffuseWall(), :wall_top => DiffuseWall())
    electric = ElectricDrive(model, (qE / model.surface.charge, 0.0))
    problem = BoltzmannProblem(model, mesh, boundaries;
        source_terms = (collision, magnetic, electric), degree = 1)
    direct_solver = DGSEM(; polydeg = 1, surface_flux = flux_upwind,
                          volume_integral = VolumeIntegralWeakForm())
    direct = SemidiscretizationHyperbolic(mesh, model, Trixi.initial_condition_constant,
        direct_solver; boundary_conditions = problem.boundary_conditions,
        source_terms = problem.semi.source_terms)
    Random.seed!(403)
    probe = randn(length(problem.b))
    packed_rhs, native_rhs = similar(probe), similar(probe)
    Trixi.rhs!(packed_rhs, probe, problem.semi, 0.0)
    Trixi.rhs!(native_rhs, probe, direct, 0.0)
    @test packed_rhs ≈ native_rhs rtol = 3e-13 atol = 3e-13
    solution = solve(problem, GMRES(; restart = 40, maxiters = 500); tolerance = 1e-9)
    @test solution.converged
    @test contact_particle_flux(solution, :left) +
          contact_particle_flux(solution, :right) ≈ 0 atol = 2e-8
    @test abs(contact_particle_flux(solution, :wall_bottom)) < 2e-8
    @test abs(contact_particle_flux(solution, :wall_top)) < 2e-8

    incompatible = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    incompatible.surface.momenta[1, 1] += 1e-3
    @test_throws ArgumentError FermiSea.boundary_state(
        profile, zeros(8), SVector(-1.0, 0.0), incompatible, SVector(-1.0, 0.0), 0.0)
end
