@testset "circular local equilibrium" begin
    surface = circular_fermi_surface(; p_fermi = 1.7, v_fermi = 0.8, nangle = 16)
    remainder = 0.11 .* cos.(2 .* surface.theta)
    exact = 0.17 .+ surface.momenta * SVector(0.31, -0.22)
    phi = exact + remainder
    fit = fit_local_equilibrium(phi, surface)
    @test fit.chemical_potential ≈ 0.17 atol = 2e-14
    @test fit.drift ≈ SVector(0.31, -0.22) atol = 2e-14
    @test fit.fitted ≈ exact atol = 3e-14
    @test fit.remainder ≈ remainder atol = 3e-14
    @test momentum_moment(phi, surface, (g, i) -> g.velocities[i, 1]) ≈
          particle_current(phi, surface)[1]
    population = fit_local_equilibrium(phi, surface; drift = false)
    @test population.drift === nothing
    @test population.fitted ≈ fill(0.17, length(surface))
    @test fit_local_equilibrium(zeros(length(surface)), surface).relative_residual == 0
    @test_throws DimensionMismatch fit_local_equilibrium(zeros(3), surface)
end

@testset "plotting-independent spatial sampling and contact traces" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, mu = 0.5, nangle = 8))
    mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 1, ny = 1);
                              periodicity = false)
    boundaries = Dict(:left => FixedReservoir(0.1), :right => FixedReservoir(-0.1),
                      :wall_bottom => MaxwellWall(), :wall_top => MaxwellWall())
    problem = BoltzmannProblem(model, mesh, boundaries; degree = 1)
    phi = zeros(length(problem.b))
    native = Trixi.wrap_array_native(phi, problem.semi)
    coordinates = problem.semi.cache.elements.node_coordinates
    for element in axes(native, 4), j in axes(native, 3), i in axes(native, 2),
        variable in axes(native, 1)
        x, y = coordinates[1, i, j, element], coordinates[2, i, j, element]
        native[variable, i, j, element] = (1 + 0.2variable) * (0.3 + 0.7x - 0.4y)
    end
    solution = BoltzmannSolution(phi, problem, 0.0, true, 0, 0.0, :converged, (;), (;))
    sampler = SpatialSampler(solution, particle_density)
    raw_sampler = SpatialSampler(solution)
    point = (0.13, -0.27)
    expected_state = [(1 + 0.2variable) *
                      (0.3 + 0.7point[1] - 0.4point[2]) for variable in 1:8]
    @test raw_sampler(point...) ≈ expected_state atol = 3e-12
    @test sampler(point...) ≈ particle_density(expected_state, model.surface) atol = 3e-12
    @test isnan(sampler(3.0, 3.0))
    @test SpatialSampler(solution, particle_density; outside = :nothing)(3.0, 3.0) === nothing
    @test_throws DomainError SpatialSampler(solution, particle_density;
                                             outside = :error)(3.0, 3.0)

    solved = solve(problem, GMRES(; restart = 30, maxiters = 300); tolerance = 1e-9)
    left = contact_particle_flux(solved, :left)
    right = contact_particle_flux(solved, :right)
    @test left + right ≈ 0 atol = 2e-9
end
