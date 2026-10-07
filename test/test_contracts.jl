@testset "source composition and rate updates" begin
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8)
    model = BoltzmannEquation(surface)
    collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.7)
    drive = ElectricDrive(model, (0.1, -0.2))
    phi = SVector{8}(1:8)
    x = SVector(0.0, 0.0)
    for terms in ((), (collision,), (collision, drive))
        composed = SourceTerms(terms...)
        @test composed.terms isa Tuple
        @test composed.terms == SourceTerms(terms).terms
        prepared = prepare_source(composed)
        expected = foldl((out, term) -> out + prepare_source(term)(phi, x, 0.0, model),
                         terms; init = zero(phi))
        @test prepared(phi, x, 0.0, model) ≈ expected
        mesh = UnstructuredMesh2D(small_unstructured_mesh(; nx = 1, ny = 1))
        boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(-0.1),
                        wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
        problem = BoltzmannProblem(model, mesh, boundaries; source_terms = composed, degree = 1)
        tuple_problem = BoltzmannProblem(model, mesh, boundaries; source_terms = terms, degree = 1)
        @test problem.b ≈ tuple_problem.b
        trial = collect(1.0:length(problem.b))
        @test problem.A * trial ≈ tuple_problem.A * trial
    end
    mesh = UnstructuredMesh2D(small_unstructured_mesh())
    boundaries = (; left = FixedReservoir(0.1), right = FixedReservoir(-0.1),
                    wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    previous = nothing
    for rate in (0.02, 0.5, 10.0)
        problem = BoltzmannProblem(model, mesh, boundaries;
            source_terms = Callaway(model; gamma_mr = 0.2, gamma_mc = rate), degree = 1)
        solution = solve(problem; initial_guess = previous, tolerance = 1e-9)
        cold = solve(problem; tolerance = 1e-9)
        @test solution.converged
        @test weighted_norm(solution.phi - cold.phi, problem) < 1e-7 * weighted_norm(cold.phi, problem)
        @test abs(contact_particle_flux(solution, :left) + contact_particle_flux(solution, :right)) < 1e-9
        previous === nothing || @test_throws ArgumentError prepare_preconditioner(
            problem, prepare_preconditioner(previous.problem, MomentSweep()))
        previous = solution
    end
    @test_throws ErrorException solve(previous.problem, GMRES(; maxiters = 0))
    diagnostic = solve(previous.problem, GMRES(; maxiters = 0); throw_on_failure = false)
    @test !diagnostic.converged
end

@testset "higher-degree nodes and nested samples" begin
    solution = plotdata_solution(UnstructuredMesh2D(small_unstructured_mesh()); degree = 3)
    observable = (phi, surface) -> (; density = particle_density(phi, surface),
                                 flow = (; current = particle_current(phi, surface)))
    data = PlotData(solution, :nested => observable)
    coordinates = solution.problem.semi.cache.elements.node_coordinates
    @test data.x == vec(coordinates[1, :, :, :])
    @test data.y == vec(coordinates[2, :, :, :])
    for sampler in (SpatialSampler(solution, observable), SpatialSampler(data, :nested))
        outside = sampler(10.0, 10.0)
        @test isnan(outside.density)
        @test all(isnan, outside.flow.current)
        inside = sampler(0.3, 0.2)
        @test isfinite(inside.density)
        @test all(isfinite, inside.flow.current)
    end
    @test Docs.doc(SpatialSampler) !== nothing
    @test ndims(state(solution)) == 2
    @test Base.mightalias(state(solution), solution.phi)
end

@testset "electrical field and charged storage" begin
    mktempdir() do directory
        for charge in (-2.0, 2.0)
            surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 8, charge)
            model = BoltzmannEquation(surface)
            collision = Callaway(model; gamma_mr = 0.5, gamma_mc = 1.0)
            electric_field = 0.3
            drive = ElectricDrive(model, (electric_field, 0.0))
            @test drive.charged_field == SVector(charge * electric_field, 0.0)
            profile = solve(developed_channel(model; width = 2.0,
                walls = (MaxwellWall(), MaxwellWall()), source_terms = collision,
                electric_field); ncells = 8)
            expected = charge * electric_field / 0.5 .* surface.velocities[:, 1]
            @test profile.states ≈ repeat(expected, 1, 8) atol = 2e-12
            @test profile.charged_field == charge * electric_field
            @test_throws UndefKeywordError developed_channel(model; width = 2.0, drive = 0.3)
            mesh = UnstructuredMesh2D(small_unstructured_mesh())
            boundaries = (; left = IncomingProfile(profile), right = IncomingProfile(profile),
                            wall_bottom = MaxwellWall(), wall_top = MaxwellWall())
            problem = BoltzmannProblem(model, mesh, boundaries;
                                       source_terms = (collision, drive), degree = 1)
            solution = solve(problem; tolerance = 1e-9)
            filename = joinpath(directory, "charge$(charge).h5")
            save_solution(solution, filename)
            loaded = read_solution(filename)
            @test loaded.phi == solution.phi
            @test loaded.problem.source_terms[2].charged_field == drive.charged_field
            @test loaded.problem.boundary_conditions.left.profile.charged_field == profile.charged_field
            @test loaded.problem.b ≈ problem.b atol = 2e-14
            @test loaded.problem.A * solution.phi ≈ problem.A * solution.phi atol = 2e-14
        end
    end
end

@testset "Makie dispatch remains local" begin
    figure = Figure()
    axis = Axis(figure[1, 1])
    plot!(axis, [0.0, 1.0], [1.0, 0.0])
    Makie.colorbuffer(figure)
    extension = Base.get_extension(FermiSea, :FermiSeaMakieExt)
    owned_types = Union{BoltzmannSolution,PlotData,CircularFermiSurface}
    for function_name in (:plot, :plot!, :streamplot!)
        for method in methods(getproperty(Makie, function_name))
            method.module === extension || continue
            positional_types = Base.unwrap_unionall(method.sig).parameters[2:end]
            @test any(type -> type <: owned_types, positional_types)
        end
    end
end
