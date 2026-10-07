using StaticArrays

function plotdata_solution(mesh; degree = 2)
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0,
        v_fermi = 1.0, nangle = 8))
    boundaries = mesh isa TreeMesh ?
        (; x_neg = FixedReservoir(0.1), x_pos = FixedReservoir(-0.1),
           y_neg = DiffuseWall(), y_pos = DiffuseWall()) :
        (; left = FixedReservoir(0.1), right = FixedReservoir(-0.1),
           wall_bottom = DiffuseWall(), wall_top = DiffuseWall())
    problem = BoltzmannProblem(model, mesh, boundaries; degree)
    return solve(problem, GMRES(; restart = 40, maxiters = 400);
                 tolerance = 1e-8, throw_on_failure = true)
end

@testset "nodal PlotData" begin
    for mesh in (TreeMesh((-0.5, -0.5), (0.5, 0.5); initial_refinement_level = 1,
                         n_cells_max = 100),
                 UnstructuredMesh2D(small_unstructured_mesh(; nx = 2, ny = 2)))
        solution = plotdata_solution(mesh)
        data = PlotData(solution, particle_density, particle_current)
        coords = solution.problem.semi.cache.elements.node_coordinates
        @test data.x == vec(coords[1, :, :, :])
        @test data.y == vec(coords[2, :, :, :])
        u = state(solution)
        direct = [particle_density(view(u, :, i, j, e), solution.problem.semi.equations.surface)
                  for e in axes(u, 4), j in axes(u, 3), i in axes(u, 2)]
        @test data.values[:particle_density] == vec(direct)
        @test data.values[:particle_current] isa Tuple

        regridded = PlotData(data; refine = 2)
        @test regridded.nodal === data.nodal
        @test regridded.refine == 2
        @test size(regridded.triangles, 2) == 2 * size(coords, 4) *
              (2 * (size(coords, 2) - 1))^2
        stored_sampler = SpatialSampler(data, :particle_density)
        current_sampler = SpatialSampler(data, :particle_current)
        for e in axes(coords, 4)
            # Sample element interiors. Element interfaces can be discontinuous
            i, j = 2, 2
            x, y = coords[1, i, j, e], coords[2, i, j, e]
            @test stored_sampler(x, y) ≈ data.nodal.values[:particle_density][i, j, e]
            @test current_sampler(x, y) isa SVector{2,Float64}
        end
        @test all(isnan, current_sampler(10.0, 10.0))

        refined = PlotData(solution, particle_density; refine = 3)
        n = length(solution.problem.semi.solver.basis.nodes)
        m = 3 * (n - 1) + 1
        ne = size(coords, 4)
        @test length(refined.x) == ne * m^2
        @test size(refined.triangles, 2) == 2 * ne * (m - 1)^2
        @test minimum(refined.triangles) >= 1
        @test maximum(refined.triangles) <= length(refined.x)
        for e in 1:ne
            inds = ((e - 1) * m^2 + 1):(e * m^2)
            triinds = findall(t -> all(in(inds), refined.triangles[:, t]), axes(refined.triangles, 2))
            @test length(triinds) == 2 * (m - 1)^2
        end
        sampler = SpatialSampler(solution, particle_density)
        for e in 1:ne, j in 2:(m-1), i in 2:(m-1)
            k = (e - 1) * m^2 + (j - 1) * m + i
            @test refined.values[:particle_density][k] ≈ sampler(refined.x[k], refined.y[k]) atol = 1e-10
        end

        quadratic = :quadratic => ((phi, surface) -> sum(abs2, phi))
        nonlinear = PlotData(solution, quadratic; refine = 3)
        nonlinear_sampler = SpatialSampler(solution, last(quadratic))
        k = 2 * m + 2
        @test nonlinear.values[:quadratic][k] ≈ nonlinear_sampler(nonlinear.x[k], nonlinear.y[k]) atol = 1e-10
    end

    solution = plotdata_solution(TreeMesh((-0.5, -0.5), (0.5, 0.5);
        initial_refinement_level = 1, n_cells_max = 100))
    paired = PlotData(solution, :pair => ((phi, surface) ->
        (; a = particle_density(phi, surface), b = particle_current(phi, surface))))
    @test paired.values[:pair] isa NamedTuple
    pair_sampler = SpatialSampler(paired, (:pair, :a))
    @test pair_sampler(0.0, 0.0) isa Real
    @test_throws ArgumentError PlotData(solution, phi -> sum(phi))
    @test_throws ArgumentError PlotData(solution, :bad => ((phi, surface) -> Dict(:x => 1)))
    @test_throws ArgumentError PlotData(solution, :same => particle_density,
                                        :same => particle_current)

    tree_data = PlotData(solution)
    outlines = FermiSea.outline_segments(tree_data)
    @test !isempty(outlines)
    diameter = hypot(extrema(tree_data.nodal.x)[2] - extrema(tree_data.nodal.x)[1],
                     extrema(tree_data.nodal.y)[2] - extrema(tree_data.nodal.y)[1])
    endpoint_counts = Dict{Tuple{Int,Int},Int}()
    endpoint_key(p) = (round(Int, p[1] / (1e-9 * diameter)),
                       round(Int, p[2] / (1e-9 * diameter)))
    for outline in outlines, endpoint in (first(outline), last(outline))
        key = endpoint_key(endpoint)
        endpoint_counts[key] = get(endpoint_counts, key, 0) + 1
    end
    @test all(==(2), values(endpoint_counts))
    length_outline = sum(hypot(outline[i+1][1] - outline[i][1],
                               outline[i+1][2] - outline[i][2])
                         for outline in outlines for i in 1:(length(outline)-1))
    @test length_outline ≈ 4 atol = 1e-10
end
