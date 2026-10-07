using Test
using FermiSea

@testset "MaxwellWall HDF5 I/O" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1.0,
                                                     v_fermi = 1.0, nangle = 8))
    mktemp() do path, io
        close(io)
        FermiSea.HDF5.h5open(path, "w") do file
            for p_scatter in (0.0, 0.63, 1.0)
                wall = MaxwellWall(; p_scatter)
                for (name, boundary) in (("plain", wall),
                                         ("prepared", FermiSea.prepare_boundary(wall)))
                    group = FermiSea.HDF5.create_group(file, "$(name)_$(p_scatter)")
                    FermiSea._write_boundary(group, boundary)
                    attrs = FermiSea.HDF5.attributes(group)
                    @test read(attrs["type"]) == "MaxwellWall"
                    @test read(attrs["p_scatter"]) == p_scatter
                    loaded = FermiSea._read_boundary(group, model)
                    @test loaded isa MaxwellWall
                    @test loaded.p_scatter == p_scatter
                end
            end
            group = FermiSea.HDF5.create_group(file, "legacy")
            attrs = FermiSea.HDF5.attributes(group)
            attrs["type"] = "SpecularWall"
            attrs["specularity"] = 0.37
            loaded = FermiSea._read_boundary(group, model)
            @test loaded isa MaxwellWall
            @test loaded.p_scatter ≈ 0.63
        end
    end
end

@testset "CircularFermiSurface HDF5 I/O" begin
    mktempdir() do dir
        surface = circular_fermi_surface(; p_fermi = 1.2, v_fermi = 0.8, mu = 0.5,
                                       nangle = 12, charge = 2, spin_degeneracy = 2)
        path = joinpath(dir, "surface.h5")
        @test save_surface(surface, path) == path
        got = read_surface(path)
        for field in fieldnames(typeof(surface))
            @test getfield(got, field) == getfield(surface, field)
        end
        FermiSea.HDF5.h5open(path, "r+") do file
            file["grid/momenta"][1, 1] += 0.1
        end
        @test_throws ArgumentError read_surface(path)
    end
end

@testset "Solution and PlotData HDF5 I/O" begin
    surface = circular_fermi_surface(p_fermi = 1.2, v_fermi = 0.8, mu = 0.5, nangle = 8)
    model = BoltzmannEquation(surface)
    tree = TreeMesh((-0.5, -0.5), (0.5, 0.5); initial_refinement_level = 0,
                    n_cells_max = 8)
    boundaries = (; x_neg = FixedReservoir(0.1), x_pos = FixedReservoir(-0.1),
                    y_neg = DiffuseWall(), y_pos = DiffuseWall())
    problem = BoltzmannProblem(model, tree, boundaries; degree = 1)
    phi = fill(0.02, length(problem.b))
    sol = BoltzmannSolution(phi, problem, 0.0, true, 0, 1e-10, :converged,
                            (; x_neg = 0.0, x_pos = 0.0), (; x_neg = 0.0, x_pos = 0.0))
    mktempdir() do dir
        path = joinpath(dir, "solution.h5")
        @test save_solution(sol, path; compress = true) == path
        loaded = read_solution(path)
        @test loaded.phi == sol.phi
        @test loaded.contact_offsets == sol.contact_offsets
        @test loaded.contact_currents == sol.contact_currents
        @test contact_particle_flux(loaded, :x_neg) ≈ contact_particle_flux(sol, :x_neg) atol = 1e-12
        @test read_plotdata(path).nodal.names == [:particle_density, :particle_current]

        plot_path = joinpath(dir, "plotdata.h5")
        data = PlotData(sol, particle_density; refine = 1)
        @test save_plotdata(data, plot_path; compress = true) == plot_path
        @test read_plotdata(plot_path; refine = 3).values[:particle_density] ≈
              PlotData(sol, particle_density; refine = 3).values[:particle_density]

        for suffix in ("_x", "_y"), other in (particle_density, particle_current)
            component = Symbol("j", suffix)
            fields = (:j => particle_current, component => other)
            conflicting = PlotData(sol, fields...)
            @test_throws ArgumentError save_plotdata(conflicting, plot_path)
            @test read_plotdata(plot_path).nodal.names == [:particle_density]
            nested = PlotData(sol, :nested => ((phi, surface) ->
                (; j = particle_current(phi, surface),
                   component => other(phi, surface))))
            @test_throws ArgumentError save_plotdata(nested, plot_path)
            @test_throws ArgumentError save_solution(sol, path; observables = fields)
            @test read_solution(path).phi == sol.phi
        end

        collision = Callaway(model; gamma_mr = 0.2, gamma_mc = 0.7)
        drive = ElectricDrive(model, (0.1, 0.0))
        sources = SourceTerms(SourceTerms(collision), SourceTerms((), drive))
        nested_problem = BoltzmannProblem(model, tree, boundaries;
                                          source_terms = sources, degree = 1)
        nested_solution = solve(nested_problem)
        save_solution(nested_solution, path)
        restored = read_solution(path)
        @test restored.phi == nested_solution.phi
        @test restored.problem.b ≈ nested_problem.b
        @test restored.problem.A * phi ≈ nested_problem.A * phi

        observable_path = joinpath(dir, "observables.h5")
        @test save_solution(sol, observable_path; state = false) == observable_path
        @test_throws ArgumentError read_solution(observable_path)
        @test read_plotdata(observable_path).nodal.names == [:particle_density, :particle_current]
        @test_throws ArgumentError save_solution(sol, joinpath(dir, "empty.h5");
                                                  state = false, observables = ())

        custom = CustomCollision(model, (out, input, workspace) -> fill!(out, 0.0))
        custom_problem = BoltzmannProblem(model, tree, boundaries;
                                          source_terms = custom, degree = 1)
        custom_sol = BoltzmannSolution(zeros(length(custom_problem.b)), custom_problem,
            0.0, true, 0, 1e-10, :converged, (;), (;))
        custom_path = joinpath(dir, "custom.h5")
        @test_throws ArgumentError save_solution(custom_sol, custom_path; state = false)
        @test !isfile(custom_path)
    end

    @test_throws ArgumentError save_solution(sol, tempname() * ".h5";
        observables = (:bad => (phi, g) -> Dict(:x => 1),))
end

@testset "Unstructured solution and IncomingProfile I/O" begin
    mesh_path = small_unstructured_mesh(; nx = 1, ny = 1)
    mesh = UnstructuredMesh2D(mesh_path; periodicity = false)
    surface = circular_fermi_surface(p_fermi = 1.2, v_fermi = 0.8, mu = 0.5, nangle = 8)
    model = BoltzmannEquation(surface)
    collision = Callaway(model; gamma_mr = 0.6, gamma_mc = 1.0)
    drive = ElectricDrive(model, (0.1, 0.0))
    channelsources = (collision, drive)
    profile = ChannelProfile(model, 2.0, SVector(1.0, 0.0), SVector(0.0, 1.0),
        map(FermiSea.prepare_boundary, (DiffuseWall(), DiffuseWall())),
        FermiSea.prepare_source(FermiSea.SourceTerms(channelsources)), 0.1,
        [-0.25, 0.25], [0.5, 0.5], zeros(length(surface), 2), SVector(0.0, 0.0),
        0.0, SVector(0.0, 0.0))
    incoming = IncomingProfile(profile; origin = (-0.5, 0.0))
    boundaries = Dict(:left => incoming, :right => FixedReservoir(-0.1),
                      :wall_bottom => DiffuseWall(), :wall_top => DiffuseWall())
    problem = BoltzmannProblem(model, mesh, boundaries; source_terms = channelsources, degree = 1)
    sol = BoltzmannSolution(zeros(length(problem.b)), problem, 0.0, true, 0, 1e-10,
                            :converged, (;), (;))
    mktemp() do path, io
        close(io)
        save_solution(sol, path)
        loaded = read_solution(path)
        @test loaded.phi == sol.phi
        @test loaded.problem.source_terms isa Tuple
        @test loaded.problem.boundary_conditions.left isa IncomingProfile
        @test loaded.problem.boundary_conditions.left.profile.states == profile.states
    end
end
