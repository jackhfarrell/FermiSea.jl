using CairoMakie

@testset "Makie extension" begin
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0, nangle = 12)
    Makie.colorbuffer(plot(surface))
    Makie.colorbuffer(plot(surface, :nodes; arrows = true))
    Makie.colorbuffer(plot(surface, :nodes; color = :weight))

    solution = plotdata_solution(TreeMesh((-0.5, -0.5), (0.5, 0.5);
        initial_refinement_level = 1, n_cells_max = 100))
    data = PlotData(solution, particle_density, particle_current; refine = 2)
    for source in (solution, data)
        for component in (:magnitude, :x, :y)
            fig = Figure()
            ax = Axis(fig[1, 1]; aspect = DataAspect(), xautolimitmargin = (0, 0),
                      yautolimitmargin = (0, 0))
            plot!(ax, source, source === solution ? particle_current : :particle_current;
                  component, lengthscale = 3, refine = 2)
            autolimits!(ax)
            Makie.colorbuffer(fig)
            @test ax.finallimits[].widths[1] ≈ 3 atol = 1e-8
        end
    end
    @test_throws ArgumentError plot(solution, particle_density; component = :x)

    fig = Figure(); ax = Axis(fig[1, 1])
    streamplot!(ax, solution, particle_current)
    Makie.colorbuffer(fig)
    mktempdir() do dir
        path = joinpath(dir, "plotdata.h5")
        save_plotdata(data, path)
        Makie.colorbuffer(plot(read_plotdata(path), :particle_current; component = :y, refine = 2))
        fig = Figure(); ax = Axis(fig[1, 1]); streamplot!(ax, read_plotdata(path), :particle_current)
        Makie.colorbuffer(fig)
    end
end
