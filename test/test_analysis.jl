function boundary_node_normals(semi, boundary_name)
    _, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    boundaries = cache.boundaries
    contravariant_vectors = cache.elements.contravariant_vectors
    index_range = collect(Trixi.eachnode(solver))
    boundary_normals = Dict{Tuple{Int, Int},
                            SVector{2, eltype(contravariant_vectors)}}()

    for boundary in semi.boundary_conditions.boundary_symbol_indices[boundary_name]
        element      = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction    = Trixi.indices2direction(node_indices)

        i_start, i_step = Trixi.index_to_start_step_2d(node_indices[1], index_range)
        j_start, j_step = Trixi.index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_start
        j_node = j_start
        for node in index_range
            normal = Trixi.get_normal_direction(direction, contravariant_vectors,
                                                i_node, j_node, element)
            boundary_normals[(boundary, node)] =
                SVector(normal[1], normal[2]) / norm(normal)
            i_node += i_step
            j_node += j_step
        end
    end

    return boundary_normals
end

@testset "contact analysis" begin
    mesh_file = joinpath(@__DIR__, "..", "assets", "square_bells", "square_bells.inp")
    mesh = P4estMesh{2}(mesh_file; polydeg=1,
                        boundary_symbols=[:contact_bottom, :contact_top, :walls])

    equations = IsotropicFermiHarmonics2D(2)
    initial_condition(x, t, equations) =
        SVector(ntuple(i -> i == 1 ? 1.0 : 0.0, Val(nvariables(equations))))
    boundary_conditions = (;
        contact_bottom = CurrentContactBC(0.1),
        contact_top    = CurrentContactBC(-0.1),
        walls          = MaxwellWallBC(1.0),
    )

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
               DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs);
               boundary_conditions=boundary_conditions)
    ode = semidiscretize(semi, (0.0, 0.01))

    bottom_indices = semi.boundary_conditions.boundary_symbol_indices[:contact_bottom]
    solver_nodes = collect(Trixi.eachnode(semi.solver))
    bottom_bc = boundary_conditions.contact_bottom
    bottom_cache = FermiSea._boundary_projector_cache(semi, bottom_bc)
    @test all((haskey(bottom_cache.projector_face_indices, boundary) ||
               all(haskey(bottom_cache.projector_indices, (boundary, node))
                   for node in solver_nodes))
              for boundary in bottom_indices)
    @test all(key[1] in bottom_indices && key[2] in solver_nodes
              for key in keys(bottom_cache.projector_indices))
    @test all(boundary in bottom_indices
              for boundary in keys(bottom_cache.projector_face_indices))
    @test length(bottom_cache.projectors) == 1
    @test length(bottom_cache.projectors) <=
          length(bottom_cache.projector_face_indices) + length(bottom_cache.projector_indices)

    du = similar(ode.u0)
    ode.f(du, ode.u0, ode.p, 0.0)

    bottom_length = contact_boundary_length(semi, :contact_bottom)
    top_length = contact_boundary_length(semi, :contact_top)

    @test bottom_length > 0
    @test top_length > 0
    @test contact_boundary_length(semi, (:contact_bottom, :contact_top)) ≈
          bottom_length + top_length

    bottom_current = Trixi.analyze(ContactCurrent(:contact_bottom), du, ode.u0, 0.0, semi)
    bottom_average = Trixi.analyze(ContactCurrentAverage(:contact_bottom), du, ode.u0,
                                   0.0, semi)

    @test isfinite(bottom_current)
    @test bottom_average ≈ bottom_current / bottom_length
    @test Trixi.analyze(SteadyStateResidual(), du, ode.u0, 0.0, semi) ==
          maximum(abs, du)
    @test Trixi.default_analysis_integrals(equations) == (SteadyStateResidual(),)
    @test Trixi.pretty_form_ascii(SteadyStateResidual()) == "steady_state"
    @test Trixi.pretty_form_ascii(ContactCurrent(:contact_bottom)) == "I_contact_bottom"
    @test Trixi.pretty_form_ascii(ContactCurrentAverage(:contact_bottom)) ==
          "j_contact_bottom_avg"
    @test Trixi.pretty_form_ascii(ContactCurrent(:contact_bottom, :contact_top)) ==
          "I_contact_bottom_and_contact_top"
end
