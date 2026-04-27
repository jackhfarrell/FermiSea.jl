@testset "boundary conditions" begin
    equations = IsotropicFermiHarmonics2D(2)
    normal = SVector(1.0, 0.0)
    surface_flux = Trixi.flux_lax_friedrichs

    wall = MaxwellWallBC(0.0)
    uniform = SVector(2.0, 0.0, 0.0, 0.0, 0.0)
    boundary = FermiSea.assemble_ghost_state(wall, uniform, normal, equations)
    @test dot(FermiSea.normal_flux_row(equations, normal), boundary) ≈ 0 atol=1.0e-12
    @test wall(uniform, normal, SVector(0.0, 0.0), 0.0, surface_flux, equations) isa SVector

    contact = OhmicContactBC(0.5)
    u = SVector(1.0, 0.4, 0.1, -0.2, 0.3)
    contact_template = SVector(0.5, 0.0, 0.0, 0.0, 0.0)
    contact_cache = FermiSea.build_projector_cache(equations.n_harmonics, normal,
                                                   FermiSea.normal_flux_row(equations,
                                                                            normal))
    contact_boundary = FermiSea.assemble_ghost_state(contact, u, normal, equations)
    expected_contact_boundary = u +
                                FermiSea._apply_P_in(contact_cache,
                                                     contact_template - u)
    @test contact_boundary ≈ expected_contact_boundary
    @test abs(dot(FermiSea.normal_flux_row(equations, normal), contact_boundary)) > 1.0e-12

    floating = FloatingProbeBC()
    @test_throws ArgumentError FermiSea.assemble_ghost_state(floating, u, normal,
                                                                     equations)

    @test_throws ArgumentError FermiSea.assemble_ghost_state(CurrentContactBC(0.7),
                                                                     u, normal,
                                                                     equations)
end

@testset "boundary projector cache deduplication" begin
    mesh_file = joinpath(@__DIR__, "..", "assets", "square_bells", "square_bells.inp")
    mesh = P4estMesh{2}(mesh_file; polydeg=1,
                        boundary_symbols=[:contact_bottom, :contact_top, :walls])

    equations = IsotropicFermiHarmonics2D(2)
    initial_condition(x, t, equations) =
        SVector(ntuple(i -> i == 1 ? 1.0 : 0.0, Val(nvariables(equations))))
    boundary_conditions = (;
        contact_bottom = CurrentContactBC(0.1),
        contact_top    = FloatingProbeBC(),
        walls          = MaxwellWallBC(1.0),
    )

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
               DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs);
               boundary_conditions=boundary_conditions)
    FermiSea.initialize_boundary_projectors!(semi)

    _, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    boundaries = cache.boundaries
    contravariant_vectors = cache.elements.contravariant_vectors
    index_range = collect(Trixi.eachnode(solver))

    bottom_nodes_by_normal = Dict{SVector{2, eltype(contravariant_vectors)},
                                  Vector{Tuple{Int, Int}}}()
    for boundary in semi.boundary_conditions.boundary_symbol_indices[:contact_bottom]
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
            n = SVector(normal[1], normal[2]) / norm(normal)
            push!(get!(bottom_nodes_by_normal, n, Tuple{Int, Int}[]), (boundary, node))
            i_node += i_step
            j_node += j_step
        end
    end

    repeated_nodes = nothing
    for nodes in values(bottom_nodes_by_normal)
        if length(nodes) > 1
            repeated_nodes = nodes
            break
        end
    end

    @test repeated_nodes !== nothing
    repeated_nodes === nothing && error("expected repeated boundary normals on sample mesh")

    first_key, second_key = repeated_nodes[1:2]
    bottom_bc = boundary_conditions.contact_bottom
    bottom_cache = FermiSea._boundary_projector_cache(semi, bottom_bc)
    @test haskey(bottom_cache.projector_face_indices, first_key[1])
    @test haskey(bottom_cache.projector_face_indices, second_key[1])
    @test bottom_cache.projector_face_indices[first_key[1]] ==
          bottom_cache.projector_face_indices[second_key[1]]
    @test isempty(bottom_cache.projector_indices)
    @test length(bottom_cache.projectors) == 1
    top_bc = boundary_conditions.contact_top
    top_cache = FermiSea._boundary_projector_cache(semi, top_bc)
    @test !isempty(top_cache.boundary_indices)
    @test !hasproperty(bottom_bc, :projectors)
    @test !hasproperty(bottom_bc, :projector_face_indices)
    @test !hasproperty(bottom_bc, :projector_indices)

    ode = semidiscretize(semi, (0.0, 1.0))
    du = similar(ode.u0)
    ode.f(du, ode.u0, ode.p, 0.0)

    potential = FermiSea._current_contact_potential(bottom_bc, equations, solver,
                                                            cache)
    integrated_current = 0.0
    weights = solver.basis.weights
    bottom_cache = FermiSea._boundary_projector_cache(cache, bottom_bc)
    for boundary in bottom_cache.boundary_indices
        element      = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction    = Trixi.indices2direction(node_indices)

        i_start, i_step = Trixi.index_to_start_step_2d(node_indices[1], index_range)
        j_start, j_step = Trixi.index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_start
        j_node = j_start
        for node in index_range
            normal_direction = Trixi.get_normal_direction(direction,
                                                          contravariant_vectors,
                                                          i_node, j_node, element)
            projector = FermiSea._lookup_projectors(bottom_cache, boundary,
                                                            node, normal_direction,
                                                            equations)
            u_inner = Trixi.get_node_vars(boundaries.u, equations, solver, node,
                                          boundary)
            u_boundary = FermiSea.assemble_ghost_state(projector, bottom_bc,
                                                               u_inner,
                                                               normal_direction,
                                                               equations, potential)
            integrated_current += weights[node] *
                                  Trixi.flux(u_boundary, normal_direction,
                                             equations)[1]

            i_node += i_step
            j_node += j_step
        end
    end
    @test integrated_current ≈ bottom_bc.current atol=1.0e-12

    floating_potential = FermiSea._current_contact_potential(top_bc, equations,
                                                                     solver, cache)
    floating_current = 0.0
    top_cache = FermiSea._boundary_projector_cache(cache, top_bc)
    for boundary in top_cache.boundary_indices
        element      = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction    = Trixi.indices2direction(node_indices)

        i_start, i_step = Trixi.index_to_start_step_2d(node_indices[1], index_range)
        j_start, j_step = Trixi.index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_start
        j_node = j_start
        for node in index_range
            normal_direction = Trixi.get_normal_direction(direction,
                                                          contravariant_vectors,
                                                          i_node, j_node, element)
            projector = FermiSea._lookup_projectors(top_cache, boundary,
                                                            node, normal_direction,
                                                            equations)
            u_inner = Trixi.get_node_vars(boundaries.u, equations, solver, node,
                                          boundary)
            u_boundary = FermiSea.assemble_ghost_state(projector, top_bc,
                                                               u_inner,
                                                               normal_direction,
                                                               equations,
                                                               floating_potential)
            floating_current += weights[node] *
                                Trixi.flux(u_boundary, normal_direction,
                                           equations)[1]

            i_node += i_step
            j_node += j_step
        end
    end
    @test floating_current ≈ 0.0 atol=1.0e-12
end
