@testset "characteristic projectors" begin
    equations = IsotropicFermiHarmonics2D(3)
    normals = (SVector(1.0, 0.0), normalize(SVector(0.6, 0.8)))

    for normal in normals
        rho_row = FermiSea.normal_flux_row(equations, normal)
        cache = FermiSea.build_projector_cache(equations.n_harmonics, normal,
                                                       rho_row)
        I7 = Matrix{Float64}(I, 7, 7)
        basis = ntuple(i -> SVector{7, Float64}(I7[:, i]), 7)
        P_in = hcat((FermiSea._apply_P_in(cache, e_i) for e_i in basis)...)
        P_out = I7 - P_in

        @test P_in + P_out ≈ I7
        @test P_in * P_out ≈ zeros(7, 7) atol=1.0e-12

        A = FermiSea._normal_matrix(equations.n_harmonics, normal)
        weights = FermiSea._gram_sqrt_weights(7, Float64)
        A_orth = [weights[i] * A[i, j] / weights[j] for i in 1:7, j in 1:7]
        n_in = count(<(-sqrt(eps(Float64))), eigvals(Symmetric(A_orth)))
        @test tr(P_in) ≈ n_in

        u_inner = SVector(1.0, 0.3, -0.2, 0.1, -0.05, 0.02, -0.01)
        u_boundary, C = FermiSea.solve_bc_constant(cache, u_inner)
        @test C isa Real
        @test dot(rho_row, u_boundary - u_inner) ≈ 0 atol=1.0e-12
    end
end
