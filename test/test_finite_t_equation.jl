@testset "finite-temperature harmonic equation" begin
    equations = IsotropicHarmonicsFiniteT2D(3, 2; mass=1.0, mu0=1.0,
                                            temperature=0.05, n_quad=128)
    nvars = nvariables(equations)

    @test equations isa Trixi.AbstractEquations{2, 14}
    @test nvars == equations.n_radial * (1 + 2 * equations.n_harmonics)
    @test size(equations.Ax) == (nvars, nvars)
    @test size(equations.Ay) == (nvars, nvars)
    @test Trixi.varnames(Trixi.cons2cons, equations)[1:6] ==
          ("a0_r0", "a0_r1", "a1_r0", "a1_r1", "b1_r0", "b1_r1")

    u = SVector{14, Float64}(randn(14))
    @test Trixi.flux(u, 1, equations) ≈ equations.Ax * u
    @test Trixi.flux(u, 2, equations) ≈ equations.Ay * u
    @test Trixi.flux(u, SVector(0.6, 0.8), equations) ≈
          0.6 * equations.Ax * u + 0.8 * equations.Ay * u
    @test Trixi.have_constant_speed(equations) == Trixi.True()
    @test Trixi.max_abs_speeds(equations) == (equations.vmax, equations.vmax)
    @test Trixi.max_abs_speed_naive(u, -u, SVector(3.0, 4.0), equations) ≈
          5.0 * equations.vmax

    for gram in equations.radial_grams
        @test gram ≈ I atol=5.0e-11 rtol=5.0e-11
    end

    S = Diagonal(equations.gram_sqrt)
    Sinv = Diagonal(1 ./ equations.gram_sqrt)
    @test S * equations.Ax * Sinv ≈ (S * equations.Ax * Sinv)' atol=1.0e-10
    @test S * equations.Ay * Sinv ≈ (S * equations.Ay * Sinv)' atol=1.0e-10

    for A in (equations.Ax, equations.Ay)
        for j in 1:nvars, i in 1:nvars
            abs(A[i, j]) <= 1.0e-11 && continue
            ell_i, _, _ = FermiSea._finite_mode(i, equations.n_radial)
            ell_j, _, _ = FermiSea._finite_mode(j, equations.n_radial)
            @test abs(ell_i - ell_j) == 1
        end
    end

    c = randn(nvars)
    @test equations.moment_matrix * (equations.hydro_projector * c) ≈
          equations.moment_matrix * c atol=1.0e-10

    density_moment = reshape(equations.moment_matrix[1, :], 1, :)
    @test density_moment * (equations.density_projector * c) ≈
          density_moment * c atol=1.0e-10
    q = @. tanh(equations.z_nodes / 2) / (2 * equations.temperature)
    expected_quadratic_density = FermiSea._finite_embedding(equations.n_harmonics,
                                                            equations.n_radial,
                                                            equations.radial_basis,
                                                            equations.quad_weights,
                                                            0, :cos, q)
    @test equations.local_equilibrium_quadratic[:, 1] ≈ expected_quadratic_density

    gamma_mr = 0.2
    gamma_mc = 0.7
    collision = LinearCollisionMatrix(equations; gamma_mr, gamma_mc)
    @test density_moment * (collision.W * c) ≈ zeros(1) atol=1.0e-10
    mc_part = gamma_mc .* (Matrix{Float64}(I, nvars, nvars) .-
                           equations.hydro_projector)
    @test equations.moment_matrix * (mc_part * c) ≈
          zeros(size(equations.moment_matrix, 1)) atol=1.0e-10

    normal = SVector(1.0, 0.0)
    wall = MaxwellWallBC(0.0)
    boundary = FermiSea.assemble_ghost_state(wall, u, normal, equations)
    @test dot(FermiSea.normal_flux_row(equations, normal), boundary) ≈ 0.0 atol=1.0e-10

    thermal_u = SVector{14, Float64}(0.03 .* equations.local_equilibrium_linear[:, 1])
    diffuse_template = FermiSea._template(MaxwellWallBC(1.0), thermal_u, normal,
                                          equations)
    @test dot(equations.density_row, diffuse_template) ≈
          dot(equations.density_row, thermal_u) atol=1.0e-11
    @test equations.moment_matrix[2:3, :] * collect(diffuse_template) ≈
          zeros(2) atol=1.0e-11
    @test abs(diffuse_template[FermiSea._finite_scalar_index(equations.n_radial,
                                                             2)]) > 1.0e-8
    diffuse_boundary = FermiSea.assemble_ghost_state(MaxwellWallBC(1.0),
                                                     thermal_u, normal, equations)
    @test dot(FermiSea.normal_flux_row(equations, normal), diffuse_boundary) ≈
          0.0 atol=1.0e-10

    contact = OhmicContactBC(0.5)
    contact_boundary = FermiSea.assemble_ghost_state(contact, u, normal, equations)
    contact_cache = FermiSea.build_projector_cache(equations, normal,
                                                   FermiSea.normal_flux_row(equations,
                                                                            normal))
    contact_template = FermiSea._template(contact, u, equations)
    @test contact_boundary ≈
          u + FermiSea._apply_P_in(contact_cache, contact_template - u)
    contact_density = 2 * equations.temperature *
                      FermiSea._finite_log1pexp((equations.mu0 + contact.bias) /
                                                equations.temperature) -
                      equations.equilibrium_density
    @test dot(equations.density_row, contact_template) ≈ contact_density
end

function _direct_force_projection(equations, u; ntheta=128)
    nvars = nvariables(equations)
    projected_x = zeros(Float64, nvars)
    projected_y = zeros(Float64, nvars)
    angular_weight = 2 / ntheta
    R = equations.n_radial
    M = equations.n_harmonics

    for q in eachindex(equations.quad_weights)
        p = equations.eps_nodes[q] == 0 ? eps(Float64) :
            sqrt(2 * equations.mass * equations.eps_nodes[q])
        z = equations.z_nodes[q]
        radial_prefactor = p / (equations.mass * equations.temperature)
        logw_p_derivative = -radial_prefactor * tanh(z / 2)

        for k in 1:ntheta
            theta = (k - 0.5) * 2 * pi / ntheta
            ct = cos(theta)
            st = sin(theta)
            g = 0.0
            gz = 0.0
            gtheta = 0.0

            for a in 1:R
                idx = FermiSea._finite_scalar_index(R, a)
                coeff = u[idx]
                g += coeff * equations.radial_basis[1][a, q]
                gz += coeff * equations.radial_basis_z_derivative[1][a, q]
            end
            for ell in 1:M
                ctheta = cos(ell * theta)
                stheta = sin(ell * theta)
                for a in 1:R
                    cidx = FermiSea._finite_cos_index(R, ell, a)
                    sidx = FermiSea._finite_sin_index(R, ell, a)
                    ccoeff = u[cidx]
                    scoeff = u[sidx]
                    basis = equations.radial_basis[ell + 1][a, q]
                    basis_z = equations.radial_basis_z_derivative[ell + 1][a, q]
                    angular = ccoeff * ctheta + scoeff * stheta
                    g += basis * angular
                    gz += basis_z * angular
                    gtheta += ell * basis * (-ccoeff * stheta + scoeff * ctheta)
                end
            end

            radial = radial_prefactor * gz + logw_p_derivative * g
            fx = ct * radial - st * gtheta / p
            fy = st * radial + ct * gtheta / p

            for a in 1:R
                idx = FermiSea._finite_scalar_index(R, a)
                basis = equations.radial_basis[1][a, q]
                projected_x[idx] += equations.quad_weights[q] * basis * fx *
                                    angular_weight / 2
                projected_y[idx] += equations.quad_weights[q] * basis * fy *
                                    angular_weight / 2
            end
            for ell in 1:M
                ctheta = cos(ell * theta)
                stheta = sin(ell * theta)
                for a in 1:R
                    cidx = FermiSea._finite_cos_index(R, ell, a)
                    sidx = FermiSea._finite_sin_index(R, ell, a)
                    basis = equations.radial_basis[ell + 1][a, q]
                    projected_x[cidx] += equations.quad_weights[q] * basis * fx *
                                         ctheta * angular_weight
                    projected_x[sidx] += equations.quad_weights[q] * basis * fx *
                                         stheta * angular_weight
                    projected_y[cidx] += equations.quad_weights[q] * basis * fy *
                                         ctheta * angular_weight
                    projected_y[sidx] += equations.quad_weights[q] * basis * fy *
                                         stheta * angular_weight
                end
            end
        end
    end

    return projected_x, projected_y
end

@testset "finite-temperature electrostatics" begin
    chi = 0.15
    equations0 = IsotropicHarmonicsFiniteT2D(3, 2; mass=1.0, mu0=1.0,
                                             temperature=0.05, n_quad=128)
    equations = IsotropicHarmonicsFiniteT2D(3, 2; mass=1.0, mu0=1.0,
                                            temperature=0.05, n_quad=128,
                                            electrostatic_chi=chi)
    nvars = nvariables(equations)
    u = SVector{14, Float64}(randn(14))
    v = SVector{14, Float64}(randn(14))

    @test Trixi.have_nonconservative_terms(equations0) == Trixi.False()
    @test Trixi.have_nonconservative_terms(equations) == Trixi.True()
    @test equations.Ax ≈ equations0.Ax .+
                         chi .* (equations.velocity_embedding_x *
                                  equations.density_row')
    @test equations.Ay ≈ equations0.Ay .+
                         chi .* (equations.velocity_embedding_y *
                                  equations.density_row')
    @test Trixi.flux(u, 1, equations) ≈ equations.Ax * u
    @test equations0.force_vmax == 0.0
    @test equations.force_vmax > 0

    @test flux_gradual_channel_volume(zero(u), v, 1, equations) ==
          zero(u)
    @test flux_no_electrostatic_nonconservative(u, v, 1, equations) == zero(u)
    @test flux_no_electrostatic_nonconservative(u, v, SVector(0.6, 0.8),
                                                equations) == zero(u)
    density_free = SVector{14, Float64}((I - equations.density_projector) * randn(nvars))
    @test dot(equations.density_row, density_free) ≈ 0.0 atol=1.0e-12
    @test flux_gradual_channel_volume(u, density_free, 1, equations) ≈
          zero(u) atol=1.0e-12

    expected_x = -chi * (equations.Dx_force * collect(u)) *
                 dot(equations.density_row, v)
    expected_n = -chi *
                 ((0.6 .* equations.Dx_force .+ 0.8 .* equations.Dy_force) *
                  collect(u)) * dot(equations.density_row, v)
    @test flux_gradual_channel_volume(u, v, 1, equations) ≈ expected_x
    @test flux_gradual_channel_volume(u, v, SVector(0.6, 0.8),
                                      equations) ≈ expected_n

    equations_parabolic = GradualChannelForce2D(equations)
    source = GradualChannelForceSource()
    grad_x = 0.3 .* v
    grad_y = -0.2 .* v
    source_expected = -chi .* ((dot(equations.density_row, grad_x) .* equations.Dx_force .+
                                dot(equations.density_row, grad_y) .* equations.Dy_force) *
                               collect(u))
    @test source(u, (grad_x, grad_y), SVector(0.0, 0.0), 0.0,
                 equations_parabolic) ≈ source_expected

    for D in (equations.Dx_force, equations.Dy_force)
        for j in 1:nvars, i in 1:nvars
            abs(D[i, j]) <= 1.0e-10 && continue
            ell_i, _, _ = FermiSea._finite_mode(i, equations.n_radial)
            ell_j, _, _ = FermiSea._finite_mode(j, equations.n_radial)
            @test abs(ell_i - ell_j) == 1
        end
    end

    direct_x, direct_y = _direct_force_projection(equations, collect(u))
    @test equations.Dx_force * collect(u) ≈ direct_x atol=1.0e-8 rtol=1.0e-8
    @test equations.Dy_force * collect(u) ≈ direct_y atol=1.0e-8 rtol=1.0e-8

    contact = OhmicContactBC(0.2)
    contact_template = FermiSea._template(contact, zero(u), equations)
    contact_fields = hydrodynamic_fields(equations, contact_template)
    @test contact_fields.electrochemical_potential ≈ contact.bias atol=1.0e-12
    @test abs(contact_fields.density_delta / equations.equilibrium_density) <
          abs(contact.bias)
    contact_boundary = FermiSea.assemble_ghost_state(contact, zero(u),
                                                     SVector(0.0, 1.0),
                                                     equations)
    boundary_fields = hydrodynamic_fields(equations, contact_boundary)
    @test boundary_fields.electrochemical_potential ≈ contact.bias atol=1.0e-12

    potential_contact = ChemicalPotentialContactBC(0.2)
    potential_template = FermiSea._template(potential_contact, zero(u), equations)
    potential_fields = hydrodynamic_fields(equations, potential_template)
    @test potential_fields.delta_mu ≈ potential_contact.delta_mu atol=1.0e-12
    exact_density_delta = 2 * equations.temperature *
                          FermiSea._finite_log1pexp((equations.mu0 +
                                                      potential_contact.delta_mu) /
                                                     equations.temperature) -
                          equations.equilibrium_density
    @test potential_fields.density_delta / equations.equilibrium_density ≈
          exact_density_delta / equations.equilibrium_density rtol=1.0e-12
    potential_boundary = FermiSea.assemble_ghost_state(potential_contact, zero(u),
                                                       SVector(0.0, 1.0),
                                                       equations)
    potential_boundary_fields = hydrodynamic_fields(equations, potential_boundary)
    @test 0 < potential_boundary_fields.density_delta <
          potential_fields.density_delta

    density_contact = DensityContactBC(0.1)
    density_template = FermiSea._template(density_contact, zero(u), equations)
    density_fields = hydrodynamic_fields(equations, density_template)
    @test density_fields.density_delta / equations.equilibrium_density ≈
          density_contact.relative_density atol=1.0e-12
    @test density_fields.delta_mu ≈
          FermiSea._finite_chemical_potential_shift_from_density_delta(equations,
                                                                       0.1 *
                                                                       equations.equilibrium_density)
    density_boundary = FermiSea.assemble_ghost_state(density_contact, zero(u),
                                                     SVector(0.0, 1.0),
                                                     equations)
    density_boundary_fields = hydrodynamic_fields(equations, density_boundary)
    @test 0 < density_boundary_fields.density_delta / equations.equilibrium_density <
          density_contact.relative_density

    normal = SVector(0.0, 1.0)
    @test Trixi.flux(potential_boundary, normal, equations) ≈
          equations.Ay * potential_boundary
end

@testset "finite-temperature nonlinear BGK" begin
    equations = IsotropicHarmonicsFiniteT2D(3, 3; mass=1.0, mu0=1.0,
                                            temperature=0.05, n_quad=160)
    nvars = nvariables(equations)
    source = NonlinearBGKCollision(equations; gamma_mr=0.2, gamma_mc=0.7)
    linear = LinearCollisionMatrix(equations; gamma_mr=0.2, gamma_mc=0.7)

    u0 = zero(SVector{nvars, Float64})
    n_total, delta_mu, vx, vy = FermiSea._finite_bgk_parameters(equations, u0)
    @test n_total ≈ equations.equilibrium_density
    @test delta_mu ≈ 0.0 atol=1.0e-13
    @test vx ≈ 0.0 atol=1.0e-13
    @test vy ≈ 0.0 atol=1.0e-13

    density_state = SVector{nvars, Float64}(0.03 .* equations.local_equilibrium_linear[:, 1])
    _, delta_mu_density, vx_density, vy_density =
        FermiSea._finite_bgk_parameters(equations, density_state)
    @test delta_mu_density > 0
    @test vx_density ≈ 0.0 atol=1.0e-13
    @test vy_density ≈ 0.0 atol=1.0e-13

    momentum_state = SVector{nvars, Float64}(0.02 .*
                                             equations.local_equilibrium_linear[:, 2])
    n_mom, _, vx_mom, vy_mom = FermiSea._finite_bgk_parameters(equations,
                                                               momentum_state)
    px_mom = dot(view(equations.moment_matrix, 2, :), momentum_state)
    @test vx_mom ≈ px_mom / (equations.mass * n_mom)
    @test vy_mom ≈ 0.0 atol=1.0e-13
    fields = hydrodynamic_fields(equations, momentum_state)
    @test hydrodynamic_density(equations, momentum_state) ≈ fields.density
    @test hydrodynamic_momentum(equations, momentum_state) ≈ fields.momentum
    @test hydrodynamic_velocity(equations, momentum_state) ≈ fields.velocity
    @test hydrodynamic_chemical_potential_shift(equations, momentum_state) ≈
          fields.delta_mu
    @test fields.electrochemical_potential ≈ fields.delta_mu
    @test fields.speed ≈ hypot(fields.velocity...)

    u = SVector{nvars, Float64}(0.02 .* randn(nvars))
    density_target = FermiSea._finite_density_local_equilibrium(equations, u)
    hydro_target = FermiSea._finite_hydro_local_equilibrium(equations, u)
    @test dot(equations.density_row, density_target) ≈
          dot(equations.density_row, u) atol=1.0e-11
    @test equations.moment_matrix * collect(hydro_target) ≈
          equations.moment_matrix * collect(u) atol=1.0e-11

    nonlinear_source = source(u, SVector(0.0, 0.0), 0.0, equations)
    @test dot(equations.density_row, nonlinear_source) ≈ 0.0 atol=1.0e-11
    mc_source = NonlinearBGKCollision(equations; gamma_mr=0.0, gamma_mc=0.7)
    mc_value = mc_source(u, SVector(0.0, 0.0), 0.0, equations)
    @test equations.moment_matrix * collect(mc_value) ≈
          zeros(size(equations.moment_matrix, 1)) atol=1.0e-11
    mr_source = NonlinearBGKCollision(equations; gamma_mr=0.2, gamma_mc=0.0)
    mr_value = mr_source(u, SVector(0.0, 0.0), 0.0, equations)
    @test dot(equations.density_row, mr_value) ≈ 0.0 atol=1.0e-11

    eps_scale = 1.0e-6
    small_u = SVector{nvars, Float64}(eps_scale .* randn(nvars))
    @test source(small_u, SVector(0.0, 0.0), 0.0, equations) ≈
          linear(small_u, SVector(0.0, 0.0), 0.0, equations) atol=1.0e-10

    direction = SVector{nvars, Float64}(randn(nvars))
    alpha = 1.0e-3
    d1 = source(alpha .* direction, SVector(0.0, 0.0), 0.0, equations) -
         linear(alpha .* direction, SVector(0.0, 0.0), 0.0, equations)
    d2 = source(2alpha .* direction, SVector(0.0, 0.0), 0.0, equations) -
         linear(2alpha .* direction, SVector(0.0, 0.0), 0.0, equations)
    @test norm(d2 - 4 .* d1) <= 5.0e-2 * max(norm(d2), eps(Float64))

    density_response = dot(equations.density_row, equations.local_equilibrium_linear[:, 1])
    bad_scale = -2 * equations.equilibrium_density / density_response
    bad_state = SVector{nvars, Float64}(bad_scale .* equations.local_equilibrium_linear[:, 1])
    @test isnan(hydrodynamic_density(equations, bad_state))
    @test isnan(hydrodynamic_chemical_potential_shift(equations, bad_state))
    bad_fields = hydrodynamic_fields(equations, bad_state)
    @test !isfinite(bad_fields.density)
    @test !isfinite(bad_fields.delta_mu)
    @test all(!isfinite, bad_fields.velocity)
    @test all(!isfinite, source(bad_state, SVector(0.0, 0.0), 0.0, equations))
    bad_template = FermiSea._template(MaxwellWallBC(1.0), bad_state, SVector(1.0, 0.0),
                                      equations)
    @test all(!isfinite, bad_template)

    truncated_equations = IsotropicHarmonicsFiniteT2D(1, 2; mass=1.0, mu0=1.0,
                                                      temperature=0.05, n_quad=96)
    @test_logs (:warn,
                r"truncates the ell=2 quadratic target components") NonlinearBGKCollision(
        truncated_equations; gamma_mr=0.1, gamma_mc=0.2)

    energy_equations = IsotropicHarmonicsFiniteT2D(2, 2; conserve_energy=true)
    @test_throws ArgumentError NonlinearBGKCollision(energy_equations;
                                                     gamma_mr=0.1,
                                                     gamma_mc=0.2)
end

@testset "finite-temperature electrostatic DG smoke" begin
    mesh_file = joinpath(@__DIR__, "..", "assets", "square_bells", "square_bells.inp")
    mesh = P4estMesh{2}(mesh_file; polydeg=1,
                        boundary_symbols=[:contact_bottom, :contact_top, :walls])
    equations = IsotropicHarmonicsFiniteT2D(1, 1; n_quad=64,
                                            electrostatic_chi=0.05)
    initial_condition(x, t, equations) =
        zero(SVector{nvariables(equations), Float64})
    solver = DGSEM(polydeg=1,
                   surface_flux=(flux_lax_friedrichs,
                                 flux_no_electrostatic_nonconservative),
                   volume_integral=VolumeIntegralFluxDifferencing(
                       (flux_central, flux_no_electrostatic_nonconservative)))
    boundary_conditions = (contact_bottom=OhmicContactBC(-0.01),
                           contact_top=OhmicContactBC(0.01),
                           walls=MaxwellWallBC(1.0))
    boundary_conditions_parabolic = (contact_bottom=boundary_condition_do_nothing,
                                     contact_top=boundary_condition_do_nothing,
                                     walls=boundary_condition_do_nothing)
    semi = SemidiscretizationHyperbolicParabolic(mesh,
        (equations, GradualChannelForce2D(equations)), initial_condition, solver;
        solver_parabolic=ParabolicFormulationBassiRebay1(),
        source_terms=NonlinearBGKCollision(equations; gamma_mr=0.1, gamma_mc=1.0),
        source_terms_parabolic=GradualChannelForceSource(),
        boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))
    ode = semidiscretize(semi, (0.0, 1.0e-3))
    du = similar(ode.u0)
    ode.f(du, ode.u0, ode.p, 0.0)
    @test all(isfinite, du)
end
