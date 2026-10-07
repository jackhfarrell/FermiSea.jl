@testset "circular surface and Callaway" begin
    surface = circular_fermi_surface(; p_fermi = 2.0, v_fermi = 4.0, mu = 3.0,
                                  nangle = 16, spin_degeneracy = 2)
    @test sum(surface.weights) ≈ 1 / (2pi) atol = 10eps()
    @test particle_density(ones(length(surface)), surface) ≈ 1 / (2pi) atol = 10eps()
    @test particle_current(ones(length(surface)), surface) ≈ zero(SVector{2}) atol = 2e-16

    model = BoltzmannEquation(surface)
    collision = Callaway(model; gamma_mr = 0.7, gamma_mc = 1.3)
    population = ones(length(surface))
    px = surface.momenta[:, 1]
    py = surface.momenta[:, 2]
    @test norm(positive_collision(collision, population)) < 2e-14
    @test positive_collision(collision, px) ≈ 0.7px atol = 2e-14
    @test positive_collision(collision, py) ≈ 0.7py atol = 2e-14

    Random.seed!(102)
    phi = randn(length(surface))
    psi = randn(length(surface))
    action = positive_collision(collision, phi)
    action_psi = positive_collision(collision, psi)
    @test dot(surface.weights, action) ≈ 0 atol = 2e-14
    @test dot(surface.weights .* px, action) ≈
          0.7 * dot(surface.weights .* px, phi) atol = 2e-14
    @test dot(surface.weights .* psi, action) ≈
          dot(surface.weights .* action_psi, phi) atol = 2e-14
    @test dot(surface.weights .* phi, action) >= -2e-14

    # The source has the opposite sign and therefore dissipates the weighted norm
    source = collision(SVector{length(surface)}(phi), SVector(0.0, 0.0), 0.0, model)
    @test source ≈ -action
    @test dot(surface.weights .* phi, source) <= 2e-14
end

@testset "circular model contracts and response scaling" begin
    for nangle in (4, 8, 32), p_fermi in (0.7, 2.0), v_fermi in (0.4, 3.0)
        surface = circular_fermi_surface(; p_fermi, v_fermi, nangle, spin_degeneracy = 2)
        @test sum(surface.weights) ≈ p_fermi / (pi * v_fermi)
        @test hypot.(surface.momenta[:, 1], surface.momenta[:, 2]) ≈ fill(p_fermi, nangle)
        @test hypot.(surface.velocities[:, 1], surface.velocities[:, 2]) ≈ fill(v_fermi, nangle)
        @test particle_current(cos.(surface.theta), surface)[1] ≈ p_fermi / (2pi)
        @test charge_current(cos.(surface.theta), surface) ≈
              surface.charge .* particle_current(cos.(surface.theta), surface)
    end
    for kwargs in ((; p_fermi = 0, v_fermi = 1),
                   (; p_fermi = 1, v_fermi = Inf),
                   (; p_fermi = 1, v_fermi = 1, mu = NaN),
                   (; p_fermi = 1, v_fermi = 1, nangle = 7),
                   (; p_fermi = 1, v_fermi = 1, nangle = 2),
                   (; p_fermi = 1, v_fermi = 1, spin_degeneracy = 0))
        @test_throws ArgumentError circular_fermi_surface(; kwargs...)
    end
end

@testset "custom collision on the circle" begin
    model = BoltzmannEquation(circular_fermi_surface(; p_fermi = 1, v_fermi = 1, nangle = 4))
    matrix = [2.0 -1.0 0.0 -1.0; -1.0 2.0 -1.0 0.0;
              0.0 -1.0 2.0 -1.0; -1.0 0.0 -1.0 2.0]
    collision = CustomCollision(model, sparse(matrix); rate_bound = 4.0)
    input = [0.4, -0.2, 0.7, 0.1]
    @test positive_collision(collision, input) ≈ matrix * input
    @test positive_collision(collision, ones(4)) ≈ zeros(4)
    @test_throws ArgumentError apply_collision!(input, collision, input)
    inputs = hcat(input, 2 .* input)
    output = similar(inputs)
    @test apply_collision!(output, collision, inputs) ≈ matrix * inputs
    source = prepare_source(collision)(SVector{4}(input), SVector(0.0, 0.0), 0.0, model)
    @test source ≈ -(matrix * input)
end
