@testset "magnetic and tomographic FFT operators" begin
    surface = circular_fermi_surface(; p_fermi = 1.0, v_fermi = 1.0,
                                  nangle = 10, charge = -1.0)
    model = BoltzmannEquation(surface)
    theta = surface.theta
    magnetic = MagneticField(model, 0.3)
    @test magnetic.frequency ≈ 0.3 atol = 2e-15
    workspace = magnetic_workspace(magnetic, 2)
    input = hcat(cos.(2theta), sin.(3theta))
    output = similar(input)
    apply_magnetic!(output, magnetic, input, workspace)
    @test output[:, 1] ≈ -0.6sin.(2theta) atol = 3e-14
    @test output[:, 2] ≈ 0.9cos.(3theta) atol = 3e-14
    @test norm(apply_magnetic!(zeros(10), magnetic, ones(10))) < 2e-14
    a, b = randn(MersenneTwister(31), 10), randn(MersenneTwister(32), 10)
    da, db = similar(a), similar(b)
    apply_magnetic!(da, magnetic, a)
    apply_magnetic!(db, magnetic, b)
    @test dot(surface.weights .* a, db) ≈ -dot(surface.weights .* da, b) atol = 3e-14
    @test length(prepare_source(magnetic).workspaces) == Threads.maxthreadid()

    tomography = TomographicCollision(model; gamma_mr = 0.2,
                                       gamma_mc = 0.7, gamma_3 = 0.1)
    for mode in 0:5
        harmonic = cos.(mode .* theta)
        expected_rate = mode == 0 ? 0.0 : mode == 1 ? 0.2 : iseven(mode) ? 0.9 :
                        0.2 + min(0.1 * (mode / 3)^4, 0.7)
        @test positive_collision(tomography, harmonic) ≈
              expected_rate .* harmonic atol = 4e-14
    end
    nyquist = (-1.0) .^ (0:9)
    @test positive_collision(tomography, nyquist) ≈ 0.9nyquist atol = 4e-14
end

@testset "magnetic frequency scaling and in-place transforms" begin
    for charge in (-2.0, 2.0), field in (0.0, 0.3)
        surface = circular_fermi_surface(; p_fermi = 1.7, v_fermi = 0.8, nangle = 16, charge)
        magnetic = MagneticField(BoltzmannEquation(surface), field)
        omega = -charge * field * 0.8 / 1.7
        input = hcat(cos.(2 .* surface.theta), sin.(3 .* surface.theta))
        expected = hcat(-2omega .* sin.(2 .* surface.theta),
                        3omega .* cos.(3 .* surface.theta))
        @test apply_magnetic!(input, magnetic, input, magnetic_workspace(magnetic, 2)) ≈
              expected atol = 3e-14
        nyquist = (-1.0) .^ (0:15)
        @test apply_magnetic!(similar(nyquist), magnetic, nyquist) ≈ zeros(16) atol = 3e-14
    end
end
