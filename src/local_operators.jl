# Magnetic motion, tomographic collisions, and electric drive act at each spatial point
# SourceTerms combines them. Prepare_source gives each thread its own scratch

"""
    MagneticField(model, B; charge=model.surface.charge)

For a perpendicular field `B` in model units, the Lorentz characteristic
`dp/dt = qB(vy,-vx)` gives `omega = -q B v_fermi / p_fermi`.
The stored angle increases counterclockwise. We put `LB = omega d/dtheta` on
the left of the equation, so Trixi uses `-LB * phi`.

The even angular-grid Nyquist derivative is zero to keep the FFT derivative real.
"""
struct MagneticField{T<:AbstractFloat}
    n::Int
    field::T
    charge::T
    frequency::T
end

function MagneticField(model::BoltzmannEquation, field::Real;
                       charge::Real = model.surface.charge)
    isfinite(field) && isfinite(charge) ||
        throw(ArgumentError("field and charge must be finite"))
    surface = model.surface
    frequency = -Float64(charge) * Float64(field) * surface.v_fermi / surface.p_fermi
    isfinite(frequency) || throw(ArgumentError("orbit frequency must be finite"))
    return MagneticField(length(surface), Float64(field), Float64(charge), Float64(frequency))
end

struct AngularFFTWorkspace{A,P,Q}
    spectrum::A
    forward::P
    inverse::Q
end

function AngularFFTWorkspace(nangle::Integer, ntransforms::Integer)
    spectrum = zeros(ComplexF64, nangle, ntransforms)
    forward = FFTW.plan_fft!(spectrum, 1; flags = FFTW.ESTIMATE)
    inverse = FFTW.plan_ifft!(spectrum, 1; flags = FFTW.ESTIMATE)
    return AngularFFTWorkspace(spectrum, forward, inverse)
end

@inline function angular_wave(index, nangle)
    mode = index - 1
    # The even angular-grid Nyquist mode has no real-valued angular derivative
    2mode == nangle && return 0
    # FFT orders negative waves after the positive ones
    return mode <= nangle ÷ 2 ? mode : mode - nangle
end

"""
    magnetic_workspace(field, ncolumns=1)

Make complex FFT scratch arrays for `apply_magnetic!` on a matrix with one
angular state per column. Each spatial column has its own transform.
"""
magnetic_workspace(field::MagneticField, ncolumns::Integer = 1) =
    AngularFFTWorkspace(field.n, ncolumns)

"""
    apply_magnetic!(output, field, input[, workspace])

Write the magnetic action `LB * input` into `output` and return it. This is the
term on the left of the steady equation. Use a vector or a matrix with one
angular state per column. Input and output can share storage because we use
FFT scratch arrays. The workspace column count must match the input; if no
workspace is passed, we make one.
"""
function apply_magnetic!(output, field::MagneticField, input,
    workspace::AngularFFTWorkspace)
    size(output) == size(input) ||
        throw(DimensionMismatch("magnetic arrays must agree"))
    size(input, 1) == field.n || throw(DimensionMismatch(
        "the first magnetic axis must have $(field.n) nodes"))
    ncolumns = input isa AbstractVector ? 1 : size(input, 2)
    size(workspace.spectrum) == (field.n, ncolumns) ||
        throw(DimensionMismatch("magnetic workspace has the wrong number of columns"))
    workspace.spectrum .= reshape(input, field.n, ncolumns)
    mul!(workspace.spectrum, workspace.forward, workspace.spectrum)
    # Angular differentiation brings down i times the Fourier mode
    for column in axes(workspace.spectrum, 2), q in axes(workspace.spectrum, 1)
        workspace.spectrum[q, column] *= im * angular_wave(q, field.n) * field.frequency
    end
    mul!(workspace.spectrum, workspace.inverse, workspace.spectrum)
    reshape(output, field.n, ncolumns) .= real.(workspace.spectrum)
    return output
end

function apply_magnetic!(output, field::MagneticField, input)
    return apply_magnetic!(output, field, input,
                           magnetic_workspace(field, input isa AbstractVector ? 1 :
                                                      size(input, 2)))
end

function (field::MagneticField)(u, x, t,
                                equations::BoltzmannEquation{N}) where {N}
    throw(ArgumentError("prepare_source(magnetic_field) before using it in Trixi"))
end

"""
    TomographicCollision(model; gamma_mr, gamma_mc, gamma_3)

Use a collision model with separate odd and even harmonic rates on one circular
Fermi surface at zero temperature. The angular nodes are uniform. Mode zero has
zero rate, `|m| = 1` has `gamma_mr`, higher even modes have
`gamma_mr + gamma_mc`, and higher odd modes have
`gamma_mr + min(gamma_3 (|m|/3)^4, gamma_mc)`. The real angular-grid Nyquist mode uses the rate
for its actual harmonic parity. Reuse a workspace for repeated calls, and make
a separate one for each concurrent call.
"""
struct TomographicCollision{T<:AbstractFloat}
    n::Int
    gamma_mr::T
    gamma_mc::T
    gamma_3::T
    rates::Vector{T}
end

struct TomographicWorkspace{A,B,P,Q}
    values::A
    spectrum::B
    forward::P
    inverse::Q
end

function TomographicCollision(model::BoltzmannEquation; gamma_mr::Real,
                              gamma_mc::Real, gamma_3::Real)
    surface = model.surface
    n = length(surface)
    all(rate -> isfinite(rate) && rate >= 0, (gamma_mr, gamma_mc, gamma_3)) ||
        throw(ArgumentError("tomographic rates must be finite and nonnegative"))
    # Number is conserved. First harmonics relax only through momentum loss
    rates = map(0:(n ÷ 2)) do mode
        mode == 0 && return 0.0
        mode == 1 && return Float64(gamma_mr)
        iseven(mode) && return Float64(gamma_mr + gamma_mc)
        # The cap keeps odd-mode relaxation no faster than the even-mode rate
        return Float64(gamma_mr + min(gamma_3 * (mode / 3)^4, gamma_mc))
    end
    return TomographicCollision(n, Float64(gamma_mr), Float64(gamma_mc),
                                Float64(gamma_3), rates)
end

function collision_workspace(collision::TomographicCollision, ncolumns::Integer = 1)
    values = zeros(Float64, collision.n, ncolumns)
    spectrum = zeros(ComplexF64, collision.n ÷ 2 + 1, ncolumns)
    forward = FFTW.plan_rfft(values, 1; flags = FFTW.ESTIMATE)
    inverse = FFTW.plan_irfft(spectrum, collision.n, 1; flags = FFTW.ESTIMATE)
    return TomographicWorkspace(values, spectrum, forward, inverse)
end

function apply_collision!(output, collision::TomographicCollision, input,
    workspace::TomographicWorkspace)
    size(output) == size(input) ||
        throw(DimensionMismatch("collision arrays must agree"))
    size(input, 1) == collision.n || throw(
        DimensionMismatch(
            "the first collision axis must have $(collision.n) nodes")
    )
    ncolumns = input isa AbstractVector ? 1 : size(input, 2)
    size(workspace.values) == (collision.n, ncolumns) || throw(DimensionMismatch(
        "tomographic workspace has the wrong number of columns"))
    # Copy before the FFT so input and output may be the same array
    workspace.values .= reshape(input, collision.n, ncolumns)
    mul!(workspace.spectrum, workspace.forward, workspace.values)
    workspace.spectrum .*= collision.rates
    mul!(workspace.values, workspace.inverse, workspace.spectrum)
    reshape(output, collision.n, ncolumns) .= workspace.values
    return output
end

function apply_collision!(output, collision::TomographicCollision, input)
    return apply_collision!(output, collision, input,
                            collision_workspace(collision,
                                input isa AbstractVector ? 1 : size(input, 2)))
end

function positive_collision(collision::TomographicCollision, input)
    output = similar(input, promote_type(Float64, eltype(input)))
    return apply_collision!(output, collision, input)
end

function (collision::TomographicCollision)(u, x, t,
    equations::BoltzmannEquation{N}) where {N}
    throw(
        ArgumentError("prepare_source(tomographic_collision) before using it in Trixi")
    )
end

struct PreparedLocalSource{O,W}
    operator::O
    workspaces::W
end

"""
    ElectricDrive(model, electric_field)

Apply the electrical drive `d_i = q E dot v_i` in the steady equation. Pass the
two-component electric field `E` in model units. We multiply by carrier charge
once and store `qE` as `charged_field`. This adds forcing without changing the
linear operator, so preconditioners include it only through the residual
right-hand side.
"""
struct ElectricDrive{T<:AbstractFloat}
    charged_field::SVector{2,T}
end

function ElectricDrive(model::BoltzmannEquation, field)
    length(field) == 2 ||
        throw(DimensionMismatch("electric field must have two entries"))
    all(isfinite, field) || throw(ArgumentError("electric field must be finite"))
    # Store qE once so evaluating the source only needs a dot product
    return ElectricDrive(SVector{2,Float64}(model.surface.charge .* field))
end

@inline function (drive::ElectricDrive)(u, x, t,
                                        equations::BoltzmannEquation{N}) where {N}
    velocities = equations.surface.velocities
    return SVector{N}(ntuple(i -> drive.charged_field[1] * velocities[i, 1] +
                                  drive.charged_field[2] * velocities[i, 2], Val(N)))
end

"""
    SourceTerms(terms...)
    SourceTerms(terms::Tuple)

Combine local Trixi sources in a tuple. Each term receives `(phi, x, t, model)`.
We use the negative collision action as a source. An empty tuple returns zero.
"""
struct SourceTerms{T<:Tuple}
    terms::T
    SourceTerms(terms::T) where {T<:Tuple} = new{T}(terms)
end

SourceTerms(terms...) = SourceTerms(terms)

@inline function (source::SourceTerms)(u, x, t, equations)
    # Starting at zero also makes an empty source tuple valid
    return foldl((output, term) -> output + term(u, x, t, equations),
                 source.terms; init = zero(u))
end

"""
    prepare_source(operator_or_terms)

Prepare sources and their callback scratch arrays for Trixi. Return the prepared
source, or `nothing` for no source. We also prepare terms inside nested source
tuples. Magnetic, tomographic, and custom actions get separate scratch arrays
for each Julia thread. The prepared source belongs to one active problem, so
prepare it again for independent concurrent solves. `BoltzmannProblem` does
this automatically.
"""
prepare_source(source::Tuple) = prepare_source(SourceTerms(source))

prepare_source(source::SourceTerms) = SourceTerms(map(prepare_source, source.terms))
prepare_source(source::PreparedLocalSource) = prepare_source(source.operator)
prepare_source(source) = source

# Each worker owns scratch because operator applications may run concurrently over DG
# nodes
prepare_source(operator::MagneticField) = PreparedLocalSource(
    operator, [magnetic_workspace(operator) for _ in 1:Threads.maxthreadid()])
prepare_source(operator::TomographicCollision) = PreparedLocalSource(
    operator, [collision_workspace(operator) for _ in 1:Threads.maxthreadid()])
prepare_source(operator::CustomCollision) = PreparedLocalSource(
    operator, [collision_workspace(operator) for _ in 1:Threads.maxthreadid()])

# These positive left-hand actions enter Trixi's evolution with a minus sign
function (source::PreparedLocalSource{<:MagneticField})(u, x, t,
    equations::BoltzmannEquation{N}) where {N}
    output = MVector{N,promote_type(Float64, eltype(u))}(undef)
    apply_magnetic!(output, source.operator, u, source.workspaces[Threads.threadid()])
    return -SVector{N}(output)
end

function (source::PreparedLocalSource{<:Union{TomographicCollision,CustomCollision}})(
        u, x, t, equations::BoltzmannEquation{N}) where {N}
    output = MVector{N,promote_type(Float64,eltype(u))}(undef)
    apply_collision!(output, source.operator, u, source.workspaces[Threads.threadid()])
    return -SVector{N}(output)
end
