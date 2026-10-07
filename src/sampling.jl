# Each sampler owns its scratch so separate samplers can handle concurrent queries

struct ElementLocator
    coordinates::Array{Float64,4}
    origins::Matrix{Float64}
    scales::Matrix{Float64}
    boxes::Matrix{Float64}
    nodes::Vector{Float64}
    barycentric::Vector{Float64}
    xi_weights::Vector{Float64}
    eta_weights::Vector{Float64}
    plus_weights::Vector{Float64}
    minus_weights::Vector{Float64}
end

function ElementLocator(physical, nodes)
    barycentric = Vector{Float64}(Trixi.barycentric_weights(nodes))
    boxes = bernstein_boxes(physical, nodes)
    coordinates = Array{Float64}(undef, size(physical))
    origins = Matrix{Float64}(undef, 2, size(physical, 4))
    scales = similar(origins)
    for element in axes(physical, 4), axis in 1:2
        values = view(physical, axis, :, :, element)
        lo, hi = extrema(values)
        hi > lo || throw(ArgumentError("degenerate spatial element $element"))
        origins[axis, element] = lo
        scales[axis, element] = hi - lo
        @views coordinates[axis, :, :, element] .= (values .- lo) ./ (hi - lo)
    end
    scratch = ntuple(_ -> zeros(Float64, length(nodes)), 4)
    return ElementLocator(
        coordinates,
        origins,
        scales,
        boxes,
        nodes,
        barycentric,
        scratch...
    )
end

"""
    SpatialSampler(solution, observable=nodal_state; outside=:nan)
    SpatialSampler(data::PlotData, field_name; outside=:nan)

Get an observable at a point by interpolating the DG angular state on the curved
physical mesh, then evaluating `observable(phi, surface)`. By default, we return a
copy of the angular state. Use `outside=:nan`, `:nothing`, or `:error` to choose
what happens outside the device. With `outside=:nan`, scalar, vector, and nested
NamedTuple results keep their structure, with numeric values replaced by NaNs.

For `PlotData`, we interpolate the stored observable values. For a solution,
we evaluate the observable after interpolating the state. These operations can
differ for nonlinear observables. Select a stored field with a Symbol, or a tuple
of Symbols for a nested NamedTuple field.

Call the sampler as `sampler(x, y)` and reuse it for many points. Each sampler
has interpolation and state scratch arrays, so build a separate one for each
concurrent task.
"""
struct SpatialSampler{S,U,O,T}
    solution::S
    u::U
    observable::O
    outside::Symbol
    locator::ElementLocator
    point_state::Vector{T}
end

"""
    nodal_state(phi, surface)

Return a copy of the unweighted momentum vector. This is the default observable
of a solution sampler. Copying prevents later queries from overwriting its result.
"""
nodal_state(phi, surface) = copy(phi)

function bernstein_boxes(physical, nodes)
    degree = length(nodes) - 1
    bernstein = [binomial(degree, k) * ((x + 1) / 2)^k *
                 ((1 - x) / 2)^(degree - k) for x in nodes, k in 0:degree]
    transform = inv(bernstein)
    roundoff = 64eps(Float64) * opnorm(transform, Inf)^2
    boxes = Matrix{Float64}(undef, 4, size(physical, 4))
    for element in axes(physical, 4), axis in 1:2
        values = view(physical, axis, :, :, element)
        lo, hi = extrema(transform * values * transform')
        padding = roundoff * max(maximum(abs, values), 1.0)
        boxes[2axis - 1, element] = prevfloat(lo - padding)
        boxes[2axis, element] = nextfloat(hi + padding)
    end
    return boxes
end

function SpatialSampler(solution::BoltzmannSolution, observable = nodal_state;
                        outside::Symbol = :nan)
    outside in (:nan, :nothing, :error) || throw(ArgumentError(
        "outside must be :nan, :nothing, or :error"))
    semi = solution.problem.semi
    semi.mesh isa Union{Trixi.TreeMesh{2},Trixi.UnstructuredMesh2D} || throw(
        ArgumentError(
            "spatial sampling supports TreeMesh{2} and UnstructuredMesh2D")
    )
    physical = semi.cache.elements.node_coordinates
    nodes = Vector{Float64}(semi.solver.basis.nodes)
    locator = ElementLocator(physical, nodes)
    nvars = length(semi.equations.surface)
    return SpatialSampler(solution, Trixi.wrap_array_native(solution.phi, semi),
                          observable, outside, locator, zeros(Float64, nvars))
end

function mapped_point(xi_weights, eta_weights, ex, ey)
    x = 0.0
    y = 0.0
    @inbounds for j in eachindex(eta_weights), i in eachindex(xi_weights)
        weight = xi_weights[i] * eta_weights[j]
        x += weight * ex[i, j]
        y += weight * ey[i, j]
    end
    return x, y
end

function invert_element_map!(locator::ElementLocator, element, target_x, target_y)
    (; nodes, barycentric, xi_weights, eta_weights, plus_weights, minus_weights) =
        locator
    ex = view(locator.coordinates, 1, :, :, element)
    ey = view(locator.coordinates, 2, :, :, element)
    xi = 0.0
    eta = 0.0
    h = 1.0e-6
    for _ in 1:30
        lagrange_basis!(xi_weights, nodes, barycentric, xi)
        lagrange_basis!(eta_weights, nodes, barycentric, eta)
        x, y = mapped_point(xi_weights, eta_weights, ex, ey)
        rx, ry = x - target_x, y - target_y
        abs(rx) <= 1.0e-12 && abs(ry) <= 1.0e-12 && return xi, eta, true
        xp, yp =
            mapped_point(lagrange_basis!(plus_weights, nodes, barycentric, xi + h),
                eta_weights, ex, ey)
        xm, ym =
            mapped_point(lagrange_basis!(minus_weights, nodes, barycentric, xi - h),
                eta_weights, ex, ey)
        x_xi, y_xi = (xp - xm) / (2h), (yp - ym) / (2h)
        xp, yp = mapped_point(xi_weights,
            lagrange_basis!(plus_weights, nodes, barycentric, eta + h), ex, ey)
        xm, ym = mapped_point(xi_weights,
            lagrange_basis!(minus_weights, nodes, barycentric, eta - h), ex, ey)
        x_eta, y_eta = (xp - xm) / (2h), (yp - ym) / (2h)
        determinant = x_xi * y_eta - x_eta * y_xi
        abs(determinant) <= 64eps(Float64) * hypot(x_xi, y_xi) *
                            hypot(x_eta, y_eta) && return xi, eta, false
        xi -= (rx * y_eta - ry * x_eta) / determinant
        eta -= (x_xi * ry - y_xi * rx) / determinant
    end
    return xi, eta, false
end

function interpolate_point_state!(sampler, element, xi, eta)
    locator = sampler.locator
    lagrange_basis!(locator.xi_weights, locator.nodes, locator.barycentric, xi)
    lagrange_basis!(locator.eta_weights, locator.nodes, locator.barycentric, eta)
    fill!(sampler.point_state, 0.0)
    @inbounds for j in eachindex(locator.eta_weights),
        i in eachindex(locator.xi_weights)

        weight = locator.xi_weights[i] * locator.eta_weights[j]
        for variable in eachindex(sampler.point_state)
            sampler.point_state[variable] +=
                weight * sampler.u[variable, i, j, element]
        end
    end
    return sampler.point_state
end

nan_value(value::Number) = oftype(float(value), NaN)
nan_value(value::AbstractArray) = map(nan_value, value)
nan_value(value::Tuple) = map(nan_value, value)
nan_value(value::NamedTuple) = map(nan_value, value)

function outside_sample(sampler, x, y)
    sampler.outside === :nothing && return nothing
    sampler.outside === :error &&
        throw(DomainError((x, y), "point lies outside the mesh"))
    prototype = sampler.observable(sampler.point_state,
                                   sampler.solution.problem.semi.equations.surface)
    return nan_value(prototype)
end

function (sampler::SpatialSampler)(x::Real, y::Real)
    located = locate_element!(sampler.locator, x, y)
    if located !== nothing
        element, xi, eta = located
        phi = interpolate_point_state!(sampler, element, xi, eta)
        return sampler.observable(phi, sampler.solution.problem.semi.equations.surface)
    end
    return outside_sample(sampler, x, y)
end

function locate_element!(locator::ElementLocator, x, y)
    for element in axes(locator.boxes, 2)
        locator.boxes[1, element] <= x <= locator.boxes[2, element] || continue
        locator.boxes[3, element] <= y <= locator.boxes[4, element] || continue
        tx = (x - locator.origins[1, element]) / locator.scales[1, element]
        ty = (y - locator.origins[2, element]) / locator.scales[2, element]
        xi, eta, converged = invert_element_map!(locator, element, tx, ty)
        converged && abs(xi) <= 1 + 1.0e-10 && abs(eta) <= 1 + 1.0e-10 || continue
        return element, xi, eta
    end
    return nothing
end

struct PlotDataSpatialSampler{L,V}
    locator::L
    values::V
    outside::Symbol
    prototype::Any
end

function selected_field(values, name::Symbol)
    haskey(values, name) || throw(ArgumentError("PlotData has no field $name"))
    return values[name]
end
function selected_field(values, name::Tuple{Vararg{Symbol}})
    isempty(name) && throw(ArgumentError("field path must not be empty"))
    value = selected_field(values, first(name))
    for key in Iterators.drop(name, 1)
        value isa NamedTuple && hasproperty(value, key) ||
            throw(ArgumentError("PlotData has no field path $name"))
        value = getproperty(value, key)
    end
    return value
end

function SpatialSampler(data::PlotData, name; outside::Symbol = :nan)
    outside in (:nan, :nothing, :error) ||
        throw(ArgumentError("outside must be :nan, :nothing, or :error"))
    field = selected_field(data.nodal.values, name isa Function ? nameof(name) : name)
    physical = Array{Float64}(undef, 2, size(data.nodal.x)...)
    @views physical[1, :, :, :] .= data.nodal.x
    @views physical[2, :, :, :] .= data.nodal.y
    locator = ElementLocator(physical, data.nodal.nodes)
    prototype = plot_sample_type(field_prototype(field))
    return PlotDataSpatialSampler(locator, field, outside, prototype)
end

field_prototype(a::AbstractArray) = zero(eltype(a))
field_prototype(a::Tuple) = map(field_prototype, a)
field_prototype(a::NamedTuple) = map(field_prototype, a)

function interpolate_field(a, wx, wy, e)
    a isa AbstractArray && return interpolate_nodes(a, wx, wy, e)
    a isa Tuple && return map(v -> interpolate_field(v, wx, wy, e), a)
    return map(v -> interpolate_field(v, wx, wy, e), a)
end

plot_sample_type(value::Tuple{<:Real,<:Real}) = SVector{2,Float64}(value)
plot_sample_type(value::NamedTuple) = map(plot_sample_type, value)
plot_sample_type(value) = value

function (sampler::PlotDataSpatialSampler)(x::Real, y::Real)
    located = locate_element!(sampler.locator, x, y)
    if located === nothing
        sampler.outside === :nothing && return nothing
        sampler.outside === :error &&
            throw(DomainError((x, y), "point lies outside the mesh"))
        return nan_value(sampler.prototype)
    end
    e, xi, eta = located
    wx = lagrange_weights(sampler.locator.nodes, xi)
    wy = lagrange_weights(sampler.locator.nodes, eta)
    return plot_sample_type(interpolate_field(sampler.values, wx, wy, e))
end
