# Stored nodal values let plotting backends choose a display resolution without losing
# data

struct NodalFields
    nodes::Vector{Float64}
    x::Array{Float64,3}
    y::Array{Float64,3}
    names::Vector{Symbol}
    values::Dict{Symbol,Any}
end

"""
    PlotData(solution, observables...; refine=1)
    PlotData(data::PlotData; refine)

Store observables at the DG nodes and a triangular display mesh. This does not
require a plotting backend. Pass `f(phi, surface)` functions or named pairs
`:name => f`. Each observable returns a real scalar, a length-two real vector,
or a nested NamedTuple of these values.
Defaults are `particle_density` and `particle_current`. Display vectors are
stored as tuples of component arrays. Triangle indices are one-based.

`refine=1` uses the actual DG nodes. Higher refinement of a solution interpolates
the state before evaluating each observable. Refinement of stored `PlotData`
interpolates the stored observables instead. These operations can differ for
nonlinear observables because stored data do not include the angular state.
"""
struct PlotData
    nodal::NodalFields
    refine::Int
    x::Vector{Float64}
    y::Vector{Float64}
    triangles::Matrix{Int32}
    values::Dict{Symbol,Any}
end

function PlotData(solution::BoltzmannSolution, observables...; refine::Integer = 1)
    obs = normalized_observables(observables...)
    data = make_plotdata(nodal_fields(solution, obs), refine)
    refine == 1 && return data

    semi = solution.problem.semi
    surface = semi.equations.surface
    u = Trixi.wrap_array_native(solution.phi, semi)
    nodes = data.nodal.nodes
    m = Int(refine) * (length(nodes) - 1) + 1
    lattice = range(-1, 1; length = m)
    ne = size(data.nodal.x, 3)
    values = Dict{Symbol,Any}()
    point_state = zeros(Float64, length(surface))
    for (name, f) in obs
        sample = nothing
        field = nothing
        k = 0
        for e in 1:ne, eta in lattice, xi in lattice
            wx = lagrange_weights(nodes, xi)
            wy = lagrange_weights(nodes, eta)
            fill!(point_state, 0.0)
            @inbounds for j in eachindex(wy),
                i in eachindex(wx),
                variable in eachindex(point_state)

                point_state[variable] += wx[i] * wy[j] * u[variable, i, j, e]
            end
            value = nodal_value(f(point_state, surface), name)
            if field === nothing
                sample = value
                field = allocate_values(sample, (m * m * ne,))
            end
            k += 1
            assign_value!(field, (k,), value)
        end
        values[name] = field
    end
    return PlotData(data.nodal, data.refine, data.x, data.y, data.triangles, values)
end

PlotData(nodal::NodalFields; refine::Integer = 1) = make_plotdata(nodal, refine)
PlotData(data::PlotData; refine::Integer) = make_plotdata(data.nodal, refine)

function normalize_observables(observables)
    result = Pair{Symbol,Any}[]
    names = Set{Symbol}()
    for observable in observables
        item = if observable isa Pair{Symbol}
            observable
        elseif observable isa Function && !startswith(string(nameof(observable)), "#")
            nameof(observable) => observable
        else
            throw(ArgumentError("pass anonymous observables as :name => f"))
        end
        name, f = item
        name in (:nodes, :triangles) &&
            throw(ArgumentError("observable name $name is reserved"))
        name in names &&
            throw(ArgumentError("observable names must be unique; duplicate $name"))
        push!(names, name)
        push!(result, name => f)
    end
    return result
end

function nodal_value(value, name)
    value isa Real && return Float64(value)
    if value isa AbstractVector && length(value) == 2 && all(x -> x isa Real, value)
        return (Float64(value[1]), Float64(value[2]))
    elseif value isa NamedTuple
        return map(x -> nodal_value(x, name), value)
    end
    throw(
        ArgumentError(
            "observable $name returned $(typeof(value)); return a Real, " *
                "a length-2 real vector, or a NamedTuple " *
                "(wrap anonymous observables as :name => f)"
        )
    )
end

function allocate_values(value, dims)
    value isa Real && return Array{Float64}(undef, dims)
    value isa Tuple && length(value) == 2 && all(x -> x isa Real, value) &&
        return (Array{Float64}(undef, dims), Array{Float64}(undef, dims))
    value isa NamedTuple && return map(x -> allocate_values(x, dims), value)
    error("invalid normalized observable value")
end

function assign_value!(dest::AbstractArray, index, value::Real)
    dest[index...] = value
end
function assign_value!(dest::Tuple, index, value::Tuple)
    assign_value!(dest[1], index, value[1])
    assign_value!(dest[2], index, value[2])
end
function assign_value!(dest::NamedTuple, index, value::NamedTuple)
    foreach(
        k -> assign_value!(getproperty(dest, k), index, getproperty(value, k)),
        keys(dest)
    )
end

function normalized_observables(observables...)
    isempty(observables) && return [nameof(particle_density) => particle_density,
                                    nameof(particle_current) => particle_current]
    return normalize_observables(observables)
end

function nodal_fields(solution::BoltzmannSolution, observables)
    semi = solution.problem.semi
    coordinates = semi.cache.elements.node_coordinates
    n, _, ne = size(coordinates, 2), size(coordinates, 3), size(coordinates, 4)
    x = Array{Float64}(undef, n, n, ne)
    y = similar(x)
    @views x .= coordinates[1, :, :, :]
    @views y .= coordinates[2, :, :, :]
    nodes = Vector{Float64}(semi.solver.basis.nodes)
    u = Trixi.wrap_array_native(solution.phi, semi)
    surface = semi.equations.surface
    names = Symbol[first(pair) for pair in observables]
    values = Dict{Symbol,Any}()
    for (name, f) in observables
        sample = nodal_value(f(view(u, :, 1, 1, 1), surface), name)
        field = allocate_values(sample, (n, n, ne))
        for e in 1:ne, j in 1:n, i in 1:n
            v =
                (i == 1 && j == 1 && e == 1) ? sample :
                nodal_value(f(view(u, :, i, j, e), surface), name)
            assign_value!(field, (i, j, e), v)
        end
        values[name] = field
    end
    return NodalFields(nodes, x, y, names, values)
end

function lagrange_basis!(out, nodes, barycentric, x)
    total = 0.0
    @inbounds for i in eachindex(nodes)
        if x == nodes[i]
            fill!(out, 0.0)
            out[i] = 1.0
            return out
        end
        out[i] = barycentric[i] / (x - nodes[i])
        total += out[i]
    end
    out ./= total
    return out
end

function lagrange_weights(nodes, x)
    return lagrange_basis!(zeros(Float64, length(nodes)), nodes,
                           Trixi.barycentric_weights(nodes), x)
end

interpolate_nodes(a, wx, wy, e) =
    sum(wx[i] * wy[j] * a[i, j, e] for j in eachindex(wy), i in eachindex(wx))
function interpolate_nodes(a::Tuple, wx, wy, e)
    return (interpolate_nodes(a[1], wx, wy, e), interpolate_nodes(a[2], wx, wy, e))
end
function interpolate_nodes(a::NamedTuple, wx, wy, e)
    return map(v -> interpolate_nodes(v, wx, wy, e), a)
end

function flattened_values(nodal, lattice, m)
    out = Dict{Symbol,Any}()
    for name in nodal.names
        source = nodal.values[name]
        sample = interpolate_nodes(source, lattice[1], lattice[1], 1)
        dest = allocate_values(sample, (m * m * size(nodal.x, 3),))
        index = 0
        for e in axes(nodal.x, 3), j in 1:m, i in 1:m
            wx, wy = lattice[i], lattice[j]
            val = interpolate_nodes(source, wx, wy, e)
            assign_value!(dest, (index + 1,), val)
            index += 1
        end
        out[name] = dest
    end
    return out
end

function make_plotdata(nodal::NodalFields, refine::Integer)
    refine >= 1 || throw(ArgumentError("refine must be a positive integer"))
    n = length(nodal.nodes)
    m = Int(refine) * (n - 1) + 1
    reference_nodes = refine == 1 ? nodal.nodes : range(-1, 1; length = m)
    lattice = [lagrange_weights(nodal.nodes, x) for x in reference_nodes]
    ne = size(nodal.x, 3)
    x, y = Vector{Float64}(undef, m*m*ne), Vector{Float64}(undef, m*m*ne)
    k = 0
    for e in 1:ne, j in 1:m, i in 1:m
        wx, wy = lattice[i], lattice[j]
        k += 1
        x[k] = interpolate_nodes(nodal.x, wx, wy, e)
        y[k] = interpolate_nodes(nodal.y, wx, wy, e)
    end
    tris = Matrix{Int32}(undef, 3, 2 * (m - 1)^2 * ne)
    t = 0
    base = 0
    for _ in 1:ne
        for j in 1:(m-1), i in 1:(m-1)
            ll = base + (j - 1) * m + i
            t += 1
            tris[:, t] .= (ll, ll + 1, ll + m + 1)
            t += 1
            tris[:, t] .= (ll, ll + m + 1, ll + m)
        end
        base += m * m
    end
    return PlotData(nodal, Int(refine), x, y, tris, flattened_values(nodal, lattice, m))
end

function outline_segments(data::PlotData)
    n = length(data.nodal.nodes)
    edges = Dict{Any,Vector{Vector{NTuple{2,Float64}}}}()
    diameter = hypot(extrema(data.nodal.x)[2] - extrema(data.nodal.x)[1],
                     extrema(data.nodal.y)[2] - extrema(data.nodal.y)[1])
    tol = 1e-9 * diameter
    for e in axes(data.nodal.x, 3)
        for ids in ([(i, 1) for i in 1:n], [(i, n) for i in 1:n],
                    [(1, j) for j in 1:n], [(n, j) for j in 1:n])
            line = [(data.nodal.x[i,j,e], data.nodal.y[i,j,e]) for (i,j) in ids]
            a, b = first(line), last(line)
            keypoint(p) = (round(Int, p[1] / tol), round(Int, p[2] / tol))
            ka, kb = keypoint(a), keypoint(b)
            key = ka <= kb ? (ka, kb) : (kb, ka)
            get!(edges, key, Vector{NTuple{2,Float64}}[])
            push!(edges[key], ka <= kb ? line : reverse(line))
        end
    end
    return [only(lines) for lines in values(edges) if length(lines) == 1]
end
