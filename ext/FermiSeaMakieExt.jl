# Keeping them in an extension leaves the core package independent of plotting backends

module FermiSeaMakieExt

using FermiSea
using Makie
import Makie: plot, plot!, streamplot!

"""
    plot(surface::CircularFermiSurface; color=:steelblue, label="Fermi surface", kwargs...)
    plot(surface::CircularFermiSurface, :nodes; color=:energy, arrows=false, kwargs...)

Return a Makie Figure of the circular Fermi surface or its momentum nodes.
Node colours represent `:energy` or `:weight`. Axes use model momentum units.
"""
function plot(surface::FermiSea.CircularFermiSurface; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "pₓ", ylabel = "pᵧ")
    plot!(ax, surface; kwargs...)
    return fig
end

function plot!(ax, surface::FermiSea.CircularFermiSurface; color = :steelblue,
               label = "Fermi surface", kwargs...)
    # A dense analytic circle keeps the outline independent of angular resolution
    theta = range(0, 2pi; length = 257)
    lines!(ax, surface.p_fermi .* cos.(theta), surface.p_fermi .* sin.(theta);
           color, label, kwargs...)
    return ax
end

function plot(surface::FermiSea.CircularFermiSurface, ::Val{:nodes}; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "pₓ", ylabel = "pᵧ")
    plot!(ax, surface, Val(:nodes); kwargs...)
    return fig
end

function plot(surface::FermiSea.CircularFermiSurface, selector::Symbol; kwargs...)
    selector === :nodes || throw(ArgumentError("the momentum node view selector must be :nodes"))
    return plot(surface, Val(:nodes); kwargs...)
end

function plot!(ax, surface::FermiSea.CircularFermiSurface, ::Val{:nodes}; color = :energy,
               arrows = false, kwargs...)
    color in (:energy, :weight) || throw(ArgumentError("color must be :energy or :weight"))
    c = color === :energy ? surface.energies : surface.weights
    scatter!(ax, surface.momenta[:, 1], surface.momenta[:, 2]; color = c, kwargs...)
    if arrows
        extent = max(maximum(surface.momenta[:, 1]) - minimum(surface.momenta[:, 1]),
                     maximum(surface.momenta[:, 2]) - minimum(surface.momenta[:, 2]))
        scale = 0.1extent / maximum(hypot.(surface.velocities[:, 1], surface.velocities[:, 2]))
        arrows2d!(ax, surface.momenta[:, 1], surface.momenta[:, 2],
                  scale .* surface.velocities[:, 1], scale .* surface.velocities[:, 2])
    end
    return ax
end
function plot!(ax, surface::FermiSea.CircularFermiSurface, selector::Symbol; kwargs...)
    selector === :nodes || throw(ArgumentError("the momentum node view selector must be :nodes"))
    return plot!(ax, surface, Val(:nodes); kwargs...)
end

function source_data(source, observable; refine)
    if source isa FermiSea.BoltzmannSolution
        return FermiSea.PlotData(source, observable; refine)
    elseif source isa FermiSea.PlotData
        return source.refine == refine ? source : FermiSea.PlotData(source; refine)
    end
    throw(ArgumentError("source must be a solution or PlotData"))
end

stored_name(observable::Function) = nameof(observable)
stored_name(observable::Pair) = first(observable)
stored_name(observable::Symbol) = observable
stored_name(observable::Tuple{Vararg{Symbol}}) = observable
function plot_values(data, observable, component)
    field = FermiSea.selected_field(data.values, stored_name(observable))
    if field isa Tuple && length(field) == 2 && all(x -> x isa AbstractVector, field)
        component === :magnitude && return hypot.(field[1], field[2])
        component === :x && return field[1]
        component === :y && return field[2]
        throw(ArgumentError("component must be :magnitude, :x, or :y"))
    end
    field isa AbstractVector || throw(ArgumentError("selected observable is not a scalar or vector field"))
    component in (:magnitude, :x, :y) || throw(ArgumentError("component must be :magnitude, :x, or :y"))
    component === :magnitude && return field
    throw(ArgumentError("component $component is not valid for a scalar observable"))
end

"""
    plot!(axis, source::Union{BoltzmannSolution,PlotData}, observable; component=:magnitude,
          lengthscale=1, valuescale=1, refine=4, outline=true, kwargs...)

Return a Makie mesh plot of a scalar field or vector magnitude/component. Coordinates
are multiplied by `lengthscale`. Values are divided by `valuescale`. Signed vector
components default to a centred diverging scale, and magnitudes to sequential colours.
Makie keywords override the local defaults. Stored fields are selected by Symbol or
nested field path. Solutions accept observables or named pairs. A saved file must
first be read with `read_plotdata`.
"""
function plot!(ax, source::Union{FermiSea.BoltzmannSolution,FermiSea.PlotData}, observable; component = :magnitude, lengthscale = 1,
               valuescale = 1, refine = 4, outline = true, kwargs...)
    observable isa Function || observable isa Pair || observable isa Symbol || observable isa Tuple{Vararg{Symbol}} ||
        throw(ArgumentError("observable must be a function, pair, or stored field name"))
    data = source_data(source, observable; refine)
    values = plot_values(data, observable, component)
    scaled = values ./ valuescale
    if !haskey(kwargs, :colorrange)
        if component in (:x, :y)
            limit = maximum(abs, scaled)
            kwargs = (; kwargs..., colorrange = (-limit, limit))
        else
            kwargs = (; kwargs..., colorrange = extrema(scaled))
        end
    end
    if !haskey(kwargs, :colormap)
        kwargs = (; kwargs..., colormap = component in (:x, :y) ? :balance : :viridis)
    end
    faces = [Makie.GeometryBasics.TriangleFace(data.triangles[1, i],
                data.triangles[2, i], data.triangles[3, i])
             for i in axes(data.triangles, 2)]
    plt = mesh!(ax, Point2f.(data.x .* lengthscale, data.y .* lengthscale),
                faces; color = scaled, shading = NoShading, kwargs...)
    if outline
        for segment in FermiSea.outline_segments(data)
            lines!(ax, first.(segment) .* lengthscale, last.(segment) .* lengthscale;
                   color = :black, linewidth = 1)
        end
    end
    return plt
end

"""
    plot(source::Union{BoltzmannSolution,PlotData}, observable; kwargs...)

Return a Figure containing a spatial field axis, device outline, and colour bar.
Field scaling and component selection follow `plot!`. Defaults apply only to
this figure and do not alter global Makie themes.
"""
function plot(source::Union{FermiSea.BoltzmannSolution,FermiSea.PlotData}, observable; component = :magnitude, lengthscale = 1,
              valuescale = 1, refine = 4, outline = true, kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y")
    plt = plot!(ax, source, observable; component, lengthscale, valuescale, refine, outline, kwargs...)
    Colorbar(fig[1, 2], plt)
    return fig
end

"""
    streamplot!(axis, source::Union{BoltzmannSolution,PlotData}, observable;
                lengthscale=1, valuescale=1, kwargs...)

Return a Makie streamline plot of a two-component observable. Solution sampling
evaluates the observable on an interpolated state. Stored-field sampling interpolates
saved values. Coordinates are multiplied by `lengthscale` and vectors divided by
`valuescale`. Points outside the device return NaN vectors.
"""
function streamplot!(ax, source::Union{FermiSea.BoltzmannSolution,FermiSea.PlotData}, observable; lengthscale = 1, valuescale = 1, kwargs...)
    sampler_observable = observable isa Pair ? last(observable) : observable
    sampler = if source isa FermiSea.BoltzmannSolution
        FermiSea.SpatialSampler(source, sampler_observable)
    else
        data = source
        data isa FermiSea.PlotData || throw(ArgumentError("source must be a solution or PlotData"))
        FermiSea.SpatialSampler(data, stored_name(observable))
    end
    nodal = source isa FermiSea.PlotData ? source.nodal :
            nothing
    if nodal === nothing
        points = source.problem.semi.cache.elements.node_coordinates
        xs, ys = vec(points[1, :, :, :]), vec(points[2, :, :, :])
    else
        xs, ys = vec(nodal.x), vec(nodal.y)
    end
    xspan = Tuple(extrema(xs) .* lengthscale)
    yspan = Tuple(extrema(ys) .* lengthscale)
    field(p) = begin
        v = sampler(p[1] / lengthscale, p[2] / lengthscale) ./ valuescale
        Point2f(v[1], v[2])
    end
    return streamplot!(ax, field, xspan, yspan; kwargs...)
end

end
