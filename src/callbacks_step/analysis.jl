# some custom bahaviours for Trix's built-in Analysis callbacks

# TODO: push upstream if this doesn't already exist? Seems natural people would want to log
# the steady-state residual in the analysis callback...
"""
    SteadyStateResidual()

This way analysis callback can compute the steady-state residual that Trixi is using to 
determine stop-condition (convergence)
"""
struct SteadyStateResidual end
function _analyze_steady_state_residual(du, semi::Trixi.AbstractSemidiscretization)
    _, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    du = ndims(du) == 1 ? Trixi.wrap_array(du, semi) : du
    residual = zero(eltype(du))
    for element in Trixi.eachelement(solver, cache)
        for j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)
            du_local = Trixi.get_node_vars(du, equations, solver, i, j, element)
            residual = Base.max(residual,
                                Trixi.residual_steady_state(du_local, equations))
        end
    end
    if Trixi.mpi_isparallel()
        global_residual = Trixi.MPI.Reduce!(Ref(residual), Base.max,
                                            Trixi.mpi_root(), Trixi.mpi_comm())
        if Trixi.mpi_isroot()
            residual = global_residual[]
        end
    end
    return residual
end

# have to hack it seems to get it to work w/ the existing AnalysisCallback machinery, I
# wish there were a more natural way
function Trixi.analyze(::SteadyStateResidual, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    return _analyze_steady_state_residual(du, semi)
end
Trixi.pretty_form_utf(::SteadyStateResidual) = "steady state"
Trixi.pretty_form_ascii(::SteadyStateResidual) = "steady_state"

# for isotropic Fermi Harmonics, incude by default
Trixi.default_analysis_integrals(::IsotropicFermiHarmonics2D) = (SteadyStateResidual(),)

# Really annoying error. Basically I figure out that the default implementation of
# integrate_via_indices segfaults when the integrand is SVector w/ length > 40 or something,
# and we need to be able to handle this case
# Override the method with the same semantics but a manual Threads.@threads
# reduction over per-thread partial integrals
# TODO: try to upstream this??
function Trixi.integrate_via_indices(func::Func, u,
                                     mesh::Trixi.P4estMesh{2},
                                     equations::IsotropicFermiHarmonics2D,
                                     dg::Trixi.DGSEM, cache, args...;
                                     normalize=true) where {Func}
    weights = dg.basis.weights

    integral0 = zero(func(u, 1, 1, 1, equations, dg, args...))
    nthreads  = Threads.nthreads()
    partial_integrals = [integral0 for _ in 1:nthreads]
    partial_volumes   = zeros(real(mesh), nthreads)

    elements = collect(eachelement(dg, cache))
    Threads.@threads for idx in eachindex(elements)
        tid     = Threads.threadid()
        element = elements[idx]
        acc_i   = partial_integrals[tid]
        acc_v   = partial_volumes[tid]
        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
            w = volume_jacobian * weights[i] * weights[j]
            acc_i += w * func(u, i, j, element, equations, dg, args...)
            acc_v += w
        end
        partial_integrals[tid] = acc_i
        partial_volumes[tid]   = acc_v
    end

    integral     = sum(partial_integrals)
    total_volume = sum(partial_volumes)
    return normalize ? integral / total_volume : integral
end


"""
    ContactNormalCurrent()

Pointwise normal current used internally by the contact-current analysis
integrals.

For `IsotropicFermiHarmonics2D`, the first component of the physical flux is
the charge/current flux through the supplied normal direction. 
"""
struct ContactNormalCurrent end

"""
    ContactCurrent(boundary_symbol)
    ContactCurrent(boundary_symbol, boundary_symbols...)

Analysis integral for the total current through one or more named boundaries.

The boundary symbols are the same symbols used in the semidiscretization's
boundary-condition dictionary.  This is intended to be passed as an
`analysis_integral`, for example

```julia
AnalysisCallback(semi;
                 analysis_integrals=(ContactCurrent(:contact_bottom),))
```

If multiple boundaries are supplied, their currents are added together.
"""
struct ContactCurrent{N}
    boundary_symbols::NTuple{N, Symbol}
end

ContactCurrent(boundary_symbol::Symbol) = ContactCurrent((boundary_symbol,))
ContactCurrent(boundary_symbol::Symbol, boundary_symbols::Symbol...) =
    ContactCurrent((boundary_symbol, boundary_symbols...))

"""
    ContactCurrentAverage(boundary_symbol)
    ContactCurrentAverage(boundary_symbol, boundary_symbols...)

Analysis integral for the boundary-averaged normal current.

This computes `ContactCurrent(...) / contact_boundary_length(...)`, which is
often the more convenient quantity when comparing contacts of different lengths
or looking at a current density-like diagnostic in the terminal output.
"""
struct ContactCurrentAverage{N}
    boundary_symbols::NTuple{N, Symbol}
end

ContactCurrentAverage(boundary_symbol::Symbol) = ContactCurrentAverage((boundary_symbol,))
ContactCurrentAverage(boundary_symbol::Symbol, boundary_symbols::Symbol...) =
    ContactCurrentAverage((boundary_symbol, boundary_symbols...))

"""
    contact_current_normal(u, normal_direction, equations)

Return the normal charge-current flux for a single state `u`.

The incoming `normal_direction` may be scaled by the mesh geometry.  We normalize
it before calling the physical flux so this helper reports the current per unit
boundary length; Trixi's surface integration supplies the geometric surface
factor separately.
"""
@inline function contact_current_normal(u, normal_direction,
                                        equations::IsotropicFermiHarmonics2D)
    nrm = sqrt(normal_direction[1]^2 + normal_direction[2]^2)
    nrm > 0 || return zero(eltype(u))
    unit_normal = normal_direction / nrm
    return flux(u, unit_normal, equations)[1]
end

@inline function (::ContactNormalCurrent)(u, normal_direction, x, t,
                                          equations::IsotropicFermiHarmonics2D)
    return contact_current_normal(u, normal_direction, equations)
end

# Let Trixi do the geometric surface integration, all we have to do is give it the 
# appropriate pointwise flux to integrate and the correct boundary symbols
function Trixi.analyze(quantity::ContactCurrent, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    surface_integral = Trixi.AnalysisSurfaceIntegral(quantity.boundary_symbols,
                                                     ContactNormalCurrent())
    return Trixi.analyze(surface_integral, du, u, t, semi)
end

function Trixi.analyze(quantity::ContactCurrentAverage, du, u, t,
                       semi::Trixi.AbstractSemidiscretization)
    current = Trixi.analyze(ContactCurrent(quantity.boundary_symbols), du, u, t, semi)
    length = contact_boundary_length(semi, quantity.boundary_symbols)
    return current / length
end

function _contact_boundary_label(boundary_symbols)
    return join(string.(boundary_symbols), "_and_")
end

Trixi.pretty_form_utf(quantity::ContactCurrent) =
    "I_$(_contact_boundary_label(quantity.boundary_symbols))"
Trixi.pretty_form_ascii(quantity::ContactCurrent) =
    "I_$(_contact_boundary_label(quantity.boundary_symbols))"
Trixi.pretty_form_utf(quantity::ContactCurrentAverage) =
    "j_$(_contact_boundary_label(quantity.boundary_symbols))_avg"
Trixi.pretty_form_ascii(quantity::ContactCurrentAverage) =
    "j_$(_contact_boundary_label(quantity.boundary_symbols))_avg"


# it can also be helpful to be able to calculate the total boundary length; we should
# already know this from our geometry, but this seemed to be the best way to get Trixi
# to uunderstand it...
struct BoundaryLength end
@inline (::BoundaryLength)(u, normal_direction, x, t, equations) = one(eltype(u))

"""
    contact_boundary_length(semi, boundary_symbol)
    contact_boundary_length(semi, boundary_symbols)

Return the total geometric length of one or more named P4est boundaries.

This uses the same face quadrature weights and normal vectors that Trixi uses
for surface integrals, so the result is consistent with `ContactCurrent`.
"""
contact_boundary_length(semi::Trixi.AbstractSemidiscretization,
                        boundary_symbol::Symbol) =
    contact_boundary_length(semi, (boundary_symbol,))

function contact_boundary_length(semi::Trixi.AbstractSemidiscretization,
                                 boundary_symbols::NTuple{N, Symbol}) where {N}
    _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
    surface_integral = Trixi.AnalysisSurfaceIntegral(boundary_symbols, BoundaryLength())
    return Trixi.analyze(surface_integral, nothing, cache.boundaries.u, zero(Float64),
                         semi)
end
