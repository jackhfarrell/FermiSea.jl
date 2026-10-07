"""
We solve the steady linear Boltzmann equation for electron transport in
two-dimensional devices, using one circular Fermi surface at zero temperature.
Choose a mesh, collisions, and boundaries, then solve for the flow. We keep the
angular state unweighted and use consistent model units for all inputs.
"""
module FermiSea

using FFTW
using Krylov
using LinearAlgebra
using SparseArrays
using StaticArrays
using Trixi

include("surface.jl")
include("equations.jl")
include("collisions.jl")
include("local_operators.jl")
include("boundaries.jl")
include("channel.jl")
include("optimized.jl")
include("transport_sweep.jl")
include("steady.jl")
include("factorization.jl")
include("moment_sweep.jl")
include("moments.jl")
include("plot_data.jl")
include("sampling.jl")
include("io.jl")

export CircularFermiSurface, circular_fermi_surface,
       BoltzmannEquation, flux_upwind, streaming_fluxes!,
       Callaway, CustomCollision, TomographicCollision, MagneticField, ElectricDrive,
       collision_workspace, magnetic_workspace, prepare_source, SourceTerms,
       apply_collision!, apply_magnetic!, positive_collision,
       FixedReservoir, FloatingContact, DiffuseWall, MaxwellWall,
       IncomingProfile, developed_channel, ChannelProfile, channel_current,
       BoltzmannProblem, GMRES, TransportSweep, MomentSweep,
       AutomaticPreconditioner,
       BoltzmannSolution, solve, state, prepare_preconditioner, release_preconditioner!,
       weighted_norm, particle_density, particle_current, charge_current,
       momentum_moment, fit_local_equilibrium,
       contact_current, contact_particle_flux,
       SpatialSampler, nodal_state, PlotData, read_plotdata, save_plotdata,
       save_surface, read_surface, save_solution, read_solution

end
