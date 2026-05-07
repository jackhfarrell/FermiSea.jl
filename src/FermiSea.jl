module FermiSea

using LinearAlgebra
using Logging
using LoggingExtras
using StaticArrays
using Trixi

import Trixi: AbstractEquations, DiscreteCallback
import Trixi: cons2cons, cons2entropy, cons2prim, entropy
import Trixi: flux, have_constant_speed, max_abs_speed_naive, max_abs_speeds
import Trixi: nvariables, varnames
import Trixi: residual_steady_state

include("equations/equations.jl")
include("auxiliary/auxiliary.jl")
include("callbacks_step/callbacks_step.jl")
include("visualization/visualization.jl")

export IsotropicFermiHarmonics2D, IsotropicHarmonicsFiniteT2D
export LinearCollisionMatrix, NonlinearBGKCollision, MagneticFieldSource, SourceTerms
export flux_electrostatic_nonconservative
export hydrodynamic_density, hydrodynamic_momentum, hydrodynamic_velocity
export hydrodynamic_chemical_potential_shift, hydrodynamic_fields
export MaxwellWallBC, OhmicContactBC, ChemicalPotentialContactBC, DensityContactBC
export FloatingProbeBC, CurrentContactBC
export SteadyStateResidual, ContactCurrent, ContactCurrentAverage
export contact_current_normal, contact_boundary_length
export MonitorCallback, FlushOutputCallback, make_flushing_logger
export save_mesh_native, save_cartesian

end
