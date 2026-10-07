using HDF5

"""
    save_surface(surface, filename)

Save a circular surface in a version-two HDF5 file and return `filename`.
We store its model parameters, momentum nodes, and response weights. We write
to a temporary file before replacing the destination.
"""
function save_surface(surface::CircularFermiSurface, filename)
    return write_hdf5(filename) do file
        attrs = attributes(file)
        # We keep the version-two grid tags so existing HDF5 files remain readable
        attrs["fermisea_format"] = "grid"
        attrs["format_version"] = 2
        _provenance(file)
        _write_surface(create_group(file, "grid"), surface)
    end
end

"""
    read_surface(filename)

Read a circular Fermi surface from a version-two HDF5 file. We check that the
stored nodes and weights agree with the model parameters.
"""
function read_surface(filename)
    return h5open(filename, "r") do file
        _format_check(file, "grid")
        _read_surface(file["grid"])
    end
end

"""
    save_plotdata(data, filename; compress=false)

Save observables from `PlotData` in a version-one HDF5 file and return `filename`.
We write to a temporary file before replacing the destination. The file keeps
values at the DG nodes, and we rebuild display refinement when reading it.
It does not contain the angular state. Compression preserves numeric values.
"""
function save_plotdata(data::PlotData, filename; compress::Bool = false)
    return write_hdf5(filename) do file
        attributes(file)["fermisea_format"] = "plotdata"
        attributes(file)["format_version"] = 1
        _provenance(file)
        _write_observables(file, data.nodal; compress)
    end
end

"""
    read_plotdata(filename; refine=1)

Return `PlotData` from an observable file or the observable section of a solution
file. Refinement interpolates stored fields, including nonlinear observables.
We cannot re-evaluate them on the original angular state. Plot the result with
Makie, for example `plot(read_plotdata(path), :particle_current)`.
"""
function read_plotdata(filename; refine::Integer = 1)
    return h5open(filename, "r") do file
        attrs = attributes(file)
        format = read(attrs["fermisea_format"])
        format in ("plotdata", "solution") || throw(
            ArgumentError(
                "expected fermisea_format = plotdata or solution, found $format"
            )
        )
        version = read(attrs["format_version"])
        (format == "plotdata" ? version == 1 : version == 2) ||
            throw(ArgumentError("unsupported $format format_version $version"))
        haskey(file, "observables") || throw(ArgumentError("file has no /observables"))
        make_plotdata(_read_nodal(file["observables"]), refine)
    end
end

"""
    save_solution(solution, filename; observables=(particle_density, particle_current),
                  state=true, compress=false)

Save the angular state, model, mesh, boundaries, solve diagnostics, and nodal
observables in a version-two HDF5 file. Return `filename`. We write to a temporary
file before replacing the destination. Set `state=false` to save only observables;
this requires at least one observable. We cannot save `CustomCollision` operators.
The stored `field` and channel `drive` datasets contain `qE`, with carrier charge
already included.
"""
function save_solution(solution::BoltzmannSolution, filename;
    observables =
    (particle_density, particle_current), state::Bool = true,
    compress::Bool = false)
    obs =
        isempty(observables) ? Pair{Symbol,Any}[] :
        normalized_observables(observables...)
    !state && isempty(obs) &&
        throw(ArgumentError("state = false requires at least one observable"))
    sources = solution.problem.source_terms
    _contains_custom(sources) && throw(ArgumentError("CustomCollision cannot be saved"))
    bcs = solution.problem.boundary_conditions
    bclist =
        bcs isa NamedTuple ? collect(values(bcs)) :
        bcs isa AbstractDict ? collect(values(bcs)) : (bcs,)
    any(_contains_custom, bclist) &&
        throw(ArgumentError("CustomCollision cannot be saved"))
    if state
        surface = solution.problem.semi.equations.surface
        for bc in bclist
            wall = bc isa PreparedMaxwellWall ? bc.wall : bc
            if wall isa IncomingProfile
                compatible_profile_surface(wall.profile.model.surface, surface) || throw(
                    ArgumentError(
                        "IncomingProfile channel surface is incompatible with " *
                            "the device surface"
                    )
                )
            end
        end
    end
    nodal = isempty(obs) ? nothing : nodal_fields(solution, obs)
    return write_hdf5(filename) do file
        attrs = attributes(file)
        attrs["fermisea_format"] = "solution"
        attrs["format_version"] = 2
        attrs["state"] = Int8(state)
        _provenance(file)
        nodal === nothing || _write_observables(file, nodal; compress)
        state || return
        problem = solution.problem
        semi = problem.semi
        _write_surface(create_group(file, "grid"), semi.equations.surface)
        mesh_group = create_group(file, "mesh")
        _write_mesh(mesh_group, semi.mesh, semi.cache.elements.node_coordinates)
        problem_group = create_group(file, "problem")
        attributes(problem_group)["degree"] = problem.degree
        _write_boundaries(
            create_group(problem_group, "boundary_conditions"),
            problem.boundary_conditions
        )
        source_group = create_group(problem_group, "sources")
        terms = _source_terms(problem.source_terms)
        attributes(source_group)["count"] = length(terms)
        foreach(i -> _write_source(source_group, i, terms[i]), eachindex(terms))
        solution_group = create_group(file, "solution")
        solution_attrs = attributes(solution_group)
        for (k, v) in
            (("residual", solution.residual), ("converged", Int8(solution.converged)),
            ("iterations", solution.iterations), ("tolerance", solution.tolerance),
            ("status", String(solution.status)))
            solution_attrs[k] = v
        end
        if compress
            HDF5.write(
                solution_group,
                "phi",
                solution.phi;
                chunk = (length(solution.phi),),
                deflate = 3
            )
        else
            solution_group["phi"] = solution.phi
        end
        _offsets(
            create_group(solution_group, "contact_offsets"),
            solution.contact_offsets
        )
        _offsets(
            create_group(solution_group, "contact_currents"),
            solution.contact_currents
        )
    end
end

"""
    read_solution(filename)

Rebuild a `BoltzmannSolution` from a version-two state file and check its mesh
coordinates. The stored electrical forcing already includes carrier charge, so
we use it directly. Use [`read_plotdata`](@ref) for files that contain only
observables. We rebuild preconditioner factors when the restored problem needs
them.
"""
function read_solution(filename)
    h5open(filename, "r") do file
        _format_check(file, "solution")
        Int(read(attributes(file)["state"])) == 1 ||
            throw(ArgumentError("solution file has state = 0; use read_plotdata"))
        surface = _read_surface(file["grid"])
        mesh_group = file["mesh"]
        mesh = _read_mesh(mesh_group)
        problem_group = file["problem"]
        model = BoltzmannEquation(surface)
        boundaries = _read_boundaries(
            problem_group["boundary_conditions"],
            model
        )
        sources = _read_sources(problem_group["sources"], model)
        problem = BoltzmannProblem(model, mesh, boundaries; source_terms = sources,
            degree = Int(read(attributes(problem_group)["degree"])))
        stored = read(mesh_group["node_coordinates"])
        isapprox(
            problem.semi.cache.elements.node_coordinates,
            stored;
            rtol = 1e-12,
            atol = 0
        ) ||
            throw(
                ArgumentError(
                    "rebuilt mesh does not match the stored node coordinates"
                )
            )
        solution_group = file["solution"]
        attrs = attributes(solution_group)
        return BoltzmannSolution(read(solution_group["phi"]), problem,
            Float64(read(attrs["residual"])),
            Bool(read(attrs["converged"])), Int(read(attrs["iterations"])),
            Float64(read(attrs["tolerance"])),
            Symbol(read(attrs["status"])),
            _read_offsets(solution_group["contact_offsets"]),
            _read_offsets(solution_group["contact_currents"]))
    end
end

_io_value(value::StaticArrays.StaticVector) = collect(value)
_io_value(value::Tuple) = collect(value)
_io_value(value) = value
_io_write(parent, name, value) = (parent[name] = _io_value(value))

function write_hdf5(writer, filename)
    path = abspath(String(filename))
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path))
    close(io)
    try
        # Replace the destination after HDF5 closes so a failed write leaves no partial
        # file
        h5open(temp, "w") do file
            writer(file)
        end
        mv(temp, path; force = true)
    catch
        rm(temp; force = true)
        rethrow()
    end
    return filename
end

function _provenance(file)
    group = create_group(file, "provenance")
    attrs = attributes(group)
    attrs["julia"] = string(VERSION)
    attrs["fermisea"] = string(Base.pkgversion(@__MODULE__))
    attrs["trixi"] = string(Base.pkgversion(Trixi))
    attrs["hdf5"] = string(Base.pkgversion(HDF5))
    return group
end

function _format_check(file, expected)
    attrs = attributes(file)
    format = read(attrs["fermisea_format"])
    version = read(attrs["format_version"])
    format == expected ||
        throw(ArgumentError("expected fermisea_format = $expected, found $format"))
    expected_version = expected == "plotdata" ? 1 : 2
    version == expected_version || throw(ArgumentError("expected format_version = $expected_version, found $version"))
end

_write_attr(obj, key, value::Symbol) = (attributes(obj)[key] = String(value))
_write_attr(obj, key, value) = (attributes(obj)[key] = value)
_read_symbol(obj, key) = Symbol(read(attributes(obj)[key]))

function _write_surface(group::HDF5.Group, surface::CircularFermiSurface)
    for name in (:momenta, :velocities, :energies, :weights, :theta)
        group[String(name)] = getproperty(surface, name)
    end
    for name in (:p_fermi, :v_fermi, :mu, :charge, :spin_degeneracy)
        attributes(group)[String(name)] = getproperty(surface, name)
    end
end

function _read_surface(group::HDF5.Group)
    attrs = attributes(group)
    surface = circular_fermi_surface(;
        p_fermi = read(attrs["p_fermi"]), v_fermi = read(attrs["v_fermi"]),
        mu = read(attrs["mu"]), charge = read(attrs["charge"]),
        spin_degeneracy = Int(read(attrs["spin_degeneracy"])),
        nangle = length(group["weights"]))
    for name in (:momenta, :velocities, :energies, :weights, :theta)
        stored = read(group[String(name)])
        expected = getproperty(surface, name)
        size(stored) == size(expected) &&
            isapprox(stored, expected; rtol = 64eps(Float64), atol = 0) ||
            throw(ArgumentError("stored $name disagree with the circular surface parameters"))
        # Preserve the stored quadrature exactly after checking its model assumptions
        copyto!(expected, stored)
    end
    return surface
end

# Solution and nodal-field encodings share the same atomic writer as surfaces
function _write_field(parent, name, value; compress = false)
    if value isa AbstractArray
        if compress && !isempty(value)
            HDF5.write(parent, name, value; chunk = size(value), deflate = 3)
        else
            parent[name] = value
        end
    elseif value isa Tuple && length(value) == 2
        _write_field(parent, name * "_x", value[1]; compress)
        _write_field(parent, name * "_y", value[2]; compress)
    elseif value isa NamedTuple
        group = create_group(parent, name)
        attributes(group)["kind"] = "namedtuple"
        attributes(group)["field_order"] = join(String.(keys(value)), ",")
        foreach(
            k -> _write_field(group, String(k), getproperty(value, k); compress),
            keys(value)
        )
    else
        throw(ArgumentError("unsupported observable field $(typeof(value))"))
    end
end

function _validate_field_names(fields)
    for (name, value) in pairs(fields)
        if value isa Tuple && length(value) == 2
            for suffix in ("_x", "_y")
                component = Symbol(name, suffix)
                haskey(fields, component) && throw(ArgumentError(
                    "observable field $component conflicts with a stored component of $name"))
            end
        elseif value isa NamedTuple
            _validate_field_names(value)
        end
    end
    return nothing
end

function _write_observables(parent, nodal::NodalFields; compress = false)
    _validate_field_names(nodal.values)
    group = create_group(parent, "observables")
    attributes(group)["names"] = String.(nodal.names)
    ng = create_group(group, "nodes")
    ng["x"] = nodal.x
    ng["y"] = nodal.y
    attributes(ng)["polydeg"] = length(nodal.nodes) - 1
    ng["reference_nodes"] = nodal.nodes
    n, _, ne = size(nodal.x)
    tris = Matrix{Int32}(undef, 3, 2 * (n - 1)^2 * ne)
    k = 0
    for e in 1:ne, j in 1:(n - 1), i in 1:(n - 1)
        ll = (e - 1) * n^2 + (j - 1) * n + i
        tris[:, k += 1] .= Int32.(ll .- 1 .+ (0, 1, n + 1))
        tris[:, k += 1] .= Int32.(ll .- 1 .+ (0, n + 1, n))
    end
    group["triangles"] = tris
    for name in nodal.names
        _write_field(group, String(name), nodal.values[name]; compress)
    end
    return group
end

function _read_field(parent, name)
    if haskey(parent, name)
        obj = parent[name]
        if obj isa HDF5.Group
            fields = split(read(attributes(obj)["field_order"]), ",")
            return (; (Symbol(k) => _read_field(obj, k) for k in fields)...)
        end
        return read(obj)
    elseif haskey(parent, name * "_x") && haskey(parent, name * "_y")
        return (read(parent[name * "_x"]), read(parent[name * "_y"]))
    end
    throw(ArgumentError("missing observable field $name"))
end

function _read_nodal(group)
    nodes = read(group["nodes/reference_nodes"])
    x, y = read(group["nodes/x"]), read(group["nodes/y"])
    names = Symbol.(read(attributes(group)["names"]))
    values =
        Dict{Symbol,Any}(name => _read_field(group, String(name)) for name in names)
    return NodalFields(nodes, x, y, names, values)
end

function _source_terms(raw)
    raw === nothing && return ()
    raw isa SourceTerms && return _source_terms(raw.terms)
    raw isa Tuple && return map(_source_terms, raw) |> Iterators.flatten |> collect
    raw isa PreparedLocalSource && return _source_terms(raw.operator)
    return (raw,)
end

function _contains_custom(x)
    x isa CustomCollision && return true
    x isa PreparedLocalSource && return _contains_custom(x.operator)
    x isa SourceTerms && return any(_contains_custom, x.terms)
    x isa Tuple && return any(_contains_custom, x)
    x isa IncomingProfile && return _contains_custom(x.profile.sources)
    return false
end

function _write_source(parent, i, source)
    source isa PreparedLocalSource && (source = source.operator)
    group = create_group(parent, lpad(string(i), 4, '0'))
    attrs = attributes(group)
    if source isa Callaway
        attrs["type"] = "Callaway"
        attrs["gamma_mr"] = source.gamma_mr
        attrs["gamma_mc"] = source.gamma_mc
    elseif source isa TomographicCollision
        attrs["type"] = "TomographicCollision"
        attrs["gamma_mr"] = source.gamma_mr
        attrs["gamma_mc"] = source.gamma_mc
        attrs["gamma_3"] = source.gamma_3
    elseif source isa MagneticField
        attrs["type"] = "MagneticField"
        attrs["field"] = source.field
        attrs["charge"] = source.charge
    elseif source isa ElectricDrive
        attrs["type"] = "ElectricDrive"
        _io_write(group, "field", source.charged_field)
    else
        throw(ArgumentError("unsupported source type $(typeof(source))"))
    end
end

function _read_sources(parent, model)
    term_count = Int(read(attributes(parent)["count"]))
    term_count == 0 && return nothing
    result = Any[]
    for i in 1:term_count
        source_group = parent[lpad(string(i), 4, '0')]
        attrs = attributes(source_group)
        source_type = read(attrs["type"])
        source = if source_type == "Callaway"
            Callaway(
                model;
                gamma_mr = read(attrs["gamma_mr"]),
                gamma_mc = read(attrs["gamma_mc"])
            )
        elseif source_type == "TomographicCollision"
            TomographicCollision(model; gamma_mr = read(attrs["gamma_mr"]),
                gamma_mc = read(attrs["gamma_mc"]), gamma_3 = read(attrs["gamma_3"]))
        elseif source_type == "MagneticField"
            MagneticField(model, read(attrs["field"]); charge = read(attrs["charge"]))
        elseif source_type == "ElectricDrive"
            # Stored forcing includes charge, so decoding bypasses the public field
            # constructor
            ElectricDrive(SVector{2,Float64}(read(source_group["field"])))
        else
            throw(ArgumentError("unsupported stored source type $source_type"))
        end
        push!(result, source)
    end
    return length(result) == 1 ? only(result) : Tuple(result)
end

function _write_boundary(group, bc)
    bc isa PreparedDiffuseWall && (bc = DiffuseWall())
    bc isa PreparedMaxwellWall && (bc = bc.wall)
    attrs = attributes(group)
    if bc isa FixedReservoir
        attrs["type"] = "FixedReservoir"
        attrs["delta_mu"] = bc.delta_mu
    elseif bc isa FloatingContact
        attrs["type"] = "FloatingContact"
        attrs["target_current"] = bc.target_current
    elseif bc isa DiffuseWall
        attrs["type"] = "DiffuseWall"
    elseif bc isa MaxwellWall
        attrs["type"] = "MaxwellWall"
        attrs["p_scatter"] = bc.p_scatter
    elseif bc isa IncomingProfile
        attrs["type"] = "IncomingProfile"
        _io_write(group, "origin", bc.origin)
        profile = bc.profile
        profile_group = create_group(group, "profile")
        for (key, value) in (("width", profile.width), ("axis", profile.axis),
            ("transverse", profile.transverse), ("drive", profile.charged_field),
            ("coordinates", profile.coordinates), ("cell_widths", profile.cell_widths),
            ("states", profile.states), ("mean_current", profile.mean_current),
            ("residual", profile.residual),
            ("wall_particle_flux", profile.wall_particle_flux))
            _io_write(profile_group, key, value)
        end
        _write_boundaries(create_group(profile_group, "walls"), profile.walls)
        stored_group = create_group(profile_group, "sources")
        terms = _source_terms(profile.sources)
        attributes(stored_group)["count"] = length(terms)
        foreach(i -> _write_source(stored_group, i, terms[i]), eachindex(terms))
    elseif bc === Trixi.boundary_condition_periodic
        attrs["type"] = "periodic"
    else
        throw(ArgumentError("unsupported boundary type $(typeof(bc))"))
    end
end

function _read_boundary(group, model)
    attrs = attributes(group)
    source_type = read(attrs["type"])
    source_type == "FixedReservoir" &&
        return FixedReservoir(read(attrs["delta_mu"]))
    source_type == "FloatingContact" && return FloatingContact(
        target_current = read(attrs["target_current"])
    )
    source_type == "DiffuseWall" && return DiffuseWall()
    source_type == "periodic" && return Trixi.boundary_condition_periodic
    if source_type == "MaxwellWall"
        return MaxwellWall(; p_scatter = read(attrs["p_scatter"]))
    end
    if source_type == "SpecularWall"
        return MaxwellWall(; p_scatter = 1 - read(attrs["specularity"]))
    end
    if source_type == "IncomingProfile"
        profile_group = group["profile"]
        walls = _read_boundaries(profile_group["walls"], model)
        wall_values = walls isa NamedTuple ? Tuple(values(walls)) : walls
        channel_sources = _read_sources(profile_group["sources"], model)
        prepared_sources = prepare_source(channel_sources)
        profile = ChannelProfile(model, Float64(read(profile_group["width"])),
            SVector{2,Float64}(read(profile_group["axis"])),
            SVector{2,Float64}(read(profile_group["transverse"])),
            map(prepare_boundary, wall_values), prepared_sources,
            Float64(read(profile_group["drive"])),
            read(profile_group["coordinates"]), read(profile_group["cell_widths"]),
            read(profile_group["states"]),
            SVector{2,Float64}(read(profile_group["mean_current"])),
            Float64(read(profile_group["residual"])),
            SVector{2,Float64}(read(profile_group["wall_particle_flux"])))
        return IncomingProfile(
            profile;
            origin = SVector{2,Float64}(read(group["origin"]))
        )
    end
    throw(ArgumentError("unsupported stored boundary type $source_type"))
end

function _write_boundaries(parent, boundaries)
    if boundaries isa Tuple
        attributes(parent)["kind"] = "tuple"
        attributes(parent)["count"] = length(boundaries)
        foreach(
            i -> _write_boundary(
                create_group(parent, lpad(string(i), 4, '0')),
                boundaries[i]
            ),
            eachindex(boundaries)
        )
    elseif boundaries isa NamedTuple || boundaries isa AbstractDict
        attributes(parent)["kind"] = "named"
        names = collect(keys(boundaries))
        attributes(parent)["names"] = String.(names)
        for name in names
            bc =
                boundaries isa AbstractDict ? boundaries[name] :
                getproperty(boundaries, name)
            _write_boundary(create_group(parent, String(name)), bc)
        end
    else
        attributes(parent)["kind"] = "single"
        _write_boundary(create_group(parent, "value"), boundaries)
    end
end

function _read_boundaries(parent, model)
    attrs = attributes(parent)
    kind = read(attrs["kind"])
    if kind == "single"
        return _read_boundary(parent["value"], model)
    elseif kind == "tuple"
        return Tuple(
            _read_boundary(parent[lpad(string(i), 4, '0')], model) for i in 1:Int(read(attrs["count"]))
        )
    end
    names = String.(read(attrs["names"]))
    return (;
        (
            Symbol(name) => _read_boundary(parent[name], model)
            for name in names
        )...
    )
end

function _write_mesh(group, mesh, coordinates)
    group["node_coordinates"] = coordinates
    if mesh isa Trixi.UnstructuredMesh2D
        isfile(mesh.filename) || throw(
            ArgumentError("unstructured mesh file $(mesh.filename) no longer exists")
        )
        # Embedded mesh text keeps a saved solution independent of its original mesh
        # path
        text = read(mesh.filename, String)
        attributes(group)["type"] = "UnstructuredMesh2D"
        attributes(group)["basename"] = basename(mesh.filename)
        attributes(group)["periodicity"] = Int8(mesh.periodicity)
        group["text"] = text
    elseif mesh isa Trixi.TreeMesh{2}
        tree = mesh.tree
        leaves = Trixi.leaf_cells(tree)
        levels = tree.levels[leaves]
        all(==(first(levels)), levels) ||
            throw(ArgumentError("only uniformly refined TreeMesh meshes can be saved"))
        attributes(group)["type"] = "TreeMesh"
        _io_write(
            group,
            "coordinates_min",
            tree.center_level_0 .- tree.length_level_0 / 2
        )
        _io_write(
            group,
            "coordinates_max",
            tree.center_level_0 .+ tree.length_level_0 / 2
        )
        _io_write(group, "periodicity", Int8.(tree.periodicity))
        attributes(group)["initial_refinement_level"] = first(levels)
        attributes(group)["n_cells_max"] = length(tree.levels)
    else
        throw(ArgumentError("unsupported mesh type $(typeof(mesh))"))
    end
end

function _read_mesh(group)
    attrs = attributes(group)
    source_type = read(attrs["type"])
    if source_type == "TreeMesh"
        periodicity = Tuple(Bool.(read(group["periodicity"])))
        mesh = Trixi.TreeMesh(Tuple(read(group["coordinates_min"])),
            Tuple(read(group["coordinates_max"]));
            initial_refinement_level = Int(read(attrs["initial_refinement_level"])),
            n_cells_max = Int(read(attrs["n_cells_max"])), periodicity)
    elseif source_type == "UnstructuredMesh2D"
        dir = mktempdir()
        path = joinpath(dir, String(read(attrs["basename"])))
        write(path, read(group["text"], String))
        mesh = Trixi.UnstructuredMesh2D(
            path;
            periodicity = Bool(read(attrs["periodicity"]))
        )
    else
        throw(ArgumentError("unsupported stored mesh type $source_type"))
    end
    return mesh
end

function _offsets(group, value)
    names = collect(keys(value))
    attributes(group)["names"] = join(String.(names), ",")
    for name in names
        attributes(group)[String(name)] = getproperty(value, name)
    end
end
function _read_offsets(group)
    encoded = String(read(attributes(group)["names"]))
    names = isempty(encoded) ? String[] : split(encoded, ",")
    return (;
        (Symbol(name) => Float64(read(attributes(group)[name])) for name in names)...
    )
end
