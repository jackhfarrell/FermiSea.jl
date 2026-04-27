using HDF5

# ---------------------------------------------------------------------------
# Private helpers ported from the earlier archive implementation.
# ---------------------------------------------------------------------------

@inline function _lagrange_basis(nodes, xi)
    n     = length(nodes)
    basis = zeros(n)
    @inbounds for i in 1:n
        li = 1.0
        for j in 1:n
            j != i && (li *= (xi - nodes[j]) / (nodes[i] - nodes[j]))
        end
        basis[i] = li
    end
    return basis
end

function _interpolate_state!(out, basis_xi, basis_eta, element_u)
    fill!(out, 0.0)
    nvars = size(element_u, 1)
    n     = length(basis_xi)
    @inbounds for node_j in 1:n, node_i in 1:n
        w = basis_xi[node_i] * basis_eta[node_j]
        for v in 1:nvars
            out[v] += w * element_u[v, node_i, node_j]
        end
    end
    return out
end

function _map_point(basis_xi, basis_eta, ex, ey)
    x = 0.0; y = 0.0
    n = length(basis_xi)
    @inbounds for node_j in 1:n, node_i in 1:n
        w  = basis_xi[node_i] * basis_eta[node_j]
        x += w * ex[node_i, node_j]
        y += w * ey[node_i, node_j]
    end
    return x, y
end

# Invert the element coordinate map via Newton with finite-difference Jacobian.
# Returns (xi, eta, converged::Bool).
function _invert_element_map(nodes, ex, ey, x_target, y_target;
                             tol=1e-12, max_iter=20)
    xi  = 0.0
    eta = 0.0
    h   = 1e-6
    @inbounds for _ in 1:max_iter
        bx  = _lagrange_basis(nodes, xi)
        by  = _lagrange_basis(nodes, eta)
        xm, ym = _map_point(bx, by, ex, ey)
        rx = xm - x_target
        ry = ym - y_target
        abs(rx) <= tol && abs(ry) <= tol && return (xi, eta, true)

        bxp = _lagrange_basis(nodes, xi + h);  bxm = _lagrange_basis(nodes, xi - h)
        byp = _lagrange_basis(nodes, eta + h); bym = _lagrange_basis(nodes, eta - h)

        dxdxi,  dydxi  = let (a, b) = _map_point(bxp, by, ex, ey), (c, d) = _map_point(bxm, by, ex, ey)
            (a - c) / (2h), (b - d) / (2h)
        end
        dxdeta, dydeta = let (a, b) = _map_point(bx, byp, ex, ey), (c, d) = _map_point(bx, bym, ex, ey)
            (a - c) / (2h), (b - d) / (2h)
        end

        det = dxdxi * dydeta - dxdeta * dydxi
        abs(det) < 1e-15 && return (xi, eta, false)
        xi  -= (rx * dydeta - ry * dxdeta) / det
        eta -= (dxdxi * ry  - dydxi * rx)  / det
    end
    return (xi, eta, false)
end

# ---------------------------------------------------------------------------
# Output variable selection
# ---------------------------------------------------------------------------

_default_output_var_indices(num_vars) = 1:min(num_vars, 3)

# ---------------------------------------------------------------------------
# save_mesh_native
# ---------------------------------------------------------------------------

"""
    save_mesh_native(u_or_sol, semi, filename; refine=4) -> filename

Save the DG solution to `filename` (HDF5) on a mesh-native visualization grid.
Each element is subdivided `refine` times in each reference direction, giving
`(refine+1)²` points and `2·refine²` triangles per element.

Datasets: `x`, `y`, `triangles` (0-based Int32), and by default the leading
hydrodynamic variables named by `varnames(equations)` (e.g. `"a0"`, `"a1"`,
`"b1"`).

Attributes: `grid_type = "mesh_native_triangles"`, `refine`, `time`.
"""
function save_mesh_native(u_or_sol, semi, filename; refine=4)
    u_ode = u_or_sol isa AbstractVector ? u_or_sol : u_or_sol.u[end]
    t     = u_or_sol isa AbstractVector ? NaN       : float(u_or_sol.t[end])

    refine = Int(refine)
    refine >= 1 || throw(ArgumentError("refine must be >= 1"))

    _, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    num_vars         = Trixi.nvariables(equations)
    output_var_ids   = _default_output_var_indices(num_vars)
    num_output_vars  = length(output_var_ids)
    num_nodes        = Trixi.nnodes(solver)
    num_elems        = Trixi.nelements(solver, cache)
    vnames           = varnames(cons2cons, equations)
    output_vnames    = vnames[output_var_ids]
    u_wrap           = Trixi.wrap_array(u_ode, semi)
    bnodes           = solver.basis.nodes

    side          = refine + 1
    pts_per_elem  = side^2
    tris_per_elem = 2 * refine^2
    num_pts       = num_elems * pts_per_elem
    num_tris      = num_elems * tris_per_elem

    x_out  = Vector{Float64}(undef, num_pts)
    y_out  = Vector{Float64}(undef, num_pts)
    u_out  = [Vector{Float64}(undef, num_pts) for _ in 1:num_output_vars]
    tris   = Matrix{Int32}(undef, num_tris, 3)

    vis_nodes   = collect(range(-1.0, 1.0, length=side))
    basis_cache = [_lagrange_basis(bnodes, xi) for xi in vis_nodes]
    state_buf   = zeros(Float64, num_output_vars)
    tri_idx     = 1

    @inbounds for elem in 1:num_elems
        ex = zeros(num_nodes, num_nodes)
        ey = zeros(num_nodes, num_nodes)
        eu = zeros(num_output_vars, num_nodes, num_nodes)
        for j in 1:num_nodes, i in 1:num_nodes
            c = Trixi.get_node_coords(cache.elements.node_coordinates,
                                      equations, solver, i, j, elem)
            ex[i, j] = c[1];  ey[i, j] = c[2]
            v = Trixi.get_node_vars(u_wrap, equations, solver, i, j, elem)
            for k in 1:num_output_vars
                eu[k, i, j] = v[output_var_ids[k]]
            end
        end

        pt_base = (elem - 1) * pts_per_elem
        for eta_i in 1:side, xi_i in 1:side
            bx = basis_cache[xi_i]
            by = basis_cache[eta_i]
            pt = pt_base + (eta_i - 1) * side + xi_i
            x_out[pt], y_out[pt] = _map_point(bx, by, ex, ey)
            _interpolate_state!(state_buf, bx, by, eu)
            for k in 1:num_output_vars;  u_out[k][pt] = state_buf[k];  end
        end

        for cell_eta in 1:refine, cell_xi in 1:refine
            ll = pt_base + (cell_eta - 1) * side + cell_xi
            lr = ll + 1;  ul = ll + side;  ur = ul + 1
            tris[tri_idx, 1] = Int32(ll-1); tris[tri_idx, 2] = Int32(lr-1); tris[tri_idx, 3] = Int32(ur-1); tri_idx += 1
            tris[tri_idx, 1] = Int32(ll-1); tris[tri_idx, 2] = Int32(ur-1); tris[tri_idx, 3] = Int32(ul-1); tri_idx += 1
        end
    end

    @info "Writing mesh-native HDF5" file=filename
    h5open(filename, "w") do f
        f["x"]         = x_out
        f["y"]         = y_out
        f["triangles"] = tris
        for k in 1:num_output_vars
            f[output_vnames[k]] = u_out[k]
        end
        attributes(f)["grid_type"]               = "mesh_native_triangles"
        attributes(f)["refine"]                  = refine
        attributes(f)["connectivity_index_base"] = 0
        isnan(t) || (attributes(f)["time"] = t)
    end
    return filename
end

# ---------------------------------------------------------------------------
# save_cartesian
# ---------------------------------------------------------------------------

"""
    save_cartesian(u_or_sol, semi, filename; nvisnodes=200) -> filename

Save the DG solution to `filename` (HDF5) on a uniform Cartesian grid
(`nvisnodes × nvisnodes`) covering the mesh bounding box. Points outside the
domain are NaN; the boolean `mask` dataset marks in-domain points.

Datasets: `x`, `y`, `mask`, and by default the leading hydrodynamic variables
named by `varnames(equations)` (e.g. `"a0"`, `"a1"`, `"b1"`).

Attributes: `grid_type = "uniform_cartesian"`, `nvisnodes`, `time`.
"""
function save_cartesian(u_or_sol, semi, filename; nvisnodes=200)
    u_ode = u_or_sol isa AbstractVector ? u_or_sol : u_or_sol.u[end]
    t     = u_or_sol isa AbstractVector ? NaN       : float(u_or_sol.t[end])

    nvisnodes = Int(nvisnodes)

    _, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    num_vars         = Trixi.nvariables(equations)
    output_var_ids   = _default_output_var_indices(num_vars)
    num_output_vars  = length(output_var_ids)
    num_nodes        = Trixi.nnodes(solver)
    num_elems        = Trixi.nelements(solver, cache)
    vnames           = varnames(cons2cons, equations)
    output_vnames    = vnames[output_var_ids]
    nodes            = solver.basis.nodes
    u_wrap           = Trixi.wrap_array(u_ode, semi)

    # Precompute element coordinates, solution, and bounding boxes once.
    elem_x  = [zeros(num_nodes, num_nodes) for _ in 1:num_elems]
    elem_y  = [zeros(num_nodes, num_nodes) for _ in 1:num_elems]
    elem_u  = [zeros(num_output_vars, num_nodes, num_nodes) for _ in 1:num_elems]
    elem_bb = Matrix{Float64}(undef, num_elems, 4)  # xmin xmax ymin ymax
    xmin_g, xmax_g, ymin_g, ymax_g = Inf, -Inf, Inf, -Inf

    @inbounds for elem in 1:num_elems
        for j in 1:num_nodes, i in 1:num_nodes
            c = Trixi.get_node_coords(cache.elements.node_coordinates,
                                      equations, solver, i, j, elem)
            elem_x[elem][i, j] = c[1];  elem_y[elem][i, j] = c[2]
            v = Trixi.get_node_vars(u_wrap, equations, solver, i, j, elem)
            for k in 1:num_output_vars
                elem_u[elem][k, i, j] = v[output_var_ids[k]]
            end
        end
        xmn, xmx = extrema(elem_x[elem]);  ymn, ymx = extrema(elem_y[elem])
        elem_bb[elem, 1] = xmn;  elem_bb[elem, 2] = xmx
        elem_bb[elem, 3] = ymn;  elem_bb[elem, 4] = ymx
        xmin_g = min(xmin_g, xmn);  xmax_g = max(xmax_g, xmx)
        ymin_g = min(ymin_g, ymn);  ymax_g = max(ymax_g, ymx)
    end

    xs = range(xmin_g, xmax_g, length=nvisnodes)
    ys = range(ymin_g, ymax_g, length=nvisnodes)

    u_grid    = [fill(NaN, nvisnodes, nvisnodes) for _ in 1:num_output_vars]
    mask      = fill(false, nvisnodes, nvisnodes)
    state_buf = zeros(Float64, num_output_vars)

    @info "Cartesian grid evaluation" nvisnodes file=filename
    @inbounds for yi in 1:nvisnodes, xi in 1:nvisnodes
        x_t = xs[xi];  y_t = ys[yi]
        for elem in 1:num_elems
            (x_t < elem_bb[elem,1] || x_t > elem_bb[elem,2] ||
             y_t < elem_bb[elem,3] || y_t > elem_bb[elem,4]) && continue
            xi_ref, eta_ref, ok = _invert_element_map(
                nodes, elem_x[elem], elem_y[elem], x_t, y_t)
            (ok && abs(xi_ref) <= 1.0 + 1e-10 && abs(eta_ref) <= 1.0 + 1e-10) || continue
            bx = _lagrange_basis(nodes, xi_ref)
            by = _lagrange_basis(nodes, eta_ref)
            _interpolate_state!(state_buf, bx, by, elem_u[elem])
            mask[xi, yi] = true
            for k in 1:num_output_vars;  u_grid[k][xi, yi] = state_buf[k];  end
            break
        end
    end

    @info "Writing cartesian HDF5" file=filename
    h5open(filename, "w") do f
        f["x"]    = collect(xs)
        f["y"]    = collect(ys)
        f["mask"] = mask
        for k in 1:num_output_vars
            f[output_vnames[k]] = u_grid[k]
        end
        attributes(f)["grid_type"] = "uniform_cartesian"
        attributes(f)["nvisnodes"] = nvisnodes
        isnan(t) || (attributes(f)["time"] = t)
    end
    return filename
end
