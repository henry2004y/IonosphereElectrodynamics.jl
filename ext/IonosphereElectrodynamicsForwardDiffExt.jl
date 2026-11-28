module IonosphereElectrodynamicsForwardDiffExt

using ForwardDiff: Dual, Partials, partials, value
using IonosphereElectrodynamics:
    IonosphereElectrodynamics,
    IonosphereSolver,
    _BiCGSTABWorkspace,
    _FivePointHemisphereOp,
    _assemble_hemisphere_rhs!,
    _is_static_conductance,
    _matvec_stencil_only!,
    _similar_op,
    _smooth_jr!,
    _solve_bicgstab!,
    _solve_hemisphere_linear!,
    _solve_ionosphere_stateless,
    _unpack_hemisphere_phi!

function IonosphereElectrodynamics._solve_hemisphere_linear!(
        op::_FivePointHemisphereOp{Dual{Tag, V, N}},
        ws::_BiCGSTABWorkspace{Dual{Tag, V, N}},
        tol::Dual{Tag, V, N},
        max_iter::Int;
        rebuild_ilu::Bool = true,
    ) where {Tag, V <: Real, N}
    n = length(ws.rhs)
    op_p = _similar_op(V, op)
    ws_p = _BiCGSTABWorkspace{V}(n)
    @inbounds for k in 1:n
        op_p.cA[k] = value(op.cA[k])
        op_p.cB[k] = value(op.cB[k])
        op_p.cC[k] = value(op.cC[k])
        op_p.cD[k] = value(op.cD[k])
        op_p.cE[k] = value(op.cE[k])
        ws_p.rhs[k] = value(ws.rhs[k])
        ws_p.x[k] = value(ws.x[k])
    end

    tol_p = value(tol)
    ok, iters, resid_p = _solve_hemisphere_linear!(
        op_p, ws_p, tol_p, max_iter; rebuild_ilu = true,
    )

    if N == 0
        @inbounds for k in 1:n
            ws.x[k] = Dual{Tag, V, 0}(ws_p.x[k], Partials{0, V}(()))
        end
        return ok, iters, Dual{Tag, V, 0}(resid_p, Partials{0, V}(()))
    end

    x_p = copy(ws_p.x)
    op_dot = _similar_op(V, op)
    ws_t = _BiCGSTABWorkspace{V}(n)
    ax_dot = Vector{V}(undef, n)
    dx_cols = Matrix{V}(undef, n, N)

    for m in 1:N
        has_a_dot = false
        @inbounds for k in 1:n
            da = partials(op.cA[k], m)
            db = partials(op.cB[k], m)
            dc = partials(op.cC[k], m)
            dd = partials(op.cD[k], m)
            de = partials(op.cE[k], m)
            op_dot.cA[k] = da
            op_dot.cB[k] = db
            op_dot.cC[k] = dc
            op_dot.cD[k] = dd
            op_dot.cE[k] = de
            if !(iszero(da) && iszero(db) && iszero(dc) && iszero(dd) && iszero(de))
                has_a_dot = true
            end
        end

        if has_a_dot
            _matvec_stencil_only!(ax_dot, x_p, op_dot)
            @inbounds for k in 1:n
                ws_t.rhs[k] = partials(ws.rhs[k], m) - ax_dot[k]
                ws_t.x[k] = partials(ws.x[k], m)
            end
        else
            @inbounds for k in 1:n
                ws_t.rhs[k] = partials(ws.rhs[k], m)
                ws_t.x[k] = partials(ws.x[k], m)
            end
        end

        _solve_hemisphere_linear!(op_p, ws_t, tol_p, max_iter; rebuild_ilu = false)
        @inbounds for k in 1:n
            dx_cols[k, m] = ws_t.x[k]
        end
    end

    @inbounds for k in 1:n
        ps = ntuple(m -> dx_cols[k, m], Val(N))
        ws.x[k] = Dual{Tag, V, N}(x_p[k], Partials{N, V}(ps))
    end
    return ok, iters, Dual{Tag, V, N}(resid_p, zero(Partials{N, V}))
end

# Fast path: primal solver + Dual jr_in with static conductance reuses cached primal ILU(0)
function IonosphereElectrodynamics.solve_ionosphere(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{Dual{Tag, FT, N}};
        t::Real = zero(FT),
    ) where {FT <: AbstractFloat, Tag, N}
    is_static = _is_static_conductance(solver.config.conductance) && solver.op_assembled
    if !(t isa Dual) && is_static
        nt, np = solver.n_theta, solver.n_phi
        size(jr_in) == (nt, np) || throw(
            DimensionMismatch("jr_outward size $(size(jr_in)) != expected $((nt, np))"),
        )

        jr_val = Matrix{FT}(undef, nt, np)
        @inbounds for k in eachindex(jr_val, jr_in)
            jr_val[k] = value(jr_in[k])
        end
        phi_p = _solve_ionosphere_stateless(solver, jr_val, FT(t))

        if N == 0
            return [
                Dual{Tag, FT, 0}(phi_p[i, j], Partials{0, FT}(()))
                    for i in 1:nt, j in 1:np
            ]
        end

        dphi_all = Array{FT, 3}(undef, nt, np, N)
        dphi_buf = Matrix{FT}(undef, nt, np)
        for m in 1:N
            @inbounds for k in eachindex(solver.jr_outward, jr_in)
                solver.jr_outward[k] = partials(jr_in[k], m)
            end
            _smooth_jr!(solver)
            for is_north in (true, false)
                op = is_north ? solver.op_N : solver.op_S
                ws = is_north ? solver.ws_N : solver.ws_S
                i_min = is_north ? solver.i_min_N : solver.i_min_S
                _assemble_hemisphere_rhs!(
                    ws.rhs, ws.x, solver, solver.jr_outward, is_north, false,
                )
                _solve_bicgstab!(
                    op, ws, solver.config.tol, solver.config.max_iter, Val(false),
                )
                _unpack_hemisphere_phi!(
                    dphi_buf, ws.x, op, i_min, is_north, solver.n_theta_hem, nt,
                )
            end
            @inbounds for j in 1:np, i in 1:nt
                dphi_all[i, j, m] = dphi_buf[i, j]
            end
        end

        phi_dual = Matrix{Dual{Tag, FT, N}}(undef, nt, np)
        @inbounds for j in 1:np, i in 1:nt
            ps = ntuple(m -> dphi_all[i, j, m], Val(N))
            phi_dual[i, j] = Dual{Tag, FT, N}(phi_p[i, j], Partials{N, FT}(ps))
        end
        return phi_dual
    end

    # Dynamic conductance or Dual t: invoke generic promoted Dual solver
    # (which still uses our implicit SIMD _solve_hemisphere_linear! above)
    T = float(promote_type(Dual{Tag, FT, N}, typeof(t)))
    sol_T = IonosphereElectrodynamics.IonosphereSolver(
        solver.config;
        r_iono = solver.frame.r_iono,
        dipole_moment = solver.frame.dipole_moment,
        center = solver.frame.center,
        lat_boundary_deg = rad2deg(solver.lat_boundary_rad),
        FT = T,
    )
    return _solve_ionosphere_stateless(sol_T, jr_in, T(t))
end

end # module IonosphereElectrodynamicsForwardDiffExt
