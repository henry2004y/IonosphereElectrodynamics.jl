# ==============================================================================
# Exact 5-Point Transpose Operator, Transpose ILU(0) & Implicit Adjoint Engine
# ==============================================================================

@inline @muladd function _matvec_transpose_boundary_point!(
        y::Vector{FT},
        x::Vector{FT},
        op::_FivePointHemisphereOp{FT},
        k::Int,
        i::Int,
        j::Int,
        pole_adj::FT,
        base_im1::Int,
        base_jp1::Int,
        t::Int,
    ) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    is_north = op.is_north
    is_neumann = op.is_neumann
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE
    @inbounds begin
        acc = cA[k] * x[k]
        if i > 1
            km = base_im1 + t
            acc += cC[km] * x[km]
        elseif !is_north && is_neumann
            acc += cB[k] * x[k]
        end
        if i < n_th
            kp = base_jp1 + t + 1
            acc += cB[kp] * x[kp]
        elseif is_north && is_neumann
            acc += cC[k] * x[k]
        end
        if (is_north && i == 1) || (!is_north && i == n_th)
            acc += pole_adj
        end
        k_jm1 = j > 1 ? (base_im1 + t + 1) : op.wrap_prev[i]
        k_jp1 = j < n_ph ? (base_jp1 + t) : op.wrap_next[i]
        acc += cE[k_jm1] * x[k_jm1] + cD[k_jp1] * x[k_jp1]
        y[k] = acc
    end
    return nothing
end

@muladd function _matvec_transpose!(
        y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT},
    ) where {FT <: Real}
    n_ph = op.n_ph
    n_diag = op.n_diag
    diag_ptr = op.diag_ptr
    diag_i_lo = op.diag_i_lo
    pole_idx = op.pole_idx
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE
    c_pole = op.is_north ? cB : cC

    pole_adj = zero(FT)
    @inbounds @simd for j in 1:n_ph
        kp = pole_idx[j]
        pole_adj += c_pole[kp] * x[kp]
    end
    pole_adj /= FT(n_ph)

    @inbounds for d in 1:n_diag
        p_d = diag_ptr[d]
        L = diag_ptr[d + 1] - p_d
        i_lo = diag_i_lo[d]
        i_hi = i_lo + L - 1
        j_hi = d + 1 - i_lo
        j_lo = d + 1 - i_hi
        base_im1 = d > 1 ? (diag_ptr[d - 1] + (i_lo - diag_i_lo[d - 1]) - 1) : 0
        base_jp1 = d < n_diag ? (diag_ptr[d + 1] + (i_lo - diag_i_lo[d + 1])) : 0

        _matvec_transpose_boundary_point!(
            y, x, op, p_d, i_lo, j_hi, pole_adj, base_im1, base_jp1, 0,
        )
        if L >= 2
            @simd ivdep for t in 1:(L - 2)
                k = p_d + t
                kp = base_im1 + t
                kn = base_jp1 + t
                y[k] = cA[k] * x[k] + cC[kp] * x[kp] + cB[kn + 1] * x[kn + 1] +
                    cE[kp + 1] * x[kp + 1] + cD[kn] * x[kn]
            end
            _matvec_transpose_boundary_point!(
                y, x, op, p_d + L - 1, i_hi, j_lo, pole_adj, base_im1, base_jp1, L - 1,
            )
        end
    end
    return nothing
end

# Exact transpose of _apply_ilu0!: M^{-T} = D^{-1} L^{-T} U^{-T}
@muladd function _apply_ilu0_transpose!(
        z::Vector{FT}, r::Vector{FT}, op::_FivePointHemisphereOp{FT},
    ) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    n_diag = op.n_diag
    n = n_th * n_ph
    diag_ptr = op.diag_ptr
    diag_i_lo = op.diag_i_lo
    dInv, eL, fU, e1L, f1U = op.dInv, op.eL, op.fU, op.e1L, op.f1U

    @inbounds begin
        # 1. Solve U^T w = r (forward sweep along wavefronts)
        z[1] = r[1]
        for d in 2:n_diag
            p_d = diag_ptr[d]
            L = diag_ptr[d + 1] - p_d
            i_lo = diag_i_lo[d]
            i_hi = i_lo + L - 1
            j_lo = d + 1 - i_hi
            base_im1 = diag_ptr[d - 1] + (i_lo - diag_i_lo[d - 1]) - 1
            has_head = i_lo == 1
            has_tail = j_lo == 1

            if has_head
                kp1 = base_im1 + 1
                z[p_d] = r[p_d] - f1U[kp1] * z[kp1]
            end

            t_start = has_head ? 1 : 0
            t_end = has_tail ? (L - 2) : (L - 1)
            @simd ivdep for t in t_start:t_end
                k = p_d + t
                kp = base_im1 + t
                z[k] = r[k] - fU[kp] * z[kp] - f1U[kp + 1] * z[kp + 1]
            end

            if has_tail
                k = p_d + L - 1
                kp = base_im1 + L - 1
                z[k] = r[k] - fU[kp] * z[kp]
            end
        end

        # 2. Solve L^T y = w (backward sweep along wavefronts)
        for d in (n_diag - 1):-1:1
            p_d = diag_ptr[d]
            L = diag_ptr[d + 1] - p_d
            i_lo = diag_i_lo[d]
            i_hi = i_lo + L - 1
            j_hi = d + 1 - i_lo
            base_jp1 = diag_ptr[d + 1] + (i_lo - diag_i_lo[d + 1])
            has_head = j_hi == n_ph
            has_tail = i_hi == n_th

            if has_head
                kn1 = base_jp1 + 1
                z[p_d] -= eL[kn1] * z[kn1]
            end

            t_start = has_head ? 1 : 0
            t_end = has_tail ? (L - 2) : (L - 1)
            @simd ivdep for t in t_start:t_end
                k = p_d + t
                kn = base_jp1 + t
                z[k] = z[k] - e1L[kn] * z[kn] - eL[kn + 1] * z[kn + 1]
            end

            if has_tail
                k = p_d + L - 1
                kn = base_jp1 + L - 1
                z[k] -= e1L[kn] * z[kn]
            end
        end

        # 3. Scale by D^{-1}
        @simd for k in 1:n
            z[k] *= dInv[k]
        end
    end
    return nothing
end

# Exact transpose of _smooth_jr!
@muladd function _smooth_jr_adjoint!(
        bar_jr::Matrix{FT}, bar_tmp::Matrix{FT}, n_pass::Int,
    ) where {FT <: Real}
    n_pass <= 0 && return nothing
    nt, np = size(bar_jr)
    inv_np = one(FT) / FT(np)

    for _ in 1:n_pass
        # Pull back pole copies into rows 2 and nt - 1, clearing rows 1 and nt
        north_bar = zero(FT)
        south_bar = zero(FT)
        @inbounds for j in 1:np
            north_bar += bar_jr[1, j]
            south_bar += bar_jr[nt, j]
        end
        north_bar *= inv_np
        south_bar *= inv_np

        @inbounds for j in 1:np
            bar_tmp[1, j] = zero(FT)
            bar_tmp[nt, j] = zero(FT)
            for i in 2:(nt - 1)
                bar_tmp[i, j] = bar_jr[i, j]
            end
            bar_tmp[2, j] += north_bar
            bar_tmp[nt - 1, j] += south_bar
        end

        # Pull back 5-point interior stencil
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            bar_jr[1, j] = FT(0.125) * bar_tmp[2, j]
            @simd for i in 2:(nt - 1)
                bar_jr[i, j] = FT(0.5) * bar_tmp[i, j] + FT(0.125) * (
                    bar_tmp[i - 1, j] + bar_tmp[i + 1, j] + bar_tmp[i, jm] + bar_tmp[i, jp]
                )
            end
            bar_jr[nt, j] = FT(0.125) * bar_tmp[nt - 1, j]
        end
    end
    return nothing
end

# Conductance-sensitivity trait: whether evaluate_conductance depends on jr_outward
@inline _conductance_depends_on_jr(::AbstractConductanceModel) = false
@inline _conductance_depends_on_jr(::AuroralConductance) = true

@inline @muladd function _conductance_jr_deriv(
        m::AbstractConductanceModel,
        theta::FT,
        phi::FT,
        r_hat::SVector{3, FT},
        t::FT,
        jr_outward::FT,
    ) where {FT <: Real}
    sp, sh = evaluate_conductance(m, theta, phi, r_hat, t; jr_outward = jr_outward)
    return (FT(sp), FT(sh), zero(FT), zero(FT))
end

@inline @muladd function _conductance_jr_deriv(
        m::AuroralConductance,
        theta::FT,
        phi::FT,
        r_hat::SVector{3, FT},
        t::FT,
        jr_outward::FT,
    ) where {FT <: Real}
    sp_b, sh_b, dsp_b, dsh_b = _conductance_jr_deriv(
        m.base, theta, phi, r_hat, t, jr_outward,
    )
    sp_max = FT(m.sigma_P_max)
    if jr_outward <= zero(FT) || sp_max <= zero(FT)
        return (sp_b, sh_b, dsp_b, dsh_b)
    end
    inv_fs = FT(m.inv_fac_scale)
    h_rat = FT(m.hall_ratio)
    ratio = jr_outward * inv_fs
    if m.mode === :tanh_sqrt
        s_rat = sqrt(ratio)
        f_aur = tanh(s_rat)
        df_dratio = (one(FT) - f_aur * f_aur) / (FT(2) * s_rat)
    else
        inv_rp1 = inv(ratio + one(FT))
        f_aur = ratio * inv_rp1
        df_dratio = inv_rp1 * inv_rp1
    end
    sp_aur = sp_max * f_aur
    sh_aur = h_rat * sp_aur
    dsp_aur = sp_max * df_dratio * inv_fs
    dsh_aur = h_rat * dsp_aur

    sp = sqrt(sp_b * sp_b + sp_aur * sp_aur)
    sh2 = sh_b * sh_b + sh_aur * sh_aur
    sh = sh2 > zero(FT) ? sqrt(sh2) : sh_b

    dsp = (sp_b * dsp_b + sp_aur * dsp_aur) / sp
    dsh = sh > zero(FT) ? ((sh_b * dsh_b + sh_aur * dsh_aur) / sh) : zero(FT)
    return (sp, sh, dsp, dsh)
end

@inline @muladd function _dipole_conductance_tensor_vjp(
        bar_stt::FT,
        bar_stp::FT,
        bar_spp::FT,
        sp::FT,
        sh::FT,
        cos_theta::FT,
        sn2::FT,
        cs2::FT,
        cs3::FT,
        cs4::FT,
        dipole_sign::FT,
        s0::FT,
        s0_ratio::FT,
        has_s0_par::Bool,
        use_dip_angle::Bool,
    ) where {FT <: Real}
    hall_sign = -dipole_sign
    if !use_dip_angle
        sT = _scalar_float_type(FT)
        eps_eq = FT(max(floatmin(sT), sT(1.0e-12)))
        hemi = abs(cos_theta) <= eps_eq ? zero(FT) :
            (cos_theta > zero(FT) ? one(FT) : -one(FT))
        return (bar_stt + bar_spp, hall_sign * hemi * bar_stp)
    end
    C = FT(4) * s0 * cs2 + sp * sn2
    inv_C = inv(C)
    bar_inv_C = bar_stt * (s0 * sp * cs3) +
        bar_stp * (hall_sign * FT(2) * s0 * sh * cos_theta * cs4) +
        bar_spp * (sh * sh * sn2)
    bar_C = -bar_inv_C * inv_C * inv_C
    bar_sp = bar_stt * (s0 * cs3 * inv_C) + bar_spp + bar_C * sn2
    bar_sh = bar_stp * (hall_sign * FT(2) * s0 * cos_theta * cs4 * inv_C) +
        bar_spp * (FT(2) * sh * sn2 * inv_C)
    if !has_s0_par
        bar_s0 = bar_stt * (sp * cs3 * inv_C) +
            bar_stp * (hall_sign * FT(2) * sh * cos_theta * cs4 * inv_C) +
            bar_C * (FT(4) * cs2)
        bar_sp += s0_ratio * bar_s0
    end
    return (bar_sp, bar_sh)
end

@muladd function _accumulate_hemisphere_op_vjp!(
        bar_stt::Matrix{FT},
        bar_spp::Matrix{FT},
        bar_dtt_dth::Matrix{FT},
        bar_dtp_dth::Matrix{FT},
        bar_dtp_dph::Matrix{FT},
        bar_dpp_dph::Matrix{FT},
        lambda::Vector{FT},
        x_primal::Vector{FT},
        op::_FivePointHemisphereOp{FT},
        solver::IonosphereSolver{FT},
        is_north::Bool,
    ) where {FT <: Real}
    i_min = is_north ? solver.i_min_N : solver.i_min_S
    n_th = op.n_th
    n_ph = op.n_ph
    is_neumann = op.is_neumann
    k_pin = op.k_pin
    wf_idx = op.wf_idx
    pole_idx = op.pole_idx
    sin_th = solver.sin_theta
    cos_th = solver.cos_theta
    sin_th2 = solver.sin_theta2

    inv_dth2 = solver.inv_dtheta * solver.inv_dtheta
    inv_dph2 = solver.inv_dphi * solver.inv_dphi
    inv_2dth = FT(0.5) * solver.inv_dtheta
    inv_2dph = FT(0.5) * solver.inv_dphi

    pole_avg = zero(FT)
    @inbounds @simd for j in 1:n_ph
        pole_avg += x_primal[pole_idx[j]]
    end
    pole_avg /= FT(n_ph)

    @inbounds for j in 1:n_ph
        jm = j == 1 ? n_ph : j - 1
        jp = j == n_ph ? 1 : j + 1
        for i_loc in 1:n_th
            k = wf_idx[i_loc, j]
            if is_neumann && k == k_pin
                continue
            end
            x_k = x_primal[k]
            x_im1 = if i_loc > 1
                x_primal[wf_idx[i_loc - 1, j]]
            elseif is_north
                pole_avg
            else
                is_neumann ? x_k : zero(FT)
            end
            x_ip1 = if i_loc < n_th
                x_primal[wf_idx[i_loc + 1, j]]
            elseif is_north
                is_neumann ? x_k : zero(FT)
            else
                pole_avg
            end
            x_jm1 = x_primal[wf_idx[i_loc, jm]]
            x_jp1 = x_primal[wf_idx[i_loc, jp]]

            ml = -lambda[k]
            bcA = ml * x_k
            bcB = ml * x_im1
            bcC = ml * x_ip1
            bcD = ml * x_jm1
            bcE = ml * x_jp1

            b_th2 = -FT(2) * bcA + bcB + bcC
            b_th1 = -bcB + bcC
            b_ph2 = -FT(2) * bcA + bcD + bcE
            b_ph1 = -bcD + bcE

            i = i_min + i_loc - 1
            sn = sin_th[i]
            cs = cos_th[i]
            sn2 = sin_th2[i]

            bar_stt[i, j] += b_th2 * sn2 * inv_dth2 + b_th1 * (sn * cs * inv_2dth)
            bar_spp[i, j] += b_ph2 * inv_dph2
            bar_dtt_dth[i, j] += b_th1 * (sn2 * inv_2dth)
            bar_dtp_dph[i, j] += b_th1 * (-sn * inv_2dth)
            bar_dtp_dth[i, j] += b_ph1 * (sn * inv_2dph)
            bar_dpp_dph[i, j] += b_ph1 * inv_2dph
        end
    end
    return nothing
end

"""
Compute the exact reverse-mode vector-Jacobian product `bar_jr = (∂Φ/∂jr)^T * bar_phi`
for `phi = solve_ionosphere(solver, jr_in; t)` in `O(1)` adjoint linear solves.
"""
@muladd function _solve_ionosphere_vjp(
        solver::IonosphereSolver{FT},
        bar_phi::AbstractMatrix{<:Real},
        t::FT;
        op_N::_FivePointHemisphereOp{FT} = solver.op_N,
        op_S::_FivePointHemisphereOp{FT} = solver.op_S,
        x_N::Union{Nothing, Vector{FT}} = nothing,
        x_S::Union{Nothing, Vector{FT}} = nothing,
        jr_s::Union{Nothing, Matrix{FT}} = nothing,
    ) where {FT <: Real}
    nt, np = solver.n_theta, solver.n_phi
    size(bar_phi) == (nt, np) ||
        throw(DimensionMismatch("bar_phi size $(size(bar_phi)) != expected $((nt, np))"))

    bar_jr_s = zeros(FT, nt, np)
    r_iono2 = solver.frame.r_iono * solver.frame.r_iono
    sin_th = solver.sin_theta
    sin_th2 = solver.sin_theta2
    inv_np = one(FT) / FT(np)
    has_jr_cond = _conductance_depends_on_jr(solver.config.conductance)

    bar_stt = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_stp = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_spp = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_dtt_dth = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_dtp_dth = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_dtp_dph = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)
    bar_dpp_dph = has_jr_cond ? zeros(FT, nt, np) : Matrix{FT}(undef, 0, 0)

    for is_north in (true, false)
        op = is_north ? op_N : op_S
        ws = is_north ? solver.ws_N : solver.ws_S
        i_min = is_north ? solver.i_min_N : solver.i_min_S
        i_max = is_north ? solver.i_max_N : solver.i_max_S
        n_th = op.n_th
        wf_idx = op.wf_idx

        # 1. Pack bar_phi (including pole-average pullback) into ws.rhs
        i_pole = is_north ? 1 : nt
        i_adj = is_north ? 2 : nt - 1
        pole_bar = zero(FT)
        @inbounds for j in 1:np
            pole_bar += FT(bar_phi[i_pole, j])
        end
        pole_bar *= inv_np

        rhs = ws.rhs
        @inbounds for j in 1:np
            for i_loc in 1:n_th
                i = i_min + i_loc - 1
                k = wf_idx[i_loc, j]
                v = FT(bar_phi[i, j])
                rhs[k] = i == i_adj ? (v + pole_bar) : v
            end
        end
        fill!(ws.x, zero(FT))

        # 2. Solve A^T * lambda = bar_x using transpose ILU(0)-BiCGSTAB (0 B)
        _solve_bicgstab!(op, ws, solver.config.tol, solver.config.max_iter, Val(true))
        lambda = ws.x

        # 3. Pull back lambda through rhs & optional hemispheric neutralization
        if solver.config.neutralize_fac || op.is_neumann
            den = zero(FT)
            g_sum = zero(FT)
            @inbounds for j in 1:np
                for i_loc in 1:n_th
                    i = i_min + i_loc - 1
                    den += sin_th[i]
                    k = wf_idx[i_loc, j]
                    if !(op.is_neumann && k == op.k_pin)
                        g = lambda[k] * r_iono2 * sin_th2[i]
                        bar_jr_s[i, j] += g
                        g_sum += g
                    end
                end
            end
            if den > zero(FT)
                shift_bar = g_sum / den
                @inbounds for j in 1:np
                    for i in i_min:i_max
                        bar_jr_s[i, j] -= shift_bar * sin_th[i]
                    end
                end
            end
        else
            @inbounds for j in 1:np
                for i_loc in 1:n_th
                    i = i_min + i_loc - 1
                    k = wf_idx[i_loc, j]
                    bar_jr_s[i, j] += lambda[k] * r_iono2 * sin_th2[i]
                end
            end
        end

        # 4. Dynamic conductance path: accumulate bar_A = -lambda * x_primal^T
        if has_jr_cond
            xp = is_north ? x_N : x_S
            _accumulate_hemisphere_op_vjp!(
                bar_stt, bar_spp, bar_dtt_dth, bar_dtp_dth, bar_dtp_dph, bar_dpp_dph,
                lambda, xp, op, solver, is_north,
            )
        end
    end

    if has_jr_cond
        # Pull back central differences of conductance tensor
        inv_2dth = FT(0.5) * solver.inv_dtheta
        inv_2dph = FT(0.5) * solver.inv_dphi
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            for i in 2:(nt - 1)
                dt = bar_dtt_dth[i, j] * inv_2dth
                bar_stt[i + 1, j] += dt
                bar_stt[i - 1, j] -= dt
                dp = bar_dtp_dth[i, j] * inv_2dth
                bar_stp[i + 1, j] += dp
                bar_stp[i - 1, j] -= dp
            end
            for i in 1:nt
                dp = bar_dtp_dph[i, j] * inv_2dph
                bar_stp[i, jp] += dp
                bar_stp[i, jm] -= dp
                pp = bar_dpp_dph[i, j] * inv_2dph
                bar_spp[i, jp] += pp
                bar_spp[i, jm] -= pp
            end
        end

        # Pull back dipole conductance tensor & auroral conductance into bar_jr_s
        cfg = solver.config
        cond = cfg.conductance
        sT = _scalar_float_type(FT)
        sp_floor = max(cfg.sigma_P_floor, FT(max(floatmin(sT), sT(1.0e-12))))
        dip_sign = solver.frame.dipole_sign
        use_dip = cfg.use_dip_angle
        s0_ratio = cfg.sigma_0_ratio
        s0_par_raw = cfg.sigma_parallel
        has_s0_par = s0_par_raw !== nothing
        s0_par_val = s0_par_raw === nothing ? zero(FT) : max(s0_par_raw, FT(10))

        theta = solver.theta
        phi = solver.phi
        cos_th = solver.cos_theta
        cos_th2 = solver.cos_theta2
        dip_cs3 = solver.dip_cs3
        dip_cs4 = solver.dip_cs4
        r_hat_sim = solver.r_hat_sim
        jr_src = jr_s::Matrix{FT}

        @inbounds for j in 1:np
            ph = phi[j]
            for i in 1:nt
                rh = SVector{3, FT}(
                    r_hat_sim[1, i, j],
                    r_hat_sim[2, i, j],
                    r_hat_sim[3, i, j],
                )
                sp_raw, sh_raw, dsp_djr, dsh_djr = _conductance_jr_deriv(
                    cond, theta[i], ph, rh, t, jr_src[i, j],
                )
                sp_clamped = max(sp_raw, sp_floor)
                s0 = has_s0_par ? s0_par_val : (s0_ratio * sp_clamped)
                bar_sp, bar_sh = _dipole_conductance_tensor_vjp(
                    bar_stt[i, j], bar_stp[i, j], bar_spp[i, j],
                    sp_clamped, sh_raw, cos_th[i], sin_th2[i], cos_th2[i],
                    dip_cs3[i], dip_cs4[i], dip_sign, s0, s0_ratio, has_s0_par, use_dip,
                )
                if sp_raw < sp_floor
                    bar_sp = zero(FT)
                end
                bar_jr_s[i, j] += bar_sp * dsp_djr + bar_sh * dsh_djr
            end
        end
    end

    # 5. Pull back FAC Laplacian smoothing
    if solver.config.n_smooth_fac > 0
        _smooth_jr_adjoint!(bar_jr_s, solver.jr_work, solver.config.n_smooth_fac)
    end
    return bar_jr_s
end

@inline @muladd function _dipole_conductance_tensor_jvp(
        dsp::FT,
        dsh::FT,
        sp::FT,
        sh::FT,
        cos_theta::FT,
        sn2::FT,
        cs2::FT,
        cs3::FT,
        cs4::FT,
        dipole_sign::FT,
        s0::FT,
        ds0::FT,
        use_dip_angle::Bool,
    ) where {FT <: Real}
    hall_sign = -dipole_sign
    if !use_dip_angle
        sT = _scalar_float_type(FT)
        eps_eq = FT(max(floatmin(sT), sT(1.0e-12)))
        hemi = abs(cos_theta) <= eps_eq ? zero(FT) :
            (cos_theta > zero(FT) ? one(FT) : -one(FT))
        return (dsp, hall_sign * hemi * dsh, dsp)
    end
    C = FT(4) * s0 * cs2 + sp * sn2
    dC = FT(4) * ds0 * cs2 + dsp * sn2
    inv_C = inv(C)
    d_inv_C = -dC * inv_C * inv_C

    num_tt = s0 * sp * cs3
    dnum_tt = (ds0 * sp + s0 * dsp) * cs3
    dstt = dnum_tt * inv_C + num_tt * d_inv_C

    fac_tp = hall_sign * FT(2) * cos_theta * cs4
    num_tp = fac_tp * s0 * sh
    dnum_tp = fac_tp * (ds0 * sh + s0 * dsh)
    dstp = dnum_tp * inv_C + num_tp * d_inv_C

    num_pp = sh * sh * sn2
    dnum_pp = (FT(2) * sh * dsh) * sn2
    dspp = dsp + dnum_pp * inv_C + num_pp * d_inv_C
    return (dstt, dstp, dspp)
end

"""
Compute the exact forward-mode Jacobian-vector product `dphi = (∂Φ/∂jr) * djr`
for `phi = solve_ionosphere(solver, jr_in; t)` reusing `op_N`, `op_S`, and `x_N`, `x_S`.
"""
@muladd function _solve_ionosphere_jvp(
        solver::IonosphereSolver{FT},
        djr_in::AbstractMatrix{<:Real},
        t::FT;
        op_N::_FivePointHemisphereOp{FT} = solver.op_N,
        op_S::_FivePointHemisphereOp{FT} = solver.op_S,
        x_N::Union{Nothing, Vector{FT}} = nothing,
        x_S::Union{Nothing, Vector{FT}} = nothing,
        jr_s::Union{Nothing, Matrix{FT}} = nothing,
    ) where {FT <: Real}
    nt, np = solver.n_theta, solver.n_phi
    size(djr_in) == (nt, np) ||
        throw(DimensionMismatch("djr_in size $(size(djr_in)) != expected $((nt, np))"))

    djr_s = Matrix{FT}(undef, nt, np)
    copyto!(djr_s, djr_in)
    if solver.config.n_smooth_fac > 0
        _smooth_jr!(djr_s, solver.jr_work, solver.config.n_smooth_fac)
    end

    has_jr_cond = _conductance_depends_on_jr(solver.config.conductance)
    dstt = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    dstp = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    dspp = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    ddtt_dth = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    ddtp_dth = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    ddtp_dph = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)
    ddpp_dph = has_jr_cond ? Matrix{FT}(undef, nt, np) : Matrix{FT}(undef, 0, 0)

    if has_jr_cond
        cfg = solver.config
        cond = cfg.conductance
        sT = _scalar_float_type(FT)
        sp_floor = max(cfg.sigma_P_floor, FT(max(floatmin(sT), sT(1.0e-12))))
        dip_sign = solver.frame.dipole_sign
        use_dip = cfg.use_dip_angle
        s0_ratio = cfg.sigma_0_ratio
        s0_par_raw = cfg.sigma_parallel
        has_s0_par = s0_par_raw !== nothing
        s0_par_val = s0_par_raw === nothing ? zero(FT) : max(s0_par_raw, FT(10))

        theta = solver.theta
        phi = solver.phi
        cos_th = solver.cos_theta
        sin_th2 = solver.sin_theta2
        cos_th2 = solver.cos_theta2
        dip_cs3 = solver.dip_cs3
        dip_cs4 = solver.dip_cs4
        r_hat_sim = solver.r_hat_sim
        jr_src = jr_s::Matrix{FT}

        @inbounds for j in 1:np
            ph = phi[j]
            for i in 1:nt
                rh = SVector{3, FT}(
                    r_hat_sim[1, i, j],
                    r_hat_sim[2, i, j],
                    r_hat_sim[3, i, j],
                )
                sp_raw, sh_raw, dsp_djr, dsh_djr = _conductance_jr_deriv(
                    cond, theta[i], ph, rh, t, jr_src[i, j],
                )
                sp_clamped = max(sp_raw, sp_floor)
                dsp = sp_raw < sp_floor ? zero(FT) : (dsp_djr * djr_s[i, j])
                dsh = dsh_djr * djr_s[i, j]
                s0 = has_s0_par ? s0_par_val : (s0_ratio * sp_clamped)
                ds0 = has_s0_par ? zero(FT) : (s0_ratio * dsp)
                dt, dp, pp = _dipole_conductance_tensor_jvp(
                    dsp, dsh, sp_clamped, sh_raw, cos_th[i], sin_th2[i], cos_th2[i],
                    dip_cs3[i], dip_cs4[i], dip_sign, s0, ds0, use_dip,
                )
                dstt[i, j] = dt
                dstp[i, j] = dp
                dspp[i, j] = pp
            end
        end

        inv_2dth = FT(0.5) * solver.inv_dtheta
        inv_2dph = FT(0.5) * solver.inv_dphi
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            ddtt_dth[1, j] = zero(FT)
            ddtp_dth[1, j] = zero(FT)
            ddtt_dth[nt, j] = zero(FT)
            ddtp_dth[nt, j] = zero(FT)
            for i in 2:(nt - 1)
                ddtt_dth[i, j] = (dstt[i + 1, j] - dstt[i - 1, j]) * inv_2dth
                ddtp_dth[i, j] = (dstp[i + 1, j] - dstp[i - 1, j]) * inv_2dth
            end
            for i in 1:nt
                ddtp_dph[i, j] = (dstp[i, jp] - dstp[i, jm]) * inv_2dph
                ddpp_dph[i, j] = (dspp[i, jp] - dspp[i, jm]) * inv_2dph
            end
        end
    end

    dphi_out = zeros(FT, nt, np)
    inv_dth2 = solver.inv_dtheta * solver.inv_dtheta
    inv_dph2 = solver.inv_dphi * solver.inv_dphi
    inv_2dth = FT(0.5) * solver.inv_dtheta
    inv_2dph = FT(0.5) * solver.inv_dphi
    sin_th = solver.sin_theta
    cos_th = solver.cos_theta
    sin_th2 = solver.sin_theta2

    for is_north in (true, false)
        op = is_north ? op_N : op_S
        ws = is_north ? solver.ws_N : solver.ws_S
        i_min = is_north ? solver.i_min_N : solver.i_min_S
        _assemble_hemisphere_rhs!(ws.rhs, ws.x, solver, djr_s, is_north, false)

        if has_jr_cond
            op_dot = _similar_op(FT, op)
            n_th = op.n_th
            wf_idx = op.wf_idx
            cA, cB, cC, cD, cE = op_dot.cA, op_dot.cB, op_dot.cC, op_dot.cD, op_dot.cE
            @inbounds for j in 1:np
                for i_loc in 1:n_th
                    i = i_min + i_loc - 1
                    k = wf_idx[i_loc, j]
                    sn = sin_th[i]
                    cs = cos_th[i]
                    sn2 = sin_th2[i]
                    t_th2 = dstt[i, j] * sn2 * inv_dth2
                    t_th1 = (
                        dstt[i, j] * sn * cs +
                            ddtt_dth[i, j] * sn2 -
                            ddtp_dph[i, j] * sn
                    ) * inv_2dth
                    t_ph2 = dspp[i, j] * inv_dph2
                    t_ph1 = (ddtp_dth[i, j] * sn + ddpp_dph[i, j]) * inv_2dph
                    cA[k] = -FT(2) * (t_th2 + t_ph2)
                    cB[k] = t_th2 - t_th1
                    cC[k] = t_th2 + t_th1
                    cD[k] = t_ph2 - t_ph1
                    cE[k] = t_ph2 + t_ph1
                end
            end
            if op.is_neumann
                kp = op.k_pin
                @inbounds begin
                    cA[kp] = zero(FT)
                    cB[kp] = zero(FT)
                    cC[kp] = zero(FT)
                    cD[kp] = zero(FT)
                    cE[kp] = zero(FT)
                end
            end
            xp = (is_north ? x_N : x_S)::Vector{FT}
            _matvec_stencil_only!(ws.v, xp, op_dot)
            @inbounds @simd for k in eachindex(ws.rhs, ws.v)
                ws.rhs[k] -= ws.v[k]
            end
        end

        _solve_bicgstab!(op, ws, solver.config.tol, solver.config.max_iter, Val(false))
        _unpack_hemisphere_phi!(
            dphi_out, ws.x, op, i_min, is_north, solver.n_theta_hem, nt,
        )
    end
    return dphi_out
end

# Exact pullback of _compute_gradients_outofplace(solver, phi_pot)
@muladd function _compute_gradients_vjp(
        solver::IonosphereSolver{FT},
        bar_dth::Union{Nothing, AbstractMatrix{<:Real}},
        bar_dph::Union{Nothing, AbstractMatrix{<:Real}},
        bar_pg_N::Union{Nothing, AbstractVector{<:Real}},
        bar_pg_S::Union{Nothing, AbstractVector{<:Real}},
    ) where {FT <: Real}
    nt, np = solver.n_theta, solver.n_phi
    bar_phi = zeros(FT, nt, np)
    inv_2dth = FT(0.5) * solver.inv_dtheta
    inv_2dph = FT(0.5) * solver.inv_dphi
    scale_m1 = (FT(2) / solver.frame.r_iono) / FT(np)

    # Pull back pole_grad_N/S into row 1 and row nt of dphi_dtheta
    cos_phi = solver.cos_phi
    sin_phi = solver.sin_phi
    @inbounds for j in 1:np
        b_dth_1 = bar_dth === nothing ? zero(FT) : FT(bar_dth[1, j])
        b_dth_nt = bar_dth === nothing ? zero(FT) : FT(bar_dth[nt, j])
        if bar_pg_N !== nothing
            b_dth_1 += scale_m1 * (
                FT(bar_pg_N[1]) * cos_phi[j] + FT(bar_pg_N[2]) * sin_phi[j]
            )
        end
        if bar_pg_S !== nothing
            b_dth_nt -= scale_m1 * (
                FT(bar_pg_S[1]) * cos_phi[j] + FT(bar_pg_S[2]) * sin_phi[j]
            )
        end

        # One-sided 2nd-order differences at poles
        c1 = b_dth_1 * inv_2dth
        bar_phi[1, j] += -FT(3) * c1
        bar_phi[2, j] += FT(4) * c1
        bar_phi[3, j] += -c1

        cn = b_dth_nt * inv_2dth
        bar_phi[nt, j] += FT(3) * cn
        bar_phi[nt - 1, j] += -FT(4) * cn
        bar_phi[nt - 2, j] += cn

        if bar_dth !== nothing
            for i in 2:(nt - 1)
                c = FT(bar_dth[i, j]) * inv_2dth
                bar_phi[i + 1, j] += c
                bar_phi[i - 1, j] -= c
            end
        end
    end

    if bar_dph !== nothing
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            for i in 1:nt
                c = FT(bar_dph[i, j]) * inv_2dph
                bar_phi[i, jp] += c
                bar_phi[i, jm] -= c
            end
        end
    end
    return bar_phi
end
