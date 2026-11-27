# ==============================================================================
# Unified Ionosphere Configuration & 2D Spherical ILU(0)-BiCGSTAB Solver
# ==============================================================================

const _VALID_LOW_LAT_BC = (:dirichlet, :neumann)
const _VALID_PARTICLE_BC = (:ionosphere, :absorbing)

"""
    IonosphereConfig(;
        r_iono = nothing,
        r_fac = nothing,
        r_ib = nothing,
        r_core = nothing,
        center = (0.0, 0.0, 0.0),
        dipole_moment = nothing,
        n_theta_hem = 33,
        n_phi = 64,
        lat_boundary_deg = nothing,
        low_lat_bc = :dirichlet,
        conductance = UniformConductance(5.0, 0.0),
        use_dip_angle = true,
        sigma_0_ratio = 1000.0,
        sigma_parallel = nothing,
        sigma_P_floor = 0.0,
        neutralize_fac = true,
        n_smooth_fac = 0,
        relax_alpha = 1.0,
        omega_corot = (0.0, 0.0, 0.0),
        couple_every = 1,
        u_max = 1.5,
        tol = 1.0e-8,
        max_iter = 500,
        damp_radial_B1 = true,
        particle_bc = :ionosphere,
        n_ib = 1.0,
        T_ib = 0.2,
        update_per_stage = false,
        FT = Float64,
    )

Unified configuration for 2D spherical magnetosphere–ionosphere (M–I) coupling across
`LingXiaoLBM`, `LingXiaoBoltz`, and `LingXiaoBoltzX`.
"""
struct IonosphereConfig{FT <: AbstractFloat, C <: AbstractConductanceModel}
    r_iono::Union{Nothing, FT}
    r_fac::Union{Nothing, FT}
    r_ib::Union{Nothing, FT}
    r_core::Union{Nothing, FT}
    center::SVector{3, FT}
    dipole_moment::Union{Nothing, SVector{3, FT}}
    n_theta_hem::Int
    n_phi::Int
    lat_boundary_deg::Union{Nothing, FT}
    low_lat_bc::Symbol
    conductance::C
    use_dip_angle::Bool
    sigma_0_ratio::FT
    sigma_parallel::Union{Nothing, FT}
    sigma_P_floor::FT
    neutralize_fac::Bool
    n_smooth_fac::Int
    relax_alpha::FT
    omega_corot::SVector{3, FT}
    couple_every::Int
    u_max::FT
    tol::FT
    max_iter::Int
    damp_radial_B1::Bool
    particle_bc::Symbol
    n_ib::FT
    T_ib::FT
    update_per_stage::Bool
end

function _coerce_vec3(v, name::AbstractString, ::Type{FT}) where {FT}
    length(v) == 3 || throw(ArgumentError("$name must have length 3, got $(length(v))"))
    return SVector{3, FT}(FT(v[1]), FT(v[2]), FT(v[3]))
end

function IonosphereConfig(
        ;
        r_iono::Union{Nothing, Real} = nothing,
        r_fac::Union{Nothing, Real} = nothing,
        r_ib::Union{Nothing, Real} = nothing,
        r_core::Union{Nothing, Real} = nothing,
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (0.0, 0.0, 0.0),
        dipole_moment::Union{Nothing, NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = nothing,
        n_theta_hem::Integer = 33,
        n_phi::Integer = 64,
        lat_boundary_deg::Union{Nothing, Real} = nothing,
        low_lat_bc::Symbol = :dirichlet,
        conductance::AbstractConductanceModel = UniformConductance(5.0, 0.0),
        use_dip_angle::Bool = true,
        sigma_0_ratio::Real = 1000.0,
        sigma_parallel::Union{Nothing, Real} = nothing,
        sigma_P_floor::Real = 0.0,
        neutralize_fac::Bool = true,
        n_smooth_fac::Integer = 0,
        relax_alpha::Real = 1.0,
        omega_corot::Union{Real, NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (0.0, 0.0, 0.0),
        couple_every::Integer = 1,
        u_max::Real = 1.5,
        tol::Real = 1.0e-8,
        max_iter::Integer = 500,
        damp_radial_B1::Bool = true,
        particle_bc::Symbol = :ionosphere,
        n_ib::Real = 1.0,
        T_ib::Real = 0.2,
        update_per_stage::Bool = false,
        FT::Type{<:AbstractFloat} = Float64,
    )
    r_io = r_iono === nothing ? nothing : FT(r_iono)
    r_fa = r_fac === nothing ? nothing : FT(r_fac)
    r_ib_f = r_ib === nothing ? nothing : FT(r_ib)
    r_co = r_core === nothing ? nothing : FT(r_core)
    r_io !== nothing && ((isfinite(r_io) && r_io > zero(FT)) || throw(ArgumentError("r_iono must be finite and > 0, got $r_iono")))
    r_fa !== nothing && ((isfinite(r_fa) && r_fa > zero(FT)) || throw(ArgumentError("r_fac must be finite and > 0, got $r_fac")))
    r_ib_f !== nothing && ((isfinite(r_ib_f) && r_ib_f > zero(FT)) || throw(ArgumentError("r_ib must be finite and > 0, got $r_ib")))
    r_co !== nothing && ((isfinite(r_co) && r_co >= zero(FT)) || throw(ArgumentError("r_core must be finite and >= 0, got $r_core")))
    if r_ib_f !== nothing && r_io !== nothing
        r_io <= r_ib_f || throw(ArgumentError("r_iono ($r_io) must be <= r_ib ($r_ib_f)"))
    end
    if r_ib_f !== nothing && r_fa !== nothing
        r_fa >= r_ib_f || throw(ArgumentError("r_fac ($r_fa) must be >= r_ib ($r_ib_f)"))
    end
    if r_ib_f !== nothing && r_co !== nothing
        r_co < r_ib_f || throw(ArgumentError("r_core ($r_co) must be < r_ib ($r_ib_f)"))
    end

    c_vec = _coerce_vec3(center, "center", FT)
    m_vec = if dipole_moment === nothing
        nothing
    else
        mv = _coerce_vec3(dipole_moment, "dipole_moment", FT)
        norm(mv) > zero(FT) || throw(ArgumentError("dipole_moment must be non-zero"))
        mv
    end

    nth = Int(n_theta_hem)
    np = Int(n_phi)
    nth >= 5 || throw(ArgumentError("n_theta_hem must be >= 5, got $nth"))
    (np >= 8 && iseven(np)) || throw(ArgumentError("n_phi must be an even integer >= 8, got $np"))

    lat_deg = if lat_boundary_deg === nothing
        nothing
    else
        v = FT(lat_boundary_deg)
        (isfinite(v) && zero(FT) <= v < FT(85)) ||
            throw(ArgumentError("lat_boundary_deg must satisfy 0 <= lat_boundary_deg < 85, got $lat_boundary_deg"))
        v
    end
    low_lat_bc in _VALID_LOW_LAT_BC ||
        throw(ArgumentError("low_lat_bc must be :dirichlet or :neumann, got $(repr(low_lat_bc))"))
    (isfinite(sigma_0_ratio) && sigma_0_ratio > 0) ||
        throw(ArgumentError("sigma_0_ratio must be finite and > 0, got $sigma_0_ratio"))
    sp_par = sigma_parallel === nothing ? nothing : FT(sigma_parallel)
    sp_par !== nothing && ((isfinite(sp_par) && sp_par > zero(FT)) || throw(ArgumentError("sigma_parallel must be > 0, got $sigma_parallel")))
    (isfinite(sigma_P_floor) && sigma_P_floor >= 0) ||
        throw(ArgumentError("sigma_P_floor must be finite and >= 0, got $sigma_P_floor"))

    nsm = Int(n_smooth_fac)
    nsm >= 0 || throw(ArgumentError("n_smooth_fac must be >= 0, got $nsm"))
    ra = FT(relax_alpha)
    (isfinite(ra) && zero(FT) < ra <= one(FT)) ||
        throw(ArgumentError("relax_alpha must satisfy 0 < relax_alpha <= 1, got $relax_alpha"))

    om_vec = omega_corot isa Real ?
        SVector{3, FT}(zero(FT), zero(FT), FT(omega_corot)) :
        _coerce_vec3(omega_corot, "omega_corot", FT)

    ce = Int(couple_every)
    ce >= 1 || throw(ArgumentError("couple_every must be >= 1, got $ce"))
    um = FT(u_max)
    (isfinite(um) && um > zero(FT)) || throw(ArgumentError("u_max must be finite and > 0, got $u_max"))
    tl = FT(tol)
    (isfinite(tl) && tl > zero(FT)) || throw(ArgumentError("tol must be finite and > 0, got $tol"))
    mi = Int(max_iter)
    mi >= 1 || throw(ArgumentError("max_iter must be >= 1, got $mi"))
    particle_bc in _VALID_PARTICLE_BC ||
        throw(ArgumentError("particle_bc must be :ionosphere or :absorbing, got $(repr(particle_bc))"))
    (isfinite(n_ib) && n_ib > 0) || throw(ArgumentError("n_ib must be finite and > 0, got $n_ib"))
    (isfinite(T_ib) && T_ib > 0) || throw(ArgumentError("T_ib must be finite and > 0, got $T_ib"))

    return IonosphereConfig{FT, typeof(conductance)}(
        r_io, r_fa, r_ib_f, r_co, c_vec, m_vec, nth, np, lat_deg, low_lat_bc,
        conductance, use_dip_angle, FT(sigma_0_ratio), sp_par, FT(sigma_P_floor),
        neutralize_fac, nsm, ra, om_vec, ce, um, tl, mi,
        damp_radial_B1, particle_bc, FT(n_ib), FT(T_ib), update_per_stage,
    )
end

# ------------------------------------------------------------------------------
# Five-Point Spherical Operator + ILU(0) + Right-Preconditioned BiCGSTAB (0 B)
# ------------------------------------------------------------------------------
struct _FivePointHemisphereOp{FT <: AbstractFloat}
    n_th::Int
    n_ph::Int
    is_north::Bool
    is_neumann::Bool
    cA::Vector{FT}
    cB::Vector{FT}
    cC::Vector{FT}
    cD::Vector{FT}
    cE::Vector{FT}
    dInv::Vector{FT}
    eL::Vector{FT}
    fU::Vector{FT}
    e1L::Vector{FT}
    f1U::Vector{FT}
end

function _FivePointHemisphereOp{FT}(n_th::Int, n_ph::Int, is_north::Bool, is_neumann::Bool) where {FT}
    n = n_th * n_ph
    return _FivePointHemisphereOp{FT}(
        n_th, n_ph, is_north, is_neumann,
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
    )
end

struct _BiCGSTABWorkspace{FT <: AbstractFloat}
    rhs::Vector{FT}
    x::Vector{FT}
    r::Vector{FT}
    r_hat::Vector{FT}
    p::Vector{FT}
    v::Vector{FT}
    s::Vector{FT}
    t::Vector{FT}
    y::Vector{FT}
    z::Vector{FT}
end

function _BiCGSTABWorkspace{FT}(n::Int) where {FT}
    return _BiCGSTABWorkspace{FT}(
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
    )
end

function _build_ilu0!(op::_FivePointHemisphereOp{FT}) where {FT}
    n_th = op.n_th
    n_ph = op.n_ph
    n = n_th * n_ph
    dInv, eL, fU, e1L, f1U = op.dInv, op.eL, op.fU, op.e1L, op.f1U
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE

    @inbounds for j in 1:n_ph
        off = (j - 1) * n_th
        for i in 1:n_th
            k = off + i
            dInv[k] = cA[k]
            eL[k] = i > 1 ? cB[k] : zero(FT)
            fU[k] = i < n_th ? cC[k] : zero(FT)
            e1L[k] = j > 1 ? cD[k] : zero(FT)
            f1U[k] = j < n_ph ? cE[k] : zero(FT)
        end
    end

    @inbounds for k in 1:n
        pivot = dInv[k]
        if k > 1
            pivot -= eL[k] * fU[k - 1] * dInv[k - 1]
        end
        if k > n_th
            pivot -= e1L[k] * f1U[k - n_th] * dInv[k - n_th]
        end
        if abs(pivot) < FT(1.0e-30)
            pivot = pivot >= zero(FT) ? FT(1.0e-30) : -FT(1.0e-30)
        end
        dInv[k] = one(FT) / pivot
    end
    return nothing
end

function _apply_ilu0!(z::Vector{FT}, r::Vector{FT}, op::_FivePointHemisphereOp{FT}) where {FT}
    n_th = op.n_th
    n = n_th * op.n_ph
    dInv, eL, fU, e1L, f1U = op.dInv, op.eL, op.fU, op.e1L, op.f1U

    @inbounds for k in 1:n
        val = r[k]
        if k > 1
            val -= eL[k] * z[k - 1]
        end
        if k > n_th
            val -= e1L[k] * z[k - n_th]
        end
        z[k] = val * dInv[k]
    end
    @inbounds for k in n:-1:1
        val = zero(FT)
        if k + 1 <= n
            val += fU[k] * z[k + 1]
        end
        if k + n_th <= n
            val += f1U[k] * z[k + n_th]
        end
        z[k] -= dInv[k] * val
    end
    return nothing
end

function _matvec!(y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT}) where {FT}
    n_th = op.n_th
    n_ph = op.n_ph
    is_north = op.is_north
    is_neumann = op.is_neumann
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE

    pole_avg = zero(FT)
    i_pole_nb = is_north ? 1 : n_th
    @inbounds for j in 1:n_ph
        pole_avg += x[(j - 1) * n_th + i_pole_nb]
    end
    pole_avg /= FT(n_ph)

    @inbounds for j in 1:n_ph
        j_prev = j == 1 ? n_ph : j - 1
        j_next = j == n_ph ? 1 : j + 1
        col_off = (j - 1) * n_th
        prev_off = (j_prev - 1) * n_th
        next_off = (j_next - 1) * n_th
        for i in 1:n_th
            k = col_off + i
            x_im1 = if i > 1
                x[col_off + i - 1]
            elseif is_north
                pole_avg
            else
                is_neumann ? x[k] : zero(FT)
            end
            x_ip1 = if i < n_th
                x[col_off + i + 1]
            elseif is_north
                is_neumann ? x[k] : zero(FT)
            else
                pole_avg
            end
            y[k] = cA[k] * x[k] + cB[k] * x_im1 + cC[k] * x_ip1 +
                cD[k] * x[prev_off + i] + cE[k] * x[next_off + i]
        end
    end

    if is_neumann && n_th > 0 && n_ph > 0
        k_pin = is_north ? n_th : 1
        @inbounds y[k_pin] = x[k_pin]
    end
    return nothing
end

@inline function _vec_dot(a::Vector{FT}, b::Vector{FT}) where {FT}
    s = zero(FT)
    @inbounds for k in eachindex(a, b)
        s += a[k] * b[k]
    end
    return s
end

function _solve_bicgstab!(
        op::_FivePointHemisphereOp{FT},
        ws::_BiCGSTABWorkspace{FT},
        tol::FT,
        max_iter::Int,
    ) where {FT}
    rhs, x, r, r_hat, p, v, s, t, y, z =
        ws.rhs, ws.x, ws.r, ws.r_hat, ws.p, ws.v, ws.s, ws.t, ws.y, ws.z
    n = length(rhs)

    b_norm = sqrt(_vec_dot(rhs, rhs))
    if b_norm <= FT(1.0e-30)
        fill!(x, zero(FT))
        return true, 0, zero(FT)
    end

    _matvec!(v, x, op)
    @inbounds for k in 1:n
        rk = rhs[k] - v[k]
        r[k] = rk
        r_hat[k] = rk
        p[k] = zero(FT)
        v[k] = zero(FT)
    end

    r_norm = sqrt(_vec_dot(r, r))
    rel_res = r_norm / b_norm
    if rel_res <= tol
        return true, 0, rel_res
    end

    rho_old = one(FT)
    alpha = one(FT)
    omega = one(FT)
    r_hat_norm = r_norm
    fresh_shadow = true
    iters = 0

    for it in 1:max_iter
        iters = it
        rho_new = _vec_dot(r_hat, r)
        if abs(rho_new) <= FT(1.0e-12) * r_hat_norm * r_norm
            @inbounds for k in 1:n
                rk = r[k]
                r_hat[k] = rk
                p[k] = rk
            end
            r_hat_norm = r_norm
            rho_new = r_norm * r_norm
            if rho_new <= FT(1.0e-30)
                break
            end
            fresh_shadow = true
        end

        if fresh_shadow
            @inbounds for k in 1:n
                p[k] = r[k]
            end
            fresh_shadow = false
        else
            beta = (rho_new / rho_old) * (alpha / omega)
            @inbounds for k in 1:n
                p[k] = r[k] + beta * (p[k] - omega * v[k])
            end
        end

        _apply_ilu0!(y, p, op)
        _matvec!(v, y, op)
        r_hat_v = _vec_dot(r_hat, v)
        if abs(r_hat_v) <= FT(1.0e-30)
            @inbounds for k in 1:n
                r_hat[k] = r[k]
            end
            r_hat_norm = r_norm
            fresh_shadow = true
            continue
        end
        alpha = rho_new / r_hat_v

        @inbounds for k in 1:n
            s[k] = r[k] - alpha * v[k]
        end
        s_norm = sqrt(_vec_dot(s, s))
        if s_norm / b_norm <= tol
            @inbounds for k in 1:n
                x[k] += alpha * y[k]
            end
            rel_res = s_norm / b_norm
            return true, iters, rel_res
        end

        _apply_ilu0!(z, s, op)
        _matvec!(t, z, op)
        tt = _vec_dot(t, t)
        if tt <= FT(1.0e-30)
            @inbounds for k in 1:n
                x[k] += alpha * y[k]
            end
            rel_res = s_norm / b_norm
            return rel_res <= tol, iters, rel_res
        end
        omega = _vec_dot(t, s) / tt

        @inbounds for k in 1:n
            x[k] += alpha * y[k] + omega * z[k]
            r[k] = s[k] - omega * t[k]
        end
        r_norm = sqrt(_vec_dot(r, r))
        rel_res = r_norm / b_norm
        rho_old = rho_new
        if rel_res <= tol
            return true, iters, rel_res
        end
        if abs(omega) <= FT(1.0e-14)
            @inbounds for k in 1:n
                r_hat[k] = r[k]
            end
            r_hat_norm = r_norm
            fresh_shadow = true
            omega = one(FT)
        end
    end

    return rel_res <= tol, iters, rel_res
end

# ------------------------------------------------------------------------------
# IonosphereSolver State & Public Solve API
# ------------------------------------------------------------------------------
"""
    IonosphereSolver{FT <: AbstractFloat, C <: AbstractConductanceModel}

Pre-allocated runtime solver for the 2D spherical ionospheric Ohm's law
``\\nabla_h \\cdot (\\boldsymbol{\\Sigma} \\cdot \\nabla_h \\Phi) = J_r^{\\text{outward}}``
on both hemispheres using FLEKS's 5-point spherical finite-difference stencil and
ILU(0)-preconditioned BiCGSTAB (`0 B` steady-state heap allocations).
"""
mutable struct IonosphereSolver{FT <: AbstractFloat, C <: AbstractConductanceModel}
    config::IonosphereConfig{FT, C}
    frame::DipoleFrame{FT}
    n_theta_hem::Int
    n_theta::Int
    n_phi::Int
    dtheta::FT
    dphi::FT
    lat_boundary_rad::FT
    i_min_N::Int
    i_max_N::Int
    i_min_S::Int
    i_max_S::Int

    theta::Vector{FT}
    phi::Vector{FT}
    sin_theta::Vector{FT}
    cos_theta::Vector{FT}
    sin_phi::Vector{FT}
    cos_phi::Vector{FT}
    r_hat_sim::Array{FT, 3} # (3, n_theta, n_phi)

    jr_outward::Matrix{FT}
    jr_work::Matrix{FT}
    sigma_P::Matrix{FT}
    sigma_H::Matrix{FT}
    sigma_tt::Matrix{FT}
    sigma_tp::Matrix{FT}
    sigma_pp::Matrix{FT}
    dsigma_tt_dth::Matrix{FT}
    dsigma_tp_dth::Matrix{FT}
    dsigma_tp_dph::Matrix{FT}
    dsigma_pp_dph::Matrix{FT}

    phi_solved::Matrix{FT}
    phi_pot::Matrix{FT}
    dphi_dtheta::Matrix{FT}
    dphi_dphi::Matrix{FT}
    pole_grad_N::SVector{2, FT}
    pole_grad_S::SVector{2, FT}
    has_prev_solution::Bool

    op_N::_FivePointHemisphereOp{FT}
    op_S::_FivePointHemisphereOp{FT}
    ws_N::_BiCGSTABWorkspace{FT}
    ws_S::_BiCGSTABWorkspace{FT}

    cpcp_north::FT
    cpcp_south::FT
    last_iters_north::Int
    last_iters_south::Int
    last_resid_north::FT
    last_resid_south::FT
end

"""
    IonosphereSolver(
        cfg::IonosphereConfig;
        r_iono = nothing,
        dipole_moment = nothing,
        center = nothing,
        lat_boundary_deg = nothing,
        FT = nothing,
    ) -> IonosphereSolver

Construct a pre-allocated [`IonosphereSolver`](@ref) from `cfg`, optionally overriding
`r_iono`, `dipole_moment`, `center`, or `lat_boundary_deg` from downstream simulation
geometry.
"""
function IonosphereSolver(
        cfg::IonosphereConfig{FT_cfg, C};
        r_iono::Union{Nothing, Real} = nothing,
        dipole_moment::Union{Nothing, NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = nothing,
        center::Union{Nothing, NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = nothing,
        lat_boundary_deg::Union{Nothing, Real} = nothing,
        FT::Type{<:AbstractFloat} = FT_cfg,
    ) where {FT_cfg <: AbstractFloat, C <: AbstractConductanceModel}
    r_io = r_iono !== nothing ? FT(r_iono) :
        (cfg.r_iono !== nothing ? FT(cfg.r_iono) :
        (cfg.r_ib !== nothing ? FT(0.5) * FT(cfg.r_ib) : one(FT)))
    mom = dipole_moment !== nothing ? dipole_moment :
        (cfg.dipole_moment !== nothing ? cfg.dipole_moment : SVector{3, FT}(zero(FT), zero(FT), -one(FT)))
    ctr = center !== nothing ? center : cfg.center

    frame = build_dipole_frame(mom; center = ctr, r_iono = r_io, FT = FT)

    lat_deg_val = if lat_boundary_deg !== nothing
        FT(lat_boundary_deg)
    elseif cfg.lat_boundary_deg !== nothing
        FT(cfg.lat_boundary_deg)
    elseif cfg.r_ib !== nothing && r_io < FT(cfg.r_ib)
        # Auto low-latitude boundary from dipole mapping sin(th_max) = sqrt(r_iono / r_ib)
        FT(90) - rad2deg(asin(sqrt(r_io / FT(cfg.r_ib))))
    else
        FT(20.0)
    end
    lat_rad = clamp(deg2rad(lat_deg_val), zero(FT), FT(0.49 * pi))

    nth_hem = cfg.n_theta_hem
    nt = 2 * nth_hem - 1
    np = cfg.n_phi
    dth = FT(pi) / FT(nt - 1)
    dph = FT(2 * pi) / FT(np)

    theta = Vector{FT}(undef, nt)
    sin_th = Vector{FT}(undef, nt)
    cos_th = Vector{FT}(undef, nt)
    @inbounds for i in 1:nt
        th = FT(i - 1) * dth
        theta[i] = th
        sin_th[i] = sin(th)
        cos_th[i] = cos(th)
    end
    sin_th[1] = zero(FT)
    cos_th[1] = one(FT)
    sin_th[nth_hem] = one(FT)
    cos_th[nth_hem] = zero(FT)
    sin_th[nt] = zero(FT)
    cos_th[nt] = -one(FT)

    phi = Vector{FT}(undef, np)
    sin_ph = Vector{FT}(undef, np)
    cos_ph = Vector{FT}(undef, np)
    @inbounds for j in 1:np
        ph = FT(j - 1) * dph
        phi[j] = ph
        sin_ph[j] = sin(ph)
        cos_ph[j] = cos(ph)
    end

    rot_T = transpose(frame.rot)
    r_hat_sim = Array{FT, 3}(undef, 3, nt, np)
    @inbounds for j in 1:np, i in 1:nt
        r_smg = SVector{3, FT}(sin_th[i] * cos_ph[j], sin_th[i] * sin_ph[j], cos_th[i])
        rs = rot_T * r_smg
        r_hat_sim[1, i, j] = rs[1]
        r_hat_sim[2, i, j] = rs[2]
        r_hat_sim[3, i, j] = rs[3]
    end

    i_eq = nth_hem
    eps_bnd = FT(1.0e-12)
    i_min_N = 2
    i_max_N = 1
    @inbounds for i in 2:(i_eq - 1)
        if FT(0.5 * pi) - theta[i] > lat_rad + eps_bnd
            i_max_N = i
        end
    end
    i_max_S = nt - 1
    i_min_S = nt
    @inbounds for i in (nt - 1):-1:(i_eq + 1)
        if theta[i] - FT(0.5 * pi) > lat_rad + eps_bnd
            i_min_S = i
        end
    end
    n_th_N = max(0, i_max_N - i_min_N + 1)
    n_th_S = max(0, i_max_S - i_min_S + 1)
    n_th_N > 0 && n_th_S > 0 ||
        throw(ArgumentError("Ionosphere grid has no active polar-cap rows (n_theta_hem=$nth_hem, lat_boundary_deg=$lat_deg_val)"))

    is_neumann = cfg.low_lat_bc === :neumann
    op_N = _FivePointHemisphereOp{FT}(n_th_N, np, true, is_neumann)
    op_S = _FivePointHemisphereOp{FT}(n_th_S, np, false, is_neumann)
    ws_N = _BiCGSTABWorkspace{FT}(n_th_N * np)
    ws_S = _BiCGSTABWorkspace{FT}(n_th_S * np)

    cfg_ft = if FT === FT_cfg
        cfg
    else
        IonosphereConfig(
            ;
            r_iono = r_io,
            r_fac = cfg.r_fac,
            r_ib = cfg.r_ib,
            r_core = cfg.r_core,
            center = Tuple(frame.center),
            dipole_moment = Tuple(frame.dipole_moment),
            n_theta_hem = nth_hem,
            n_phi = np,
            lat_boundary_deg = lat_deg_val,
            low_lat_bc = cfg.low_lat_bc,
            conductance = cfg.conductance,
            use_dip_angle = cfg.use_dip_angle,
            sigma_0_ratio = cfg.sigma_0_ratio,
            sigma_parallel = cfg.sigma_parallel,
            sigma_P_floor = cfg.sigma_P_floor,
            neutralize_fac = cfg.neutralize_fac,
            n_smooth_fac = cfg.n_smooth_fac,
            relax_alpha = cfg.relax_alpha,
            omega_corot = Tuple(cfg.omega_corot),
            couple_every = cfg.couple_every,
            u_max = cfg.u_max,
            tol = cfg.tol,
            max_iter = cfg.max_iter,
            damp_radial_B1 = cfg.damp_radial_B1,
            particle_bc = cfg.particle_bc,
            n_ib = cfg.n_ib,
            T_ib = cfg.T_ib,
            update_per_stage = cfg.update_per_stage,
            FT = FT,
        )
    end

    solver = IonosphereSolver{FT, typeof(cfg_ft.conductance)}(
        cfg_ft, frame, nth_hem, nt, np, dth, dph, lat_rad,
        i_min_N, i_max_N, i_min_S, i_max_S,
        theta, phi, sin_th, cos_th, sin_ph, cos_ph, r_hat_sim,
        zeros(FT, nt, np), zeros(FT, nt, np),
        zeros(FT, nt, np), zeros(FT, nt, np),
        zeros(FT, nt, np), zeros(FT, nt, np), zeros(FT, nt, np),
        zeros(FT, nt, np), zeros(FT, nt, np), zeros(FT, nt, np), zeros(FT, nt, np),
        zeros(FT, nt, np), zeros(FT, nt, np),
        zeros(FT, nt, np), zeros(FT, nt, np),
        zero(SVector{2, FT}), zero(SVector{2, FT}), false,
        op_N, op_S, ws_N, ws_S,
        zero(FT), zero(FT), 0, 0, zero(FT), zero(FT),
    )
    _compute_conductances_and_tensor!(solver, zero(FT))
    return solver
end

function _smooth_jr!(solver::IonosphereSolver{FT}) where {FT}
    n_pass = solver.config.n_smooth_fac
    n_pass <= 0 && return nothing
    nt = solver.n_theta
    np = solver.n_phi
    jr = solver.jr_outward
    tmp = solver.jr_work

    for _ in 1:n_pass
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            for i in 2:(nt - 1)
                tmp[i, j] = FT(0.5) * jr[i, j] + FT(0.125) * (
                    jr[i - 1, j] + jr[i + 1, j] + jr[i, jm] + jr[i, jp]
                )
            end
        end
        north_avg = zero(FT)
        south_avg = zero(FT)
        @inbounds for j in 1:np
            north_avg += tmp[2, j]
            south_avg += tmp[nt - 1, j]
        end
        inv_np = one(FT) / FT(np)
        north_avg *= inv_np
        south_avg *= inv_np
        @inbounds for j in 1:np
            tmp[1, j] = north_avg
            tmp[nt, j] = south_avg
        end
        copyto!(jr, tmp)
    end
    return nothing
end

function _compute_conductances_and_tensor!(solver::IonosphereSolver{FT}, t::FT) where {FT}
    nt = solver.n_theta
    np = solver.n_phi
    cfg = solver.config
    sp_floor = max(cfg.sigma_P_floor, FT(1.0e-12))
    dip_sign = solver.frame.dipole_sign
    use_dip = cfg.use_dip_angle
    s0_ratio = cfg.sigma_0_ratio
    s0_par = cfg.sigma_parallel

    @inbounds for j in 1:np
        ph = solver.phi[j]
        for i in 1:nt
            th = solver.theta[i]
            rh = SVector{3, FT}(
                solver.r_hat_sim[1, i, j],
                solver.r_hat_sim[2, i, j],
                solver.r_hat_sim[3, i, j],
            )
            sp, sh = evaluate_conductance(
                cfg.conductance, th, ph, rh, t; jr_outward = solver.jr_outward[i, j],
            )
            sp = max(FT(sp), sp_floor)
            sh_f = FT(sh)
            solver.sigma_P[i, j] = sp
            solver.sigma_H[i, j] = sh_f
            s_tt, s_tp, s_pp = dipole_conductance_tensor(
                sp, sh_f, solver.cos_theta[i], solver.sin_theta[i], dip_sign;
                sigma_0_ratio = s0_ratio, sigma_parallel = s0_par, use_dip_angle = use_dip,
            )
            solver.sigma_tt[i, j] = s_tt
            solver.sigma_tp[i, j] = s_tp
            solver.sigma_pp[i, j] = s_pp
        end
    end

    inv_2dth = FT(0.5) / solver.dtheta
    inv_2dph = FT(0.5) / solver.dphi
    @inbounds for j in 1:np
        jm = j == 1 ? np : j - 1
        jp = j == np ? 1 : j + 1
        for i in 1:nt
            if 1 < i < nt
                solver.dsigma_tt_dth[i, j] = (solver.sigma_tt[i + 1, j] - solver.sigma_tt[i - 1, j]) * inv_2dth
                solver.dsigma_tp_dth[i, j] = (solver.sigma_tp[i + 1, j] - solver.sigma_tp[i - 1, j]) * inv_2dth
            else
                solver.dsigma_tt_dth[i, j] = zero(FT)
                solver.dsigma_tp_dth[i, j] = zero(FT)
            end
            solver.dsigma_tp_dph[i, j] = (solver.sigma_tp[i, jp] - solver.sigma_tp[i, jm]) * inv_2dph
            solver.dsigma_pp_dph[i, j] = (solver.sigma_pp[i, jp] - solver.sigma_pp[i, jm]) * inv_2dph
        end
    end
    return nothing
end

function _solve_hemisphere!(
        solver::IonosphereSolver{FT},
        is_north::Bool,
    ) where {FT}
    op = is_north ? solver.op_N : solver.op_S
    ws = is_north ? solver.ws_N : solver.ws_S
    i_min = is_north ? solver.i_min_N : solver.i_min_S
    i_max = is_north ? solver.i_max_N : solver.i_max_S
    n_th = op.n_th
    n_ph = op.n_ph
    nt = solver.n_theta
    i_eq = solver.n_theta_hem

    dth2 = solver.dtheta * solver.dtheta
    dph2 = solver.dphi * solver.dphi
    two_dth = FT(2) * solver.dtheta
    two_dph = FT(2) * solver.dphi
    r_iono2 = solver.frame.r_iono * solver.frame.r_iono

    # Neutralize net hemisphere FAC over active rows when configured or in Neumann mode
    jr_shift = zero(FT)
    if solver.config.neutralize_fac || op.is_neumann
        num = zero(FT)
        den = zero(FT)
        @inbounds for j in 1:n_ph, i in i_min:i_max
            w = solver.sin_theta[i]
            num += solver.jr_outward[i, j] * w
            den += w
        end
        if den > zero(FT)
            jr_shift = num / den
        end
    end

    @inbounds for j in 1:n_ph
        off = (j - 1) * n_th
        for i_loc in 1:n_th
            i = i_min + i_loc - 1
            k = off + i_loc
            sn = solver.sin_theta[i]
            cs = solver.cos_theta[i]
            sn2 = sn * sn

            term_th2 = solver.sigma_tt[i, j] * sn2 / dth2
            term_th1 = (
                solver.sigma_tt[i, j] * sn * cs +
                solver.dsigma_tt_dth[i, j] * sn2 -
                solver.dsigma_tp_dph[i, j] * sn
            ) / two_dth
            term_ph2 = solver.sigma_pp[i, j] / dph2
            term_ph1 = (solver.dsigma_tp_dth[i, j] * sn + solver.dsigma_pp_dph[i, j]) / two_dph

            op.cA[k] = -FT(2) * (term_th2 + term_ph2)
            op.cB[k] = term_th2 - term_th1
            op.cC[k] = term_th2 + term_th1
            op.cD[k] = term_ph2 - term_ph1
            op.cE[k] = term_ph2 + term_ph1

            ws.rhs[k] = (solver.jr_outward[i, j] - jr_shift) * r_iono2 * sn2
            ws.x[k] = solver.has_prev_solution ? solver.phi_solved[i, j] : zero(FT)
        end
    end

    if op.is_neumann
        k_pin = is_north ? n_th : 1
        @inbounds begin
            ws.rhs[k_pin] = zero(FT)
            op.cA[k_pin] = one(FT)
            op.cB[k_pin] = zero(FT)
            op.cC[k_pin] = zero(FT)
            op.cD[k_pin] = zero(FT)
            op.cE[k_pin] = zero(FT)
        end
    end

    _build_ilu0!(op)
    ok, iters, resid = _solve_bicgstab!(op, ws, solver.config.tol, solver.config.max_iter)

    i_clear_lo = is_north ? 1 : i_eq
    i_clear_hi = is_north ? i_eq : nt
    @inbounds for j in 1:n_ph, i in i_clear_lo:i_clear_hi
        solver.phi_solved[i, j] = zero(FT)
    end

    phi_min = ws.x[1]
    phi_max = ws.x[1]
    @inbounds for j in 1:n_ph
        off = (j - 1) * n_th
        for i_loc in 1:n_th
            i = i_min + i_loc - 1
            val = ws.x[off + i_loc]
            solver.phi_solved[i, j] = val
            phi_min = min(phi_min, val)
            phi_max = max(phi_max, val)
        end
    end

    i_pole = is_north ? 1 : nt
    i_adj = is_north ? 2 : nt - 1
    pole_val = zero(FT)
    @inbounds for j in 1:n_ph
        pole_val += solver.phi_solved[i_adj, j]
    end
    pole_val /= FT(n_ph)
    @inbounds for j in 1:n_ph
        solver.phi_solved[i_pole, j] = pole_val
    end
    phi_min = min(phi_min, pole_val)
    phi_max = max(phi_max, pole_val)

    cpcp = phi_max - phi_min
    if is_north
        solver.cpcp_north = cpcp
        solver.last_iters_north = iters
        solver.last_resid_north = resid
    else
        solver.cpcp_south = cpcp
        solver.last_iters_south = iters
        solver.last_resid_south = resid
    end
    return ok
end

function _compute_gradients!(solver::IonosphereSolver{FT}) where {FT}
    nt = solver.n_theta
    np = solver.n_phi
    inv_2dth = FT(0.5) / solver.dtheta
    inv_2dph = FT(0.5) / solver.dphi
    phi_pot = solver.phi_pot
    dth = solver.dphi_dtheta
    dph = solver.dphi_dphi

    @inbounds for j in 1:np
        for i in 2:(nt - 1)
            dth[i, j] = (phi_pot[i + 1, j] - phi_pot[i - 1, j]) * inv_2dth
        end
        dth[1, j] = (FT(4) * phi_pot[2, j] - FT(3) * phi_pot[1, j] - phi_pot[3, j]) * inv_2dth
        dth[nt, j] = (FT(3) * phi_pot[nt, j] - FT(4) * phi_pot[nt - 1, j] + phi_pot[nt - 2, j]) * inv_2dth
    end

    @inbounds for j in 1:np
        jm = j == 1 ? np : j - 1
        jp = j == np ? 1 : j + 1
        for i in 1:nt
            dph[i, j] = (phi_pot[i, jp] - phi_pot[i, jm]) * inv_2dph
        end
    end

    # m = 1 Fourier projection of pole dPhi/dtheta for smooth polar-axis E_conv
    inv_r = one(FT) / solver.frame.r_iono
    scale_m1 = FT(2) * inv_r / FT(np)
    gx_n = zero(FT)
    gy_n = zero(FT)
    gx_s = zero(FT)
    gy_s = zero(FT)
    @inbounds for j in 1:np
        cp = solver.cos_phi[j]
        sp = solver.sin_phi[j]
        dn = dth[1, j]
        ds = -dth[nt, j]
        gx_n += dn * cp
        gy_n += dn * sp
        gx_s += ds * cp
        gy_s += ds * sp
    end
    solver.pole_grad_N = SVector{2, FT}(scale_m1 * gx_n, scale_m1 * gy_n)
    solver.pole_grad_S = SVector{2, FT}(scale_m1 * gx_s, scale_m1 * gy_s)
    return nothing
end

"""
    solve_ionosphere!(
        solver::IonosphereSolver{FT},
        jr_outward::AbstractMatrix{FT} = solver.jr_outward;
        t::Real = 0.0,
    ) -> Bool

Solve the 2D spherical ionospheric potential equation
``\\nabla_h \\cdot (\\boldsymbol{\\Sigma} \\cdot \\nabla_h \\Phi) = J_r^{\\text{outward}}``
on both hemispheres in `0 B` steady-state heap allocations. Returns `true` if both
hemispheric ILU(0)-BiCGSTAB solves converged within `solver.config.tol`.
"""
function solve_ionosphere!(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{FT} = solver.jr_outward;
        t::Real = zero(FT),
    ) where {FT <: AbstractFloat}
    if jr_in !== solver.jr_outward
        size(jr_in) == size(solver.jr_outward) ||
            throw(DimensionMismatch("jr_outward size $(size(jr_in)) != expected $(size(solver.jr_outward))"))
        copyto!(solver.jr_outward, jr_in)
    end

    _smooth_jr!(solver)
    _compute_conductances_and_tensor!(solver, FT(t))

    ok_N = _solve_hemisphere!(solver, true)
    ok_S = _solve_hemisphere!(solver, false)

    alpha = solver.config.relax_alpha
    if !solver.has_prev_solution || alpha >= one(FT)
        copyto!(solver.phi_pot, solver.phi_solved)
    else
        one_m_a = one(FT) - alpha
        @inbounds for k in eachindex(solver.phi_pot, solver.phi_solved)
            solver.phi_pot[k] = one_m_a * solver.phi_pot[k] + alpha * solver.phi_solved[k]
        end
    end
    solver.has_prev_solution = true
    _compute_gradients!(solver)
    return ok_N && ok_S
end

"""
    reset_ionosphere!(solver::IonosphereSolver)

Reset solved ionospheric FAC, potentials, gradients, and solver diagnostics to zero.
"""
function reset_ionosphere!(solver::IonosphereSolver{FT}) where {FT}
    fill!(solver.jr_outward, zero(FT))
    fill!(solver.jr_work, zero(FT))
    fill!(solver.phi_solved, zero(FT))
    fill!(solver.phi_pot, zero(FT))
    fill!(solver.dphi_dtheta, zero(FT))
    fill!(solver.dphi_dphi, zero(FT))
    solver.pole_grad_N = zero(SVector{2, FT})
    solver.pole_grad_S = zero(SVector{2, FT})
    solver.has_prev_solution = false
    solver.cpcp_north = zero(FT)
    solver.cpcp_south = zero(FT)
    solver.last_iters_north = 0
    solver.last_iters_south = 0
    solver.last_resid_north = zero(FT)
    solver.last_resid_south = zero(FT)
    return nothing
end

"""
    cross_polar_cap_potential(solver::IonosphereSolver) -> NamedTuple{(:north, :south)}

Return the northern and southern cross-polar-cap convection potential drops
``\\max(\\Phi) - \\min(\\Phi)`` across active nodes in each hemisphere.
"""
function cross_polar_cap_potential(solver::IonosphereSolver{FT}) where {FT}
    return (north = solver.cpcp_north, south = solver.cpcp_south)
end

"""
    write_ionosphere_diagnostics(filename::AbstractString, solver::IonosphereSolver; time = 0.0, cycle = 0)

Write 2D spherical ionosphere diagnostics (`theta_deg phi_deg Jr SigmaP SigmaH Phi E_theta E_phi`)
in a FLEKS-compatible ASCII table.
"""
function write_ionosphere_diagnostics(
        filename::AbstractString, solver::IonosphereSolver{FT};
        time::Real = 0.0, cycle::Integer = 0,
    ) where {FT}
    nt, np = solver.n_theta, solver.n_phi
    inv_r = one(FT) / solver.frame.r_iono
    open(filename, "w") do io
        println(io, "# Ionosphere Diagnostics")
        println(io, "# cycle = $cycle  time = $(Float64(time))  cpcpN = $(Float64(solver.cpcp_north))  cpcpS = $(Float64(solver.cpcp_south))")
        println(io, "# nTheta = $nt  nPhi = $np  rIono = $(Float64(solver.frame.r_iono))")
        println(io, "# theta_deg phi_deg Jr SigmaP SigmaH Phi Eth Eph")
        for i in 1:nt
            th_deg = rad2deg(solver.theta[i])
            sn = solver.sin_theta[i]
            for j in 1:np
                ph_deg = rad2deg(solver.phi[j])
                eth = -solver.dphi_dtheta[i, j] * inv_r
                eph = sn > FT(1.0e-8) ? (-solver.dphi_dphi[i, j] * inv_r / sn) : zero(FT)
                println(
                    io,
                    th_deg, " ", ph_deg, " ",
                    solver.jr_outward[i, j], " ",
                    solver.sigma_P[i, j], " ",
                    solver.sigma_H[i, j], " ",
                    solver.phi_pot[i, j], " ",
                    eth, " ", eph,
                )
            end
        end
    end
    return filename
end
