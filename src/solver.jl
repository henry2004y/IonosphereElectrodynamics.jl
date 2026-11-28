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
        FT = nothing,
    )

Configuration for the 2D spherical magnetosphere–ionosphere (M–I) electrostatic
coupling solver, including optional geometry and boundary metadata for coupling to
3D magnetospheric host simulations. Element type `FT` is inferred from inputs and
`conductance` when omitted (defaulting to `Float64`).
"""
struct IonosphereConfig{FT <: Real, C <: AbstractConductanceModel}
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
    i0 = firstindex(v)
    return SVector{3, FT}(FT(v[i0]), FT(v[i0 + 1]), FT(v[i0 + 2]))
end

function IonosphereConfig(
        ;
        r_iono::Union{Nothing, Real} = nothing,
        r_fac::Union{Nothing, Real} = nothing,
        r_ib::Union{Nothing, Real} = nothing,
        r_core::Union{Nothing, Real} = nothing,
        center::Union{_Default, _Vec3Input} = _DEFAULT,
        dipole_moment::Union{Nothing, _Vec3Input} = nothing,
        n_theta_hem::Integer = 33,
        n_phi::Integer = 64,
        lat_boundary_deg::Union{Nothing, Real} = nothing,
        low_lat_bc::Symbol = :dirichlet,
        conductance::Union{_Default, AbstractConductanceModel} = _DEFAULT,
        use_dip_angle::Bool = true,
        sigma_0_ratio::Union{_Default, Real} = _DEFAULT,
        sigma_parallel::Union{Nothing, Real} = nothing,
        sigma_P_floor::Union{_Default, Real} = _DEFAULT,
        neutralize_fac::Bool = true,
        n_smooth_fac::Integer = 0,
        relax_alpha::Union{_Default, Real} = _DEFAULT,
        omega_corot::Union{_Default, Real, _Vec3Input} = _DEFAULT,
        couple_every::Integer = 1,
        u_max::Union{_Default, Real} = _DEFAULT,
        tol::Union{_Default, Real} = _DEFAULT,
        max_iter::Integer = 500,
        damp_radial_B1::Bool = true,
        particle_bc::Symbol = :ionosphere,
        n_ib::Union{_Default, Real} = _DEFAULT,
        T_ib::Union{_Default, Real} = _DEFAULT,
        update_per_stage::Bool = false,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    )
    cond_eltype = conductance isa _Default ? Union{} : _conductance_eltype(conductance)
    T = _infer_ft(
        FT, r_iono, r_fac, r_ib, r_core, center, dipole_moment, lat_boundary_deg,
        cond_eltype, sigma_0_ratio, sigma_parallel, sigma_P_floor, relax_alpha,
        omega_corot, u_max, tol, n_ib, T_ib,
    )
    sT = _scalar_float_type(T)

    r_io = r_iono === nothing ? nothing : T(r_iono)
    r_fa = r_fac === nothing ? nothing : T(r_fac)
    r_ib_f = r_ib === nothing ? nothing : T(r_ib)
    r_co = r_core === nothing ? nothing : T(r_core)
    r_io !== nothing && ((isfinite(r_io) && r_io > zero(T)) || throw(ArgumentError("r_iono must be finite and > 0, got $r_iono")))
    r_fa !== nothing && ((isfinite(r_fa) && r_fa > zero(T)) || throw(ArgumentError("r_fac must be finite and > 0, got $r_fac")))
    r_ib_f !== nothing && ((isfinite(r_ib_f) && r_ib_f > zero(T)) || throw(ArgumentError("r_ib must be finite and > 0, got $r_ib")))
    r_co !== nothing && ((isfinite(r_co) && r_co >= zero(T)) || throw(ArgumentError("r_core must be finite and >= 0, got $r_core")))
    if r_ib_f !== nothing && r_io !== nothing
        r_io <= r_ib_f || throw(ArgumentError("r_iono ($r_io) must be <= r_ib ($r_ib_f)"))
    end
    if r_ib_f !== nothing && r_fa !== nothing
        r_fa >= r_ib_f || throw(ArgumentError("r_fac ($r_fa) must be >= r_ib ($r_ib_f)"))
    end
    if r_ib_f !== nothing && r_co !== nothing
        r_co < r_ib_f || throw(ArgumentError("r_core ($r_co) must be < r_ib ($r_ib_f)"))
    end

    c_vec = center isa _Default ? zero(SVector{3, T}) : _coerce_vec3(center, "center", T)
    m_vec = if dipole_moment === nothing
        nothing
    else
        mv = _coerce_vec3(dipole_moment, "dipole_moment", T)
        norm(mv) > zero(T) || throw(ArgumentError("dipole_moment must be non-zero"))
        mv
    end

    nth = Int(n_theta_hem)
    np = Int(n_phi)
    nth >= 5 || throw(ArgumentError("n_theta_hem must be >= 5, got $nth"))
    (np >= 8 && iseven(np)) || throw(ArgumentError("n_phi must be an even integer >= 8, got $np"))

    lat_deg = if lat_boundary_deg === nothing
        nothing
    else
        v = T(lat_boundary_deg)
        (isfinite(v) && zero(T) <= v < T(85)) ||
            throw(ArgumentError("lat_boundary_deg must satisfy 0 <= lat_boundary_deg < 85, got $lat_boundary_deg"))
        v
    end
    low_lat_bc in _VALID_LOW_LAT_BC ||
        throw(ArgumentError("low_lat_bc must be :dirichlet or :neumann, got $(repr(low_lat_bc))"))

    cond = conductance isa _Default ? UniformConductance{T}(5, 0) : conductance

    s0_rat = T(_val_or(sigma_0_ratio, T(1000)))
    (isfinite(s0_rat) && s0_rat > zero(T)) ||
        throw(ArgumentError("sigma_0_ratio must be finite and > 0, got $s0_rat"))
    sp_par = sigma_parallel === nothing ? nothing : T(sigma_parallel)
    sp_par !== nothing && ((isfinite(sp_par) && sp_par > zero(T)) || throw(ArgumentError("sigma_parallel must be > 0, got $sigma_parallel")))
    sp_flr = T(_val_or(sigma_P_floor, zero(T)))
    (isfinite(sp_flr) && sp_flr >= zero(T)) ||
        throw(ArgumentError("sigma_P_floor must be finite and >= 0, got $sp_flr"))

    nsm = Int(n_smooth_fac)
    nsm >= 0 || throw(ArgumentError("n_smooth_fac must be >= 0, got $nsm"))
    ra = T(_val_or(relax_alpha, one(T)))
    (isfinite(ra) && zero(T) < ra <= one(T)) ||
        throw(ArgumentError("relax_alpha must satisfy 0 < relax_alpha <= 1, got $ra"))

    om_vec = if omega_corot isa _Default
        zero(SVector{3, T})
    elseif omega_corot isa Real
        SVector{3, T}(zero(T), zero(T), T(omega_corot))
    else
        _coerce_vec3(omega_corot, "omega_corot", T)
    end

    ce = Int(couple_every)
    ce >= 1 || throw(ArgumentError("couple_every must be >= 1, got $ce"))
    um = T(_val_or(u_max, T(1.5)))
    (isfinite(um) && um > zero(T)) || throw(ArgumentError("u_max must be finite and > 0, got $um"))
    tl = T(_val_or(tol, T(max(floatmin(sT), sT(1.0e-8)))))
    (isfinite(tl) && tl > zero(T)) || throw(ArgumentError("tol must be finite and > 0, got $tl"))
    mi = Int(max_iter)
    mi >= 1 || throw(ArgumentError("max_iter must be >= 1, got $mi"))
    particle_bc in _VALID_PARTICLE_BC ||
        throw(ArgumentError("particle_bc must be :ionosphere or :absorbing, got $(repr(particle_bc))"))
    nib = T(_val_or(n_ib, one(T)))
    (isfinite(nib) && nib > zero(T)) || throw(ArgumentError("n_ib must be finite and > 0, got $nib"))
    tib = T(_val_or(T_ib, T(sT(0.2))))
    (isfinite(tib) && tib > zero(T)) || throw(ArgumentError("T_ib must be finite and > 0, got $tib"))

    return IonosphereConfig{T, typeof(cond)}(
        r_io, r_fa, r_ib_f, r_co, c_vec, m_vec, nth, np, lat_deg, low_lat_bc,
        cond, use_dip_angle, s0_rat, sp_par, sp_flr,
        neutralize_fac, nsm, ra, om_vec, ce, um, tl, mi,
        damp_radial_B1, particle_bc, nib, tib, update_per_stage,
    )
end

# ------------------------------------------------------------------------------
# Five-Point Spherical Operator (Wavefront-Reordered) + ILU(0) + BiCGSTAB (0 B)
# ------------------------------------------------------------------------------
struct _FivePointHemisphereOp{FT <: Real}
    n_th::Int
    n_ph::Int
    n_diag::Int
    is_north::Bool
    is_neumann::Bool
    diag_ptr::Vector{Int}
    diag_i_lo::Vector{Int}
    wf_idx::Matrix{Int}
    pole_idx::Vector{Int}
    wrap_prev::Vector{Int}
    wrap_next::Vector{Int}
    k_pin::Int
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

function _FivePointHemisphereOp{FT}(n_th::Int, n_ph::Int, is_north::Bool, is_neumann::Bool) where {FT <: Real}
    n = n_th * n_ph
    n_diag = n_th + n_ph - 1
    diag_ptr = Vector{Int}(undef, n_diag + 1)
    diag_i_lo = Vector{Int}(undef, n_diag)
    wf_idx = Matrix{Int}(undef, n_th, n_ph)
    ptr = 1
    @inbounds for d in 1:n_diag
        diag_ptr[d] = ptr
        i_lo = max(1, d - n_ph + 1)
        i_hi = min(n_th, d)
        diag_i_lo[d] = i_lo
        for i in i_lo:i_hi
            j = d + 1 - i
            wf_idx[i, j] = ptr
            ptr += 1
        end
    end
    @inbounds diag_ptr[n_diag + 1] = ptr

    i_pole_nb = is_north ? 1 : n_th
    pole_idx = Vector{Int}(undef, n_ph)
    @inbounds for j in 1:n_ph
        pole_idx[j] = wf_idx[i_pole_nb, j]
    end
    wrap_prev = Vector{Int}(undef, n_th)
    wrap_next = Vector{Int}(undef, n_th)
    @inbounds for i in 1:n_th
        wrap_prev[i] = wf_idx[i, n_ph]
        wrap_next[i] = wf_idx[i, 1]
    end
    k_pin = @inbounds wf_idx[is_north ? n_th : 1, 1]

    return _FivePointHemisphereOp{FT}(
        n_th, n_ph, n_diag, is_north, is_neumann,
        diag_ptr, diag_i_lo, wf_idx, pole_idx, wrap_prev, wrap_next, k_pin,
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
    )
end

function _similar_op(::Type{T}, op::_FivePointHemisphereOp) where {T <: Real}
    n = op.n_th * op.n_ph
    return _FivePointHemisphereOp{T}(
        op.n_th, op.n_ph, op.n_diag, op.is_north, op.is_neumann,
        op.diag_ptr, op.diag_i_lo, op.wf_idx, op.pole_idx, op.wrap_prev, op.wrap_next,
        op.k_pin,
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), Vector{T}(undef, n),
    )
end

function _copy_op(op::_FivePointHemisphereOp{FT}) where {FT <: Real}
    return _FivePointHemisphereOp{FT}(
        op.n_th, op.n_ph, op.n_diag, op.is_north, op.is_neumann,
        op.diag_ptr, op.diag_i_lo, op.wf_idx, op.pole_idx, op.wrap_prev, op.wrap_next,
        op.k_pin,
        copy(op.cA), copy(op.cB), copy(op.cC), copy(op.cD), copy(op.cE),
        copy(op.dInv), copy(op.eL), copy(op.fU), copy(op.e1L), copy(op.f1U),
    )
end

struct _BiCGSTABWorkspace{FT <: Real}
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

function _BiCGSTABWorkspace{FT}(n::Int) where {FT <: Real}
    return _BiCGSTABWorkspace{FT}(
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
        zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n),
    )
end

@muladd function _build_ilu0!(op::_FivePointHemisphereOp{FT}) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    n_diag = op.n_diag
    n = n_th * n_ph
    diag_ptr = op.diag_ptr
    diag_i_lo = op.diag_i_lo
    dInv, eL, fU, e1L, f1U = op.dInv, op.eL, op.fU, op.e1L, op.f1U
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE

    @inbounds @simd for k in 1:n
        dInv[k] = cA[k]
        eL[k] = cB[k]
        fU[k] = cC[k]
        e1L[k] = cD[k]
        f1U[k] = cE[k]
    end
    @inbounds for d in 1:n_diag
        p_d = diag_ptr[d]
        L = diag_ptr[d + 1] - p_d
        i_lo = diag_i_lo[d]
        i_hi = i_lo + L - 1
        j_hi = d + 1 - i_lo
        j_lo = d + 1 - i_hi
        i_lo == 1 && (eL[p_d] = zero(FT))
        i_hi == n_th && (fU[p_d + L - 1] = zero(FT))
        j_lo == 1 && (e1L[p_d + L - 1] = zero(FT))
        j_hi == n_ph && (f1U[p_d] = zero(FT))
    end

    eps_piv = _tiny(FT)
    @inbounds begin
        piv1 = dInv[1]
        if abs(piv1) < eps_piv
            piv1 = piv1 >= zero(FT) ? eps_piv : -eps_piv
        end
        dInv[1] = one(FT) / piv1

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
                piv = dInv[p_d] - e1L[p_d] * f1U[base_im1 + 1] * dInv[base_im1 + 1]
                piv = ifelse(abs(piv) < eps_piv, ifelse(piv >= zero(FT), eps_piv, -eps_piv), piv)
                dInv[p_d] = one(FT) / piv
            end

            t_start = has_head ? 1 : 0
            t_end = has_tail ? (L - 2) : (L - 1)
            @simd ivdep for t in t_start:t_end
                k = p_d + t
                kp = base_im1 + t
                piv = dInv[k] - eL[k] * fU[kp] * dInv[kp] - e1L[k] * f1U[kp + 1] * dInv[kp + 1]
                piv = ifelse(abs(piv) < eps_piv, ifelse(piv >= zero(FT), eps_piv, -eps_piv), piv)
                dInv[k] = one(FT) / piv
            end

            if has_tail
                k = p_d + L - 1
                kp = base_im1 + L - 1
                piv = dInv[k] - eL[k] * fU[kp] * dInv[kp]
                piv = ifelse(abs(piv) < eps_piv, ifelse(piv >= zero(FT), eps_piv, -eps_piv), piv)
                dInv[k] = one(FT) / piv
            end
        end

        @simd for k in 1:n
            di = dInv[k]
            eL[k] *= di
            e1L[k] *= di
            fU[k] *= di
            f1U[k] *= di
        end
    end
    return nothing
end

@muladd function _apply_ilu0!(z::Vector{FT}, r::Vector{FT}, op::_FivePointHemisphereOp{FT}) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    n_diag = op.n_diag
    diag_ptr = op.diag_ptr
    diag_i_lo = op.diag_i_lo
    dInv, eL, fU, e1L, f1U = op.dInv, op.eL, op.fU, op.e1L, op.f1U

    @inbounds begin
        z[1] = r[1] * dInv[1]
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
                z[p_d] = r[p_d] * dInv[p_d] - e1L[p_d] * z[base_im1 + 1]
            end

            t_start = has_head ? 1 : 0
            t_end = has_tail ? (L - 2) : (L - 1)
            @simd ivdep for t in t_start:t_end
                k = p_d + t
                kp = base_im1 + t
                z[k] = r[k] * dInv[k] - eL[k] * z[kp] - e1L[k] * z[kp + 1]
            end

            if has_tail
                k = p_d + L - 1
                kp = base_im1 + L - 1
                z[k] = r[k] * dInv[k] - eL[k] * z[kp]
            end
        end

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
                z[p_d] -= fU[p_d] * z[base_jp1 + 1]
            end

            t_start = has_head ? 1 : 0
            t_end = has_tail ? (L - 2) : (L - 1)
            @simd ivdep for t in t_start:t_end
                k = p_d + t
                kn = base_jp1 + t
                z[k] = z[k] - f1U[k] * z[kn] - fU[k] * z[kn + 1]
            end

            if has_tail
                k = p_d + L - 1
                kn = base_jp1 + L - 1
                z[k] -= f1U[k] * z[kn]
            end
        end
    end
    return nothing
end

@inline @muladd function _matvec_boundary_point!(
        y::Vector{FT},
        x::Vector{FT},
        op::_FivePointHemisphereOp{FT},
        k::Int,
        i::Int,
        j::Int,
        pole_avg::FT,
        base_im1::Int,
        base_jp1::Int,
        t::Int,
    ) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    is_north = op.is_north
    is_neumann = op.is_neumann
    @inbounds begin
        x_k = x[k]
        x_im1 = if i > 1
            x[base_im1 + t]
        elseif is_north
            pole_avg
        else
            is_neumann ? x_k : zero(FT)
        end
        x_ip1 = if i < n_th
            x[base_jp1 + t + 1]
        elseif is_north
            is_neumann ? x_k : zero(FT)
        else
            pole_avg
        end
        x_jm1 = j > 1 ? x[base_im1 + t + 1] : x[op.wrap_prev[i]]
        x_jp1 = j < n_ph ? x[base_jp1 + t] : x[op.wrap_next[i]]
        y[k] = op.cA[k] * x_k + op.cB[k] * x_im1 + op.cC[k] * x_ip1 +
            op.cD[k] * x_jm1 + op.cE[k] * x_jp1
    end
    return nothing
end

@muladd function _matvec_stencil_only!(
        y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT},
    ) where {FT <: Real}
    n_ph = op.n_ph
    n_diag = op.n_diag
    diag_ptr = op.diag_ptr
    diag_i_lo = op.diag_i_lo
    pole_idx = op.pole_idx
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE

    pole_avg = zero(FT)
    @inbounds @simd for j in 1:n_ph
        pole_avg += x[pole_idx[j]]
    end
    pole_avg /= FT(n_ph)

    @inbounds for d in 1:n_diag
        p_d = diag_ptr[d]
        L = diag_ptr[d + 1] - p_d
        i_lo = diag_i_lo[d]
        i_hi = i_lo + L - 1
        j_hi = d + 1 - i_lo
        j_lo = d + 1 - i_hi
        base_im1 = d > 1 ? (diag_ptr[d - 1] + (i_lo - diag_i_lo[d - 1]) - 1) : 0
        base_jp1 = d < n_diag ? (diag_ptr[d + 1] + (i_lo - diag_i_lo[d + 1])) : 0

        _matvec_boundary_point!(y, x, op, p_d, i_lo, j_hi, pole_avg, base_im1, base_jp1, 0)
        if L >= 2
            @simd ivdep for t in 1:(L - 2)
                k = p_d + t
                kp = base_im1 + t
                kn = base_jp1 + t
                y[k] = cA[k] * x[k] + cB[k] * x[kp] + cC[k] * x[kn + 1] +
                    cD[k] * x[kp + 1] + cE[k] * x[kn]
            end
            _matvec_boundary_point!(
                y, x, op, p_d + L - 1, i_hi, j_lo, pole_avg, base_im1, base_jp1, L - 1,
            )
        end
    end
    return nothing
end

@muladd function _matvec!(y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT}) where {FT <: Real}
    _matvec_stencil_only!(y, x, op)
    if op.is_neumann && op.n_th > 0 && op.n_ph > 0
        @inbounds y[op.k_pin] = x[op.k_pin]
    end
    return nothing
end

@inline _apply_op!(y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT}, ::Val{false}) where {FT} =
    _matvec!(y, x, op)
@inline _apply_op!(y::Vector{FT}, x::Vector{FT}, op::_FivePointHemisphereOp{FT}, ::Val{true}) where {FT} =
    _matvec_transpose!(y, x, op)

@inline _apply_precond!(z::Vector{FT}, r::Vector{FT}, op::_FivePointHemisphereOp{FT}, ::Val{false}) where {FT} =
    _apply_ilu0!(z, r, op)
@inline _apply_precond!(z::Vector{FT}, r::Vector{FT}, op::_FivePointHemisphereOp{FT}, ::Val{true}) where {FT} =
    _apply_ilu0_transpose!(z, r, op)

@inline @muladd function _vec_dot(a::Vector{FT}, b::Vector{FT}) where {FT <: Real}
    s = zero(FT)
    @inbounds @simd for k in eachindex(a, b)
        s += a[k] * b[k]
    end
    return s
end

@inline @muladd function _vec_dot_pair(t::Vector{FT}, s::Vector{FT}) where {FT <: Real}
    tt = zero(FT)
    ts = zero(FT)
    @inbounds @simd for k in eachindex(t, s)
        tk = t[k]
        tt += tk * tk
        ts += tk * s[k]
    end
    return (tt, ts)
end

@muladd function _solve_bicgstab!(
        op::_FivePointHemisphereOp{FT},
        ws::_BiCGSTABWorkspace{FT},
        tol::FT,
        max_iter::Int,
        transpose_mode::Val{Transpose} = Val(false),
    ) where {FT <: Real, Transpose}
    rhs, x, r, r_hat, p, v, s, t, y, z =
        ws.rhs, ws.x, ws.r, ws.r_hat, ws.p, ws.v, ws.s, ws.t, ws.y, ws.z
    n = length(rhs)
    fmin = FT(_floatmin(FT))
    tiny_abs = _tiny(FT)
    eps_rho = _eps_rho(FT)
    eps_omega = _eps_omega(FT)

    b_norm2 = _vec_dot(rhs, rhs)
    if b_norm2 <= zero(FT)
        fill!(x, zero(FT))
        return true, 0, zero(FT)
    end
    b_norm = sqrt(b_norm2)
    if b_norm <= tiny_abs
        fill!(x, zero(FT))
        return true, 0, zero(FT)
    end
    tiny_b = max(fmin, tiny_abs * b_norm2)

    _apply_op!(v, x, op, transpose_mode)
    r_norm2 = zero(FT)
    @inbounds @simd for k in 1:n
        rk = rhs[k] - v[k]
        r[k] = rk
        r_hat[k] = rk
        p[k] = zero(FT)
        v[k] = zero(FT)
        r_norm2 += rk * rk
    end

    r_norm = r_norm2 > zero(FT) ? sqrt(r_norm2) : zero(FT)
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
        if abs(rho_new) <= eps_rho * r_hat_norm * r_norm
            @inbounds @simd for k in 1:n
                r_hat[k] = r[k]
            end
            r_hat_norm = r_norm
            rho_new = r_norm * r_norm
            if rho_new <= tiny_b
                break
            end
            fresh_shadow = true
        end

        if fresh_shadow
            @inbounds @simd for k in 1:n
                p[k] = r[k]
            end
            fresh_shadow = false
        else
            beta = (rho_new / rho_old) * (alpha / omega)
            @inbounds @simd for k in 1:n
                p[k] = r[k] + beta * (p[k] - omega * v[k])
            end
        end

        _apply_precond!(y, p, op, transpose_mode)
        _apply_op!(v, y, op, transpose_mode)
        r_hat_v = _vec_dot(r_hat, v)
        if abs(r_hat_v) <= max(fmin, tiny_abs * (r_hat_norm * r_norm))
            @inbounds @simd for k in 1:n
                r_hat[k] = r[k]
            end
            r_hat_norm = r_norm
            fresh_shadow = true
            continue
        end
        alpha = rho_new / r_hat_v

        s_norm2 = zero(FT)
        @inbounds @simd for k in 1:n
            sk = r[k] - alpha * v[k]
            s[k] = sk
            s_norm2 += sk * sk
        end
        s_norm = s_norm2 > zero(FT) ? sqrt(s_norm2) : zero(FT)
        if s_norm / b_norm <= tol
            @inbounds @simd for k in 1:n
                x[k] += alpha * y[k]
            end
            rel_res = s_norm / b_norm
            return true, iters, rel_res
        end

        _apply_precond!(z, s, op, transpose_mode)
        _apply_op!(t, z, op, transpose_mode)
        tt, ts = _vec_dot_pair(t, s)
        if tt <= tiny_b
            @inbounds @simd for k in 1:n
                x[k] += alpha * y[k]
            end
            rel_res = s_norm / b_norm
            return rel_res <= tol, iters, rel_res
        end
        omega = ts / tt

        r_norm2 = zero(FT)
        @inbounds @simd for k in 1:n
            x[k] += alpha * y[k] + omega * z[k]
            rk = s[k] - omega * t[k]
            r[k] = rk
            r_norm2 += rk * rk
        end
        r_norm = r_norm2 > zero(FT) ? sqrt(r_norm2) : zero(FT)
        rel_res = r_norm / b_norm
        rho_old = rho_new
        if rel_res <= tol
            return true, iters, rel_res
        end
        if abs(omega) <= eps_omega
            @inbounds @simd for k in 1:n
                r_hat[k] = r[k]
            end
            r_hat_norm = r_norm
            fresh_shadow = true
            omega = one(FT)
        end
    end

    return rel_res <= tol, iters, rel_res
end

function _solve_hemisphere_linear!(
        op::_FivePointHemisphereOp{FT},
        ws::_BiCGSTABWorkspace{FT},
        tol::FT,
        max_iter::Int;
        rebuild_ilu::Bool = true,
    ) where {FT <: Real}
    rebuild_ilu && _build_ilu0!(op)
    return _solve_bicgstab!(op, ws, tol, max_iter, Val(false))
end

# ------------------------------------------------------------------------------
# IonosphereSolver State & Public Solve API
# ------------------------------------------------------------------------------
"""
    IonosphereSolver{FT <: Real, C <: AbstractConductanceModel}

Pre-allocated runtime solver for the 2D spherical ionospheric Ohm's law
``\\nabla_h \\cdot (\\boldsymbol{\\Sigma} \\cdot \\nabla_h \\Phi) = J_r^{\\text{outward}}``
on both hemispheres using FLEKS's 5-point spherical finite-difference stencil and
ILU(0)-preconditioned BiCGSTAB (`0 B` steady-state heap allocations for `Float64`/`Float32`).
"""
mutable struct IonosphereSolver{FT <: Real, C <: AbstractConductanceModel}
    const config::IonosphereConfig{FT, C}
    const frame::DipoleFrame{FT}
    const n_theta_hem::Int
    const n_theta::Int
    const n_phi::Int
    const dtheta::FT
    const dphi::FT
    const inv_dtheta::FT
    const inv_dphi::FT
    const lat_boundary_rad::FT
    const i_min_N::Int
    const i_max_N::Int
    const i_min_S::Int
    const i_max_S::Int

    const theta::Vector{FT}
    const phi::Vector{FT}
    const sin_theta::Vector{FT}
    const cos_theta::Vector{FT}
    const sin_theta2::Vector{FT}
    const cos_theta2::Vector{FT}
    const dip_cs3::Vector{FT}
    const dip_cs4::Vector{FT}
    const sin_phi::Vector{FT}
    const cos_phi::Vector{FT}
    const r_hat_sim::Array{FT, 3} # (3, n_theta, n_phi)

    const jr_outward::Matrix{FT}
    const jr_work::Matrix{FT}
    const sigma_P::Matrix{FT}
    const sigma_H::Matrix{FT}
    const sigma_tt::Matrix{FT}
    const sigma_tp::Matrix{FT}
    const sigma_pp::Matrix{FT}
    const dsigma_tt_dth::Matrix{FT}
    const dsigma_tp_dth::Matrix{FT}
    const dsigma_tp_dph::Matrix{FT}
    const dsigma_pp_dph::Matrix{FT}

    const phi_solved::Matrix{FT}
    const phi_pot::Matrix{FT}
    const dphi_dtheta::Matrix{FT}
    const dphi_dphi::Matrix{FT}
    pole_grad_N::SVector{2, FT}
    pole_grad_S::SVector{2, FT}
    has_prev_solution::Bool
    op_assembled::Bool

    const op_N::_FivePointHemisphereOp{FT}
    const op_S::_FivePointHemisphereOp{FT}
    const ws_N::_BiCGSTABWorkspace{FT}
    const ws_S::_BiCGSTABWorkspace{FT}

    cpcp_north::FT
    cpcp_south::FT
    last_iters_north::Int
    last_iters_south::Int
    last_resid_north::FT
    last_resid_south::FT
end

@inline function _infer_solver_ft(::Type{FT_cfg}, ::Nothing, args...) where {FT_cfg <: Real}
    T_ovr = _promote_args(Union{}, args...)
    if T_ovr === Union{} || T_ovr <: Union{Integer, Float16, Float32, Float64}
        return FT_cfg
    end
    return float(promote_type(FT_cfg, T_ovr))
end
@inline _infer_solver_ft(::Type, ::Type{FT}, args...) where {FT <: Real} = FT

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
        dipole_moment::Union{Nothing, _Vec3Input} = nothing,
        center::Union{Nothing, _Vec3Input} = nothing,
        lat_boundary_deg::Union{Nothing, Real} = nothing,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    ) where {FT_cfg <: Real, C <: AbstractConductanceModel}
    T = _infer_solver_ft(FT_cfg, FT, r_iono, dipole_moment, center, lat_boundary_deg)
    sT = _scalar_float_type(T)
    pi_t = _pi(T)

    r_io = r_iono !== nothing ? T(r_iono) :
        (
            cfg.r_iono !== nothing ? T(cfg.r_iono) :
            (cfg.r_ib !== nothing ? T(0.5) * T(cfg.r_ib) : one(T))
        )
    mom = dipole_moment !== nothing ? dipole_moment :
        (cfg.dipole_moment !== nothing ? cfg.dipole_moment : SVector{3, T}(zero(T), zero(T), -one(T)))
    ctr = center !== nothing ? center : cfg.center

    frame = build_dipole_frame(mom; center = ctr, r_iono = r_io, FT = T)

    lat_deg_val = if lat_boundary_deg !== nothing
        T(lat_boundary_deg)
    elseif cfg.lat_boundary_deg !== nothing
        T(cfg.lat_boundary_deg)
    elseif cfg.r_ib !== nothing && r_io < T(cfg.r_ib)
        # Auto low-latitude boundary from dipole mapping sin(th_max) = sqrt(r_iono / r_ib)
        T(90) - rad2deg(asin(sqrt(r_io / T(cfg.r_ib))))
    else
        T(20)
    end
    lat_rad = clamp(deg2rad(lat_deg_val), zero(T), T(sT(0.49)) * pi_t)

    nth_hem = cfg.n_theta_hem
    nt = 2 * nth_hem - 1
    np = cfg.n_phi
    dth = pi_t / T(nt - 1)
    dph = (T(2) * pi_t) / T(np)
    inv_dth = inv(dth)
    inv_dph = inv(dph)

    theta = Vector{T}(undef, nt)
    sin_th = Vector{T}(undef, nt)
    cos_th = Vector{T}(undef, nt)
    sin_th2 = Vector{T}(undef, nt)
    cos_th2 = Vector{T}(undef, nt)
    dip_cs3 = Vector{T}(undef, nt)
    dip_cs4 = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        th = T(i - 1) * dth
        theta[i] = th
        s, c = sincos(th)
        sin_th[i] = s
        cos_th[i] = c
    end
    @inbounds begin
        sin_th[1] = zero(T)
        cos_th[1] = one(T)
        sin_th[nth_hem] = one(T)
        cos_th[nth_hem] = zero(T)
        sin_th[nt] = zero(T)
        cos_th[nt] = -one(T)
        for i in 1:nt
            sn2 = sin_th[i] * sin_th[i]
            cs2 = cos_th[i] * cos_th[i]
            cs3 = one(T) + T(3) * cs2
            sin_th2[i] = sn2
            cos_th2[i] = cs2
            dip_cs3[i] = cs3
            dip_cs4[i] = sqrt(cs3)
        end
    end

    phi = Vector{T}(undef, np)
    sin_ph = Vector{T}(undef, np)
    cos_ph = Vector{T}(undef, np)
    @inbounds for j in 1:np
        ph = T(j - 1) * dph
        phi[j] = ph
        sin_ph[j], cos_ph[j] = sincos(ph)
    end

    rot_T = transpose(frame.rot)
    r_hat_sim = Array{T, 3}(undef, 3, nt, np)
    @inbounds for j in 1:np, i in 1:nt
        r_smg = SVector{3, T}(sin_th[i] * cos_ph[j], sin_th[i] * sin_ph[j], cos_th[i])
        rs = rot_T * r_smg
        r_hat_sim[1, i, j] = rs[1]
        r_hat_sim[2, i, j] = rs[2]
        r_hat_sim[3, i, j] = rs[3]
    end

    i_eq = nth_hem
    eps_bnd = T(max(floatmin(sT), sT(1.0e-12)))
    half_pi = T(0.5) * pi_t
    i_min_N = 2
    i_max_N = 1
    @inbounds for i in 2:(i_eq - 1)
        if half_pi - theta[i] > lat_rad + eps_bnd
            i_max_N = i
        end
    end
    i_max_S = nt - 1
    i_min_S = nt
    @inbounds for i in (nt - 1):-1:(i_eq + 1)
        if theta[i] - half_pi > lat_rad + eps_bnd
            i_min_S = i
        end
    end
    n_th_N = max(0, i_max_N - i_min_N + 1)
    n_th_S = max(0, i_max_S - i_min_S + 1)
    n_th_N > 0 && n_th_S > 0 ||
        throw(ArgumentError("Ionosphere grid has no active polar-cap rows (n_theta_hem=$nth_hem, lat_boundary_deg=$lat_deg_val)"))

    is_neumann = cfg.low_lat_bc === :neumann
    op_N = _FivePointHemisphereOp{T}(n_th_N, np, true, is_neumann)
    op_S = _FivePointHemisphereOp{T}(n_th_S, np, false, is_neumann)
    ws_N = _BiCGSTABWorkspace{T}(n_th_N * np)
    ws_S = _BiCGSTABWorkspace{T}(n_th_S * np)

    cfg_ft = if T === FT_cfg
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
            FT = T,
        )
    end

    solver = IonosphereSolver{T, typeof(cfg_ft.conductance)}(
        cfg_ft, frame, nth_hem, nt, np, dth, dph, inv_dth, inv_dph, lat_rad,
        i_min_N, i_max_N, i_min_S, i_max_S,
        theta, phi, sin_th, cos_th, sin_th2, cos_th2, dip_cs3, dip_cs4,
        sin_ph, cos_ph, r_hat_sim,
        zeros(T, nt, np), zeros(T, nt, np),
        zeros(T, nt, np), zeros(T, nt, np),
        zeros(T, nt, np), zeros(T, nt, np), zeros(T, nt, np),
        zeros(T, nt, np), zeros(T, nt, np), zeros(T, nt, np), zeros(T, nt, np),
        zeros(T, nt, np), zeros(T, nt, np),
        zeros(T, nt, np), zeros(T, nt, np),
        zero(SVector{2, T}), zero(SVector{2, T}), false, false,
        op_N, op_S, ws_N, ws_S,
        zero(T), zero(T), 0, 0, zero(T), zero(T),
    )
    _compute_conductances_and_tensor!(solver, zero(T))
    if _is_static_conductance(cfg_ft.conductance)
        _assemble_hemisphere_op!(solver, true)
        _assemble_hemisphere_op!(solver, false)
        if T <: AbstractFloat
            _build_ilu0!(solver.op_N)
            _build_ilu0!(solver.op_S)
        end
        solver.op_assembled = true
    end
    return solver
end

@muladd function _smooth_jr!(jr::Matrix{FT}, tmp::Matrix{FT}, n_pass::Int) where {FT <: Real}
    n_pass <= 0 && return nothing
    nt, np = size(jr)
    inv_np = one(FT) / FT(np)

    for _ in 1:n_pass
        @inbounds for j in 1:np
            jm = j == 1 ? np : j - 1
            jp = j == np ? 1 : j + 1
            @simd for i in 2:(nt - 1)
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

@inline _smooth_jr!(solver::IonosphereSolver) =
    _smooth_jr!(solver.jr_outward, solver.jr_work, solver.config.n_smooth_fac)

@muladd function _compute_conductances_and_tensor!(solver::IonosphereSolver{FT}, t::FT) where {FT <: Real}
    nt = solver.n_theta
    np = solver.n_phi
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
    jr_outward = solver.jr_outward
    sigma_P = solver.sigma_P
    sigma_H = solver.sigma_H
    sigma_tt = solver.sigma_tt
    sigma_tp = solver.sigma_tp
    sigma_pp = solver.sigma_pp

    @inbounds for j in 1:np
        ph = phi[j]
        for i in 1:nt
            th = theta[i]
            rh = SVector{3, FT}(
                r_hat_sim[1, i, j],
                r_hat_sim[2, i, j],
                r_hat_sim[3, i, j],
            )
            sp, sh = evaluate_conductance(
                cond, th, ph, rh, t; jr_outward = jr_outward[i, j],
            )
            sp_f = max(FT(sp), sp_floor)
            sh_f = FT(sh)
            sigma_P[i, j] = sp_f
            sigma_H[i, j] = sh_f
            s0 = has_s0_par ? s0_par_val : (s0_ratio * sp_f)
            s_tt, s_tp, s_pp = _dipole_conductance_tensor_precomp(
                sp_f, sh_f, cos_th[i], sin_th2[i], cos_th2[i], dip_cs3[i], dip_cs4[i],
                dip_sign, s0, use_dip,
            )
            sigma_tt[i, j] = s_tt
            sigma_tp[i, j] = s_tp
            sigma_pp[i, j] = s_pp
        end
    end

    inv_2dth = FT(0.5) * solver.inv_dtheta
    inv_2dph = FT(0.5) * solver.inv_dphi
    dtt_dth = solver.dsigma_tt_dth
    dtp_dth = solver.dsigma_tp_dth
    dtp_dph = solver.dsigma_tp_dph
    dpp_dph = solver.dsigma_pp_dph
    @inbounds for j in 1:np
        jm = j == 1 ? np : j - 1
        jp = j == np ? 1 : j + 1
        dtt_dth[1, j] = zero(FT)
        dtp_dth[1, j] = zero(FT)
        dtt_dth[nt, j] = zero(FT)
        dtp_dth[nt, j] = zero(FT)
        @simd for i in 2:(nt - 1)
            dtt_dth[i, j] = (sigma_tt[i + 1, j] - sigma_tt[i - 1, j]) * inv_2dth
            dtp_dth[i, j] = (sigma_tp[i + 1, j] - sigma_tp[i - 1, j]) * inv_2dth
        end
        @simd for i in 1:nt
            dtp_dph[i, j] = (sigma_tp[i, jp] - sigma_tp[i, jm]) * inv_2dph
            dpp_dph[i, j] = (sigma_pp[i, jp] - sigma_pp[i, jm]) * inv_2dph
        end
    end
    return nothing
end

@muladd function _assemble_hemisphere_op!(
        solver::IonosphereSolver{FT}, is_north::Bool,
    ) where {FT <: Real}
    op = is_north ? solver.op_N : solver.op_S
    i_min = is_north ? solver.i_min_N : solver.i_min_S
    n_th = op.n_th
    n_ph = op.n_ph

    inv_dth2 = solver.inv_dtheta * solver.inv_dtheta
    inv_dph2 = solver.inv_dphi * solver.inv_dphi
    inv_2dth = FT(0.5) * solver.inv_dtheta
    inv_2dph = FT(0.5) * solver.inv_dphi

    wf_idx = op.wf_idx
    sin_th = solver.sin_theta
    cos_th = solver.cos_theta
    sin_th2 = solver.sin_theta2
    sigma_tt = solver.sigma_tt
    sigma_pp = solver.sigma_pp
    dtt_dth = solver.dsigma_tt_dth
    dtp_dth = solver.dsigma_tp_dth
    dtp_dph = solver.dsigma_tp_dph
    dpp_dph = solver.dsigma_pp_dph
    cA, cB, cC, cD, cE = op.cA, op.cB, op.cC, op.cD, op.cE

    @inbounds for j in 1:n_ph
        for i_loc in 1:n_th
            i = i_min + i_loc - 1
            k = wf_idx[i_loc, j]
            sn = sin_th[i]
            cs = cos_th[i]
            sn2 = sin_th2[i]

            stt = sigma_tt[i, j]
            spp = sigma_pp[i, j]
            term_th2 = stt * sn2 * inv_dth2
            term_th1 = (
                stt * sn * cs +
                    dtt_dth[i, j] * sn2 -
                    dtp_dph[i, j] * sn
            ) * inv_2dth
            term_ph2 = spp * inv_dph2
            term_ph1 = (dtp_dth[i, j] * sn + dpp_dph[i, j]) * inv_2dph

            cA[k] = -FT(2) * (term_th2 + term_ph2)
            cB[k] = term_th2 - term_th1
            cC[k] = term_th2 + term_th1
            cD[k] = term_ph2 - term_ph1
            cE[k] = term_ph2 + term_ph1
        end
    end

    if op.is_neumann
        k_pin = op.k_pin
        @inbounds begin
            cA[k_pin] = one(FT)
            cB[k_pin] = zero(FT)
            cC[k_pin] = zero(FT)
            cD[k_pin] = zero(FT)
            cE[k_pin] = zero(FT)
        end
    end
    return nothing
end

@muladd function _assemble_hemisphere_rhs!(
        rhs::Vector{FT},
        x::Vector{FT},
        solver::IonosphereSolver{FT},
        jr_mat::Matrix{FT},
        is_north::Bool,
        use_prev::Bool,
    ) where {FT <: Real}
    op = is_north ? solver.op_N : solver.op_S
    i_min = is_north ? solver.i_min_N : solver.i_min_S
    i_max = is_north ? solver.i_max_N : solver.i_max_S
    n_th = op.n_th
    n_ph = op.n_ph
    r_iono2 = solver.frame.r_iono * solver.frame.r_iono

    # Neutralize net hemisphere FAC over active rows when configured or in Neumann mode
    jr_shift = zero(FT)
    if solver.config.neutralize_fac || op.is_neumann
        num = zero(FT)
        den = zero(FT)
        sin_th = solver.sin_theta
        @inbounds for j in 1:n_ph
            @simd for i in i_min:i_max
                w = sin_th[i]
                num += jr_mat[i, j] * w
                den += w
            end
        end
        if den > zero(FT)
            jr_shift = num / den
        end
    end

    wf_idx = op.wf_idx
    sin_th2 = solver.sin_theta2
    phi_solved = solver.phi_solved
    @inbounds for j in 1:n_ph
        for i_loc in 1:n_th
            i = i_min + i_loc - 1
            k = wf_idx[i_loc, j]
            rhs[k] = (jr_mat[i, j] - jr_shift) * r_iono2 * sin_th2[i]
            x[k] = use_prev ? phi_solved[i, j] : zero(FT)
        end
    end
    if op.is_neumann
        @inbounds rhs[op.k_pin] = zero(FT)
    end
    return nothing
end

@muladd function _unpack_hemisphere_phi!(
        phi_out::Matrix{FT},
        x::Vector{FT},
        op::_FivePointHemisphereOp{FT},
        i_min::Int,
        is_north::Bool,
        i_eq::Int,
        nt::Int,
    ) where {FT <: Real}
    n_th = op.n_th
    n_ph = op.n_ph
    wf_idx = op.wf_idx

    i_clear_lo = is_north ? 1 : i_eq
    i_clear_hi = is_north ? i_eq : nt
    @inbounds for j in 1:n_ph
        @simd for i in i_clear_lo:i_clear_hi
            phi_out[i, j] = zero(FT)
        end
    end

    phi_min = @inbounds x[1]
    phi_max = @inbounds x[1]
    @inbounds for j in 1:n_ph
        for i_loc in 1:n_th
            i = i_min + i_loc - 1
            val = x[wf_idx[i_loc, j]]
            phi_out[i, j] = val
            phi_min = min(phi_min, val)
            phi_max = max(phi_max, val)
        end
    end

    i_pole = is_north ? 1 : nt
    i_adj = is_north ? 2 : nt - 1
    pole_val = zero(FT)
    @inbounds @simd for j in 1:n_ph
        pole_val += phi_out[i_adj, j]
    end
    pole_val /= FT(n_ph)
    @inbounds for j in 1:n_ph
        phi_out[i_pole, j] = pole_val
    end
    phi_min = min(phi_min, pole_val)
    phi_max = max(phi_max, pole_val)
    return phi_max - phi_min
end

@muladd function _solve_hemisphere!(
        solver::IonosphereSolver{FT},
        is_north::Bool,
        static_cached::Bool,
    ) where {FT <: Real}
    op = is_north ? solver.op_N : solver.op_S
    ws = is_north ? solver.ws_N : solver.ws_S
    i_min = is_north ? solver.i_min_N : solver.i_min_S

    if !static_cached
        _assemble_hemisphere_op!(solver, is_north)
    end
    _assemble_hemisphere_rhs!(
        ws.rhs, ws.x, solver, solver.jr_outward, is_north, solver.has_prev_solution,
    )

    rebuild_ilu = !(static_cached && FT <: AbstractFloat)
    ok, iters, resid = _solve_hemisphere_linear!(
        op, ws, solver.config.tol, solver.config.max_iter; rebuild_ilu = rebuild_ilu,
    )

    cpcp = _unpack_hemisphere_phi!(
        solver.phi_solved, ws.x, op, i_min, is_north, solver.n_theta_hem, solver.n_theta,
    )
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

@muladd function _compute_gradients_into!(
        dth::Matrix{T},
        dph::Matrix{T},
        phi_pot::AbstractMatrix{<:Real},
        inv_dtheta::T,
        inv_dphi::T,
        inv_r::T,
        cos_phi::Vector{<:Real},
        sin_phi::Vector{<:Real},
    ) where {T <: Real}
    nt, np = size(phi_pot)
    inv_2dth = T(0.5) * inv_dtheta
    inv_2dph = T(0.5) * inv_dphi

    @inbounds for j in 1:np
        @simd for i in 2:(nt - 1)
            dth[i, j] = (T(phi_pot[i + 1, j]) - T(phi_pot[i - 1, j])) * inv_2dth
        end
        dth[1, j] = (
            T(4) * T(phi_pot[2, j]) - T(3) * T(phi_pot[1, j]) - T(phi_pot[3, j])
        ) * inv_2dth
        dth[nt, j] = (
            T(3) * T(phi_pot[nt, j]) - T(4) * T(phi_pot[nt - 1, j]) + T(phi_pot[nt - 2, j])
        ) * inv_2dth
    end

    @inbounds for j in 1:np
        jm = j == 1 ? np : j - 1
        jp = j == np ? 1 : j + 1
        @simd for i in 1:nt
            dph[i, j] = (T(phi_pot[i, jp]) - T(phi_pot[i, jm])) * inv_2dph
        end
    end

    # m = 1 Fourier projection of pole dPhi/dtheta for smooth polar-axis E_conv
    scale_m1 = (T(2) * inv_r) / T(np)
    gx_n = zero(T)
    gy_n = zero(T)
    gx_s = zero(T)
    gy_s = zero(T)
    @inbounds for j in 1:np
        cp = T(cos_phi[j])
        sp = T(sin_phi[j])
        dn = dth[1, j]
        ds = -dth[nt, j]
        gx_n += dn * cp
        gy_n += dn * sp
        gx_s += ds * cp
        gy_s += ds * sp
    end
    return (
        SVector{2, T}(scale_m1 * gx_n, scale_m1 * gy_n),
        SVector{2, T}(scale_m1 * gx_s, scale_m1 * gy_s),
    )
end

@muladd function _compute_gradients!(solver::IonosphereSolver{FT}) where {FT <: Real}
    pg_N, pg_S = _compute_gradients_into!(
        solver.dphi_dtheta,
        solver.dphi_dphi,
        solver.phi_pot,
        solver.inv_dtheta,
        solver.inv_dphi,
        one(FT) / solver.frame.r_iono,
        solver.cos_phi,
        solver.sin_phi,
    )
    solver.pole_grad_N = pg_N
    solver.pole_grad_S = pg_S
    return nothing
end

function _compute_gradients_outofplace(
        solver::IonosphereSolver{FT}, phi_pot::AbstractMatrix{<:Real},
    ) where {FT <: Real}
    T = float(promote_type(FT, eltype(phi_pot)))
    nt, np = solver.n_theta, solver.n_phi
    dth = Matrix{T}(undef, nt, np)
    dph = Matrix{T}(undef, nt, np)
    pg_N, pg_S = _compute_gradients_into!(
        dth, dph, phi_pot,
        T(solver.inv_dtheta), T(solver.inv_dphi), one(T) / T(solver.frame.r_iono),
        solver.cos_phi, solver.sin_phi,
    )
    return (dth, dph, pg_N, pg_S)
end

"""
    solve_ionosphere!(
        solver::IonosphereSolver,
        jr_outward::AbstractMatrix{<:Real} = solver.jr_outward;
        t::Real = 0.0,
    ) -> Bool

Solve the 2D spherical ionospheric potential equation
``\\nabla_h \\cdot (\\boldsymbol{\\Sigma} \\cdot \\nabla_h \\Phi) = J_r^{\\text{outward}}``
on both hemispheres in `0 B` steady-state heap allocations (for `Float64`/`Float32`).
Returns `true` if both hemispheric ILU(0)-BiCGSTAB solves converged within
`solver.config.tol`.
"""
@muladd function solve_ionosphere!(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real} = solver.jr_outward;
        t::Real = zero(FT),
    ) where {FT <: Real}
    if jr_in !== solver.jr_outward
        size(jr_in) == size(solver.jr_outward) ||
            throw(DimensionMismatch("jr_outward size $(size(jr_in)) != expected $(size(solver.jr_outward))"))
        copyto!(solver.jr_outward, jr_in)
    end

    _smooth_jr!(solver)
    static_cached = _is_static_conductance(solver.config.conductance) && solver.op_assembled
    if !static_cached
        _compute_conductances_and_tensor!(solver, FT(t))
    end

    ok_N = _solve_hemisphere!(solver, true, static_cached)
    ok_S = _solve_hemisphere!(solver, false, static_cached)
    if _is_static_conductance(solver.config.conductance) && !solver.op_assembled
        solver.op_assembled = true
    end

    alpha = solver.config.relax_alpha
    if !solver.has_prev_solution || alpha >= one(FT)
        copyto!(solver.phi_pot, solver.phi_solved)
    else
        one_m_a = one(FT) - alpha
        phi_pot = solver.phi_pot
        phi_solved = solver.phi_solved
        @inbounds @simd for k in eachindex(phi_pot, phi_solved)
            phi_pot[k] = one_m_a * phi_pot[k] + alpha * phi_solved[k]
        end
    end
    solver.has_prev_solution = true
    _compute_gradients!(solver)
    return ok_N && ok_S
end

function _solve_ionosphere_stateless(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real},
        t::FT,
    ) where {FT <: Real}
    size(jr_in) == (solver.n_theta, solver.n_phi) ||
        throw(DimensionMismatch("jr_outward size $(size(jr_in)) != expected $((solver.n_theta, solver.n_phi))"))
    copyto!(solver.jr_outward, jr_in)
    _smooth_jr!(solver)

    static_cached = _is_static_conductance(solver.config.conductance) && solver.op_assembled
    if !static_cached
        _compute_conductances_and_tensor!(solver, t)
    end

    phi_out = zeros(FT, solver.n_theta, solver.n_phi)
    rebuild_ilu = !(static_cached && FT <: AbstractFloat)
    for is_north in (true, false)
        op = is_north ? solver.op_N : solver.op_S
        ws = is_north ? solver.ws_N : solver.ws_S
        i_min = is_north ? solver.i_min_N : solver.i_min_S
        if !static_cached
            _assemble_hemisphere_op!(solver, is_north)
        end
        _assemble_hemisphere_rhs!(ws.rhs, ws.x, solver, solver.jr_outward, is_north, false)
        _solve_hemisphere_linear!(
            op, ws, solver.config.tol, solver.config.max_iter; rebuild_ilu = rebuild_ilu,
        )
        _unpack_hemisphere_phi!(
            phi_out, ws.x, op, i_min, is_north, solver.n_theta_hem, solver.n_theta,
        )
    end
    if _is_static_conductance(solver.config.conductance) && !solver.op_assembled
        solver.op_assembled = true
    end
    return phi_out
end

"""
    solve_ionosphere(
        solver::IonosphereSolver,
        jr_outward::AbstractMatrix{<:Real};
        t::Real = 0.0,
    ) -> Matrix
    solve_ionosphere(
        cfg::IonosphereConfig,
        jr_outward::AbstractMatrix{<:Real};
        r_iono = nothing,
        dipole_moment = nothing,
        center = nothing,
        lat_boundary_deg = nothing,
        t::Real = 0.0,
    ) -> Matrix

Stateless, functional 2D spherical potential solve returning a fresh potential matrix
`phi_pot` of size `(n_theta, n_phi)`. Unlike [`solve_ionosphere!`](@ref), this always
solves cold-started without blending `relax_alpha` or modifying `solver.phi_pot`, making
it directly differentiable in forward-mode (`ForwardDiff.jl`) and reverse-mode
(`ChainRulesCore.jl` / `Zygote.jl` / `Mooncake.jl`) via implicit differentiation.
"""
function solve_ionosphere(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real};
        t::Real = zero(FT),
    ) where {FT <: Real}
    T = _infer_solver_ft(FT, nothing, jr_in, t)
    if T === FT
        return _solve_ionosphere_stateless(solver, jr_in, FT(t))
    end
    sol_T = IonosphereSolver(
        solver.config;
        r_iono = solver.frame.r_iono,
        dipole_moment = solver.frame.dipole_moment,
        center = solver.frame.center,
        lat_boundary_deg = rad2deg(solver.lat_boundary_rad),
        FT = T,
    )
    return _solve_ionosphere_stateless(sol_T, jr_in, T(t))
end

function solve_ionosphere(
        cfg::IonosphereConfig{FT_cfg},
        jr_in::AbstractMatrix{<:Real};
        r_iono::Union{Nothing, Real} = nothing,
        dipole_moment::Union{Nothing, _Vec3Input} = nothing,
        center::Union{Nothing, _Vec3Input} = nothing,
        lat_boundary_deg::Union{Nothing, Real} = nothing,
        t::Union{_Default, Real} = _DEFAULT,
    ) where {FT_cfg <: Real}
    T = _infer_solver_ft(
        FT_cfg, nothing, jr_in, r_iono, dipole_moment, center, lat_boundary_deg, t,
    )
    sol = IonosphereSolver(
        cfg;
        r_iono = r_iono,
        dipole_moment = dipole_moment,
        center = center,
        lat_boundary_deg = lat_boundary_deg,
        FT = T,
    )
    return _solve_ionosphere_stateless(sol, jr_in, T(_val_or(t, zero(T))))
end

"""
    reset_ionosphere!(solver::IonosphereSolver)

Reset solved ionospheric FAC, potentials, gradients, and solver diagnostics to zero.
"""
function reset_ionosphere!(solver::IonosphereSolver{FT}) where {FT <: Real}
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
function cross_polar_cap_potential(solver::IonosphereSolver{FT}) where {FT <: Real}
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
    ) where {FT <: Real}
    nt, np = solver.n_theta, solver.n_phi
    inv_r = one(FT) / solver.frame.r_iono
    sT = _scalar_float_type(FT)
    eps_sn = FT(max(floatmin(sT), sT(1.0e-8)))
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
                eph = sn > eps_sn ? (-solver.dphi_dphi[i, j] * inv_r / sn) : zero(FT)
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
