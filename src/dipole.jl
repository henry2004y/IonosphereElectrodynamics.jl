# ==============================================================================
# Dipole Orthonormal Frame, Analytic Field-Line Mapping, Jacobian & Tensor
# ==============================================================================

"""
    DipoleFrame{FT <: Real}

Immutable, `isbits`-compatible orthonormal Solar Magnetic (SMG) coordinate frame built
from a 3D planetary dipole moment vector.

- `center`: Body / dipole center in simulation coordinates.
- `rot`: Orthonormal rotation matrix whose rows `(e1, e2, e3)` map simulation vectors to
  SMG coordinates (`x_smg = rot * (x_sim - center)`), with `e3[3] >= 0` pointing toward
  geographic/simulation North.
- `dipole_moment`, `M0`: 3D dipole moment vector and magnitude `norm(dipole_moment)`.
- `dipole_sign`: `-1` for Earth-like polarity (`m ⋅ e3 < 0`), `+1` for reversed polarity.
- `r_iono`: Ionospheric shell radius in simulation length units.
"""
struct DipoleFrame{FT <: Real}
    center::SVector{3, FT}
    rot::SMatrix{3, 3, FT, 9}
    dipole_moment::SVector{3, FT}
    M0::FT
    dipole_sign::FT
    r_iono::FT
end

@inline _convert_frame(::Type{T}, frame::DipoleFrame{T}) where {T <: Real} = frame
@inline function _convert_frame(::Type{T}, frame::DipoleFrame) where {T <: Real}
    return DipoleFrame{T}(
        SVector{3, T}(frame.center),
        SMatrix{3, 3, T, 9}(frame.rot),
        SVector{3, T}(frame.dipole_moment),
        T(frame.M0),
        T(frame.dipole_sign),
        T(frame.r_iono),
    )
end

"""
    build_dipole_frame(
        dipole_moment;
        center = (0.0, 0.0, 0.0),
        r_iono = 1.0,
        FT = nothing,
    ) -> DipoleFrame

Construct a [`DipoleFrame`](@ref) from `dipole_moment`, matching FLEKS's SMG convention:
`e3` always points toward northern simulation hemisphere (`e3[3] >= 0`), `dipole_sign`
records the polarity along `e3` (`-1` Earth-like, `+1` reversed), and `e1` is the
projection of simulation `+x` (falling back to `+y` when `|e3[1]| > 0.99`) orthogonal
to `e3`. Element type `FT` is inferred from inputs when omitted.
"""
function build_dipole_frame(
        dipole_moment::_Vec3Input;
        center::Union{_Default, _Vec3Input} = _DEFAULT,
        r_iono::Union{_Default, Real} = _DEFAULT,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    )
    T = _infer_ft(FT, dipole_moment, center, r_iono)
    sT = _scalar_float_type(T)
    ctr = _val_or(center, (zero(T), zero(T), zero(T)))
    r_io = _val_or(r_iono, one(T))

    length(dipole_moment) == 3 || throw(ArgumentError("dipole_moment must have length 3"))
    length(ctr) == 3 || throw(ArgumentError("center must have length 3"))
    (isfinite(r_io) && r_io > 0) ||
        throw(ArgumentError("r_iono must be finite and > 0 (got $r_io)"))

    im0 = firstindex(dipole_moment)
    ic0 = firstindex(ctr)
    m = SVector{3, T}(
        T(dipole_moment[im0]), T(dipole_moment[im0 + 1]), T(dipole_moment[im0 + 2]),
    )
    M0 = norm(m)
    M0 > zero(T) || throw(ArgumentError("dipole_moment must be non-zero"))
    c = SVector{3, T}(T(ctr[ic0]), T(ctr[ic0 + 1]), T(ctr[ic0 + 2]))

    ax = m / M0
    dipole_sign = ax[3] < zero(T) ? -one(T) : one(T)
    ez = dipole_sign * ax

    ref = abs(ez[1]) > T(sT(0.99)) ?
        SVector{3, T}(zero(T), one(T), zero(T)) :
        SVector{3, T}(one(T), zero(T), zero(T))
    ex_raw = ref - dot(ref, ez) * ez
    ex = ex_raw / norm(ex_raw)
    ey = cross(ez, ex)

    # Column-major storage in SMatrix with rows = (ex, ey, ez)
    rot = SMatrix{3, 3, T, 9}(
        ex[1], ey[1], ez[1],
        ex[2], ey[2], ez[2],
        ex[3], ey[3], ez[3],
    )
    return DipoleFrame{T}(c, rot, m, M0, dipole_sign, T(r_io))
end

"""
    dipole_B0(frame::DipoleFrame, x_sim; r_min = eps(...)) -> SVector{3}

Evaluate the analytic dipole magnetic field ``\\mathbf{B}_0(\\mathbf{x})`` at simulation
position `x_sim` (`SVector{3}`, `AbstractVector`, or `NTuple{3}`) using `frame.center`
and `frame.dipole_moment` (softened at `r_min`).
"""
@inline @muladd function dipole_B0(
        frame::DipoleFrame{FT1},
        x_sim::_Vec3Input;
        r_min::Union{_Default, Real} = _DEFAULT,
    ) where {FT1 <: Real}
    xs = _as_svec3(x_sim)
    T0 = float(promote_type(FT1, eltype(xs)))
    r_min_raw = _val_or(r_min, _eps(T0))
    T = float(promote_type(T0, typeof(r_min_raw)))
    dr = SVector{3, T}(
        xs[1] - frame.center[1],
        xs[2] - frame.center[2],
        xs[3] - frame.center[3],
    )
    r2 = dr[1] * dr[1] + dr[2] * dr[2] + dr[3] * dr[3]
    r_min_t = T(r_min_raw)
    r = r2 > r_min_t * r_min_t ? sqrt(r2) : r_min_t
    inv_r = inv(r)
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r2 * inv_r
    inv_r5 = inv_r3 * inv_r2
    m = SVector{3, T}(frame.dipole_moment)
    mdr = m[1] * dr[1] + m[2] * dr[2] + m[3] * dr[3]
    return (T(3) * mdr * inv_r5) * dr - inv_r3 * m
end

@inline @muladd function _map_dipole_smg_core(
        x_smg::SVector{3, FT1}, r_map::FT2, ::Val{ComputeJac},
    ) where {FT1 <: Real, FT2 <: Real, ComputeJac}
    T = float(promote_type(FT1, FT2))
    sT = _scalar_float_type(T)
    x, y, z = T(x_smg[1]), T(x_smg[2]), T(x_smg[3])
    r_m = T(r_map)
    rho2 = x * x + y * y
    r2 = rho2 + z * z
    z3 = zero(SVector{3, T})
    z_jac = zero(SMatrix{2, 3, T, 6})
    eps_r = T(max(floatmin(sT), sT(1.0e-12)))
    eps_r2 = T(max(floatmin(sT), sT(1.0e-24)))
    if r2 <= eps_r2 || r_m <= eps_r
        return (0, z3, zero(T), zero(T), z_jac)
    end

    r = sqrt(r2)
    i_hem = z >= zero(T) ? 1 : -1
    ratio = r_m / r
    xy_ratio = ratio * sqrt(ratio) # (r_map / r)^(3/2)

    xm = xy_ratio * x
    ym = xy_ratio * y
    rho_map2 = xm * xm + ym * ym
    r_map2 = r_m * r_m
    pi_t = _pi(T)
    two_pi = T(2) * pi_t

    if rho_map2 > r_map2
        # Field line closes below r_map (L < r_map).
        if rho2 > zero(T)
            inv_rho_map = r_m / sqrt(rho2)
            xm = inv_rho_map * x
            ym = inv_rho_map * y
        else
            xm = r_m
            ym = zero(T)
        end
        x_map = SVector{3, T}(xm, ym, zero(T))
        a = atan(ym, xm)
        phi_map = (a < zero(T) ? a + two_pi : a) + zero(T)
        return (0, x_map, T(0.5) * pi_t, phi_map, z_jac)
    end

    diff = r_map2 - rho_map2
    zm = diff > zero(T) ? T(i_hem) * sqrt(diff) : zero(T)
    x_map = SVector{3, T}(xm, ym, zm)
    cos_th = clamp(zm / r_m, -one(T), one(T))
    theta_map = acos(cos_th)
    a = atan(ym, xm)
    phi_map = (a < zero(T) ? a + two_pi : a) + zero(T)

    jac = z_jac
    if ComputeJac
        eps_zm = T(max(floatmin(sT), sT(1.0e-6)))
        eps_rho2 = T(max(floatmin(sT), sT(1.0e-16)))
        if abs(zm) >= eps_zm * r_m && rho2 >= eps_rho2 * r2
            rho = sqrt(rho2)
            rho_map = xy_ratio * rho
            inv_r2 = inv(r2)
            inv_rho2 = inv(rho2)
            inv_zm = inv(zm)
            z_over_r = z / r
            factor = -(T(0.5) - T(1.5) * z_over_r * z_over_r) * inv_zm / rho
            j00 = xm * factor
            j01 = ym * factor
            j02 = -T(1.5) * (rho_map * inv_zm) * (z * inv_r2)
            j10 = -y * inv_rho2
            j11 = x * inv_rho2
            j12 = zero(T)
            jac = SMatrix{2, 3, T, 6}(j00, j10, j01, j11, j02, j12)
        end
    end

    return (i_hem, x_map, theta_map, phi_map, jac)
end

"""
    map_dipole_smg(x_smg, r_map::Real) -> (i_hem, x_map, theta_map, phi_map)

Map SMG position `x_smg` (`SVector{3}`, `AbstractVector`, or `NTuple{3}`) along the
dipole field line ``r = L\\sin^2\\theta`` to radius `r_map`. Returns
`(i_hem, x_map, theta_map, phi_map)` where `i_hem` is `+1` (north), `-1` (south), or
`0` (closed equatorial field line with ``L < r_{\\text{map}}``).
"""
@inline function map_dipole_smg(x_smg::_Vec3Input, r_map::Real)
    i_hem, x_map, theta_map, phi_map, _ = _map_dipole_smg_core(
        _as_svec3(x_smg), r_map, Val(false),
    )
    return (i_hem, x_map, theta_map, phi_map)
end

"""
    map_dipole_smg_with_jacobian(
        x_smg, r_map::Real
    ) -> (i_hem, x_map, theta_map, phi_map, jac_smg)

Map SMG position `x_smg` (`SVector{3}`, `AbstractVector`, or `NTuple{3}`) along the
dipole field line ``r = L\\sin^2\\theta`` to radius `r_map`, returning
`(i_hem, x_map, theta_map, phi_map, jac_smg)` where `jac_smg::SMatrix{2, 3, T, 6}` is
the exact closed-form Jacobian ``\\partial(\\theta_m, \\phi_m) / \\partial(x, y, z)`` in
SMG coordinates.
"""
@inline function map_dipole_smg_with_jacobian(x_smg::_Vec3Input, r_map::Real)
    return _map_dipole_smg_core(_as_svec3(x_smg), r_map, Val(true))
end

@inline @muladd function _dipole_conductance_tensor_precomp(
        sigma_P::T,
        sigma_H::T,
        cos_theta::T,
        sn2::T,
        cs2::T,
        cs3::T,
        cs4::T,
        dipole_sign::T,
        s0::T,
        use_dip_angle::Bool,
    ) where {T <: Real}
    hall_sign = -dipole_sign
    if !use_dip_angle
        sT = _scalar_float_type(T)
        eps_eq = T(max(floatmin(sT), sT(1.0e-12)))
        hemi = abs(cos_theta) <= eps_eq ? zero(T) :
            (cos_theta > zero(T) ? one(T) : -one(T))
        return (sigma_P, hall_sign * hemi * sigma_H, sigma_P)
    end
    C = T(4) * s0 * cs2 + sigma_P * sn2
    inv_C = inv(C)
    s_tt = (s0 * sigma_P * cs3) * inv_C
    s_tp = hall_sign * (T(2) * s0 * sigma_H * cos_theta * cs4) * inv_C
    s_pp = sigma_P + (sigma_H * sigma_H * sn2) * inv_C
    return (s_tt, s_tp, s_pp)
end

@inline @muladd function _dipole_conductance_tensor_core(
        sigma_P::T,
        sigma_H::T,
        cos_theta::T,
        sin_theta::T,
        dipole_sign::T,
        s0::T,
        use_dip_angle::Bool,
    ) where {T <: Real}
    sn2 = sin_theta * sin_theta
    cs2 = cos_theta * cos_theta
    cs3 = one(T) + T(3) * cs2
    cs4 = sqrt(cs3)
    return _dipole_conductance_tensor_precomp(
        sigma_P, sigma_H, cos_theta, sn2, cs2, cs3, cs4, dipole_sign, s0, use_dip_angle,
    )
end

"""
    dipole_conductance_tensor(
        sigma_P::Real,
        sigma_H::Real,
        cos_theta::Real,
        sin_theta::Real,
        dipole_sign::Real;
        sigma_0_ratio = 1000,
        sigma_parallel = nothing,
        use_dip_angle = true,
    ) -> (sigma_tt, sigma_tp, sigma_pp)

Compute the height-integrated spherical conductance tensor components
``(\\Sigma_{\\theta\\theta}, \\Sigma_{\\theta\\phi}, \\Sigma_{\\phi\\phi})`` at dipole
colatitude ``\\theta`` matching Goodman (1995) / Ridley et al. (2004) / FLEKS, including
finite parallel conductance ``\\Sigma_0``, equatorial Cowling enhancement in
``\\Sigma_{\\phi\\phi}``, and exact dipole-polarity Hall sign `(-dipole_sign)`.
"""
@inline function dipole_conductance_tensor(
        sigma_P::Real,
        sigma_H::Real,
        cos_theta::Real,
        sin_theta::Real,
        dipole_sign::Real;
        sigma_0_ratio::Union{_Default, Real} = _DEFAULT,
        sigma_parallel::Union{Nothing, Real} = nothing,
        use_dip_angle::Bool = true,
    )
    T = _infer_ft(
        nothing, sigma_P, sigma_H, cos_theta, sin_theta, dipole_sign,
        sigma_0_ratio, sigma_parallel,
    )
    sp = T(sigma_P)
    sh = T(sigma_H)
    ct = T(cos_theta)
    st = T(sin_theta)
    ds = T(dipole_sign)
    s0 = if sigma_parallel === nothing
        T(_val_or(sigma_0_ratio, T(1000))) * sp
    else
        max(T(sigma_parallel), T(10))
    end
    return _dipole_conductance_tensor_core(sp, sh, ct, st, ds, s0, use_dip_angle)
end

"""
    dipole_fac_sample_point(
        frame::DipoleFrame,
        theta_i::Real,
        phi::Real,
        r_fac::Real,
    ) -> (valid, x_fac_sim, x_iono_sim, r_hat_iono, b_iono_mag, br_over_b_iono)

Map an ionospheric shell point `(theta_i, phi)` outward along the dipole field line
``\\sin^2\\theta_{\\text{FAC}} = (r_{\\text{FAC}} / R_I)\\sin^2\\theta_I`` to the
magnetospheric FAC extraction shell `r_fac`.

Returns `valid = false` if the field line closes below `r_fac`. Otherwise returns the 3D
simulation coordinates `x_fac_sim` and `x_iono_sim`, unit outward radial vector
`r_hat_iono`, dipole field magnitude `b_iono_mag` at `r_iono`, and outward dipole radial
projection `br_over_b_iono = (B0_iono ⋅ r_hat_iono) / |B0_iono|`.
"""
@inline @muladd function dipole_fac_sample_point(
        frame::DipoleFrame{FT}, theta_i::Real, phi::Real, r_fac::Real,
    ) where {FT <: Real}
    T = float(promote_type(FT, typeof(theta_i), typeof(phi), typeof(r_fac)))
    th = T(theta_i)
    ph = T(phi)
    r_f = T(r_fac)
    r_i = T(frame.r_iono)

    sin_i, cos_i = sincos(th)
    ratio = r_f / r_i
    sin_f_sq = ratio * sin_i * sin_i
    z3 = zero(SVector{3, T})
    if sin_f_sq >= one(T)
        return (false, z3, z3, z3, zero(T), zero(T))
    end
    sin_f = sqrt(ratio) * (sin_i >= zero(T) ? sin_i : -sin_i)
    cos_f = (cos_i < zero(T) ? -one(T) : one(T)) * sqrt(one(T) - sin_f_sq)
    sp, cp = sincos(ph)

    r_hat_fac_smg = SVector{3, T}(sin_f * cp, sin_f * sp, cos_f)
    r_hat_iono_smg = SVector{3, T}(sin_i * cp, sin_i * sp, cos_i)

    rot_T = transpose(SMatrix{3, 3, T, 9}(frame.rot))
    r_hat_fac_sim = rot_T * r_hat_fac_smg
    r_hat_iono_sim = rot_T * r_hat_iono_smg

    ctr = SVector{3, T}(frame.center)
    x_fac_sim = ctr + r_f * r_hat_fac_sim
    x_iono_sim = ctr + r_i * r_hat_iono_sim

    denom_i = sqrt(one(T) + T(3) * cos_i * cos_i)
    inv_denom = inv(denom_i)
    inv_r_i = inv(r_i)
    b_iono_mag = (T(frame.M0) * (inv_r_i * inv_r_i * inv_r_i)) * denom_i
    br_over_b_iono = T(frame.dipole_sign) * (T(2) * cos_i) * inv_denom

    return (true, x_fac_sim, x_iono_sim, r_hat_iono_sim, b_iono_mag, br_over_b_iono)
end
