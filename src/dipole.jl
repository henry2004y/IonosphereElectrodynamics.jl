# ==============================================================================
# Dipole Orthonormal Frame, Analytic Field-Line Mapping, Jacobian & Tensor
# ==============================================================================

"""
    DipoleFrame{FT <: AbstractFloat}

Immutable, `isbits` orthonormal Solar Magnetic (SMG) coordinate frame built from a 3D
planetary dipole moment vector.

- `center`: Body / dipole center in simulation coordinates.
- `rot`: Orthonormal rotation matrix whose rows `(e1, e2, e3)` map simulation vectors to
  SMG coordinates (`x_smg = rot * (x_sim - center)`), with `e3[3] >= 0` pointing toward
  geographic/simulation North.
- `dipole_moment`, `M0`: 3D dipole moment vector and magnitude `norm(dipole_moment)`.
- `dipole_sign`: `-1` for Earth-like polarity (`m ⋅ e3 < 0`), `+1` for reversed polarity.
- `r_iono`: Ionospheric shell radius in simulation length units.
"""
struct DipoleFrame{FT <: AbstractFloat}
    center::SVector{3, FT}
    rot::SMatrix{3, 3, FT, 9}
    dipole_moment::SVector{3, FT}
    M0::FT
    dipole_sign::FT
    r_iono::FT
end

"""
    build_dipole_frame(
        dipole_moment;
        center = (0.0, 0.0, 0.0),
        r_iono = 1.0,
        FT = Float64,
    ) -> DipoleFrame{FT}

Construct a [`DipoleFrame`](@ref) from `dipole_moment`, matching FLEKS's SMG convention:
`e3` always points toward northern simulation hemisphere (`e3[3] >= 0`), `dipole_sign`
records the polarity along `e3` (`-1` Earth-like, `+1` reversed), and `e1` is the
projection of simulation `+x` (falling back to `+y` when `|e3[1]| > 0.99`) orthogonal
to `e3`.
"""
function build_dipole_frame(
        dipole_moment::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}};
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (0.0, 0.0, 0.0),
        r_iono::Real = 1.0,
        FT::Type{<:AbstractFloat} = Float64,
    )
    length(dipole_moment) == 3 || throw(ArgumentError("dipole_moment must have length 3"))
    length(center) == 3 || throw(ArgumentError("center must have length 3"))
    (isfinite(r_iono) && r_iono > 0) || throw(ArgumentError("r_iono must be finite and > 0 (got $r_iono)"))

    m = SVector{3, FT}(FT(dipole_moment[1]), FT(dipole_moment[2]), FT(dipole_moment[3]))
    M0 = norm(m)
    M0 > zero(FT) || throw(ArgumentError("dipole_moment must be non-zero"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))

    ax = m / M0
    dipole_sign = ax[3] < zero(FT) ? -one(FT) : one(FT)
    ez = dipole_sign * ax

    ref = abs(ez[1]) > FT(0.99) ?
        SVector{3, FT}(zero(FT), one(FT), zero(FT)) :
        SVector{3, FT}(one(FT), zero(FT), zero(FT))
    ex_raw = ref - dot(ref, ez) * ez
    ex = ex_raw / norm(ex_raw)
    ey = cross(ez, ex)

    # Row-major storage in SMatrix: rows = (ex, ey, ez)
    rot = SMatrix{3, 3, FT, 9}(
        ex[1], ey[1], ez[1],
        ex[2], ey[2], ez[2],
        ex[3], ey[3], ez[3],
    )
    return DipoleFrame{FT}(c, rot, m, M0, dipole_sign, FT(r_iono))
end

"""
    dipole_B0(frame::DipoleFrame{FT}, x_sim::SVector{3, FT}; r_min = eps(FT)) -> SVector{3, FT}

Evaluate the analytic dipole magnetic field ``\\mathbf{B}_0(\\mathbf{x})`` at simulation
position `x_sim` using `frame.center` and `frame.dipole_moment` (softened at `r_min`).
"""
@inline @muladd function dipole_B0(
        frame::DipoleFrame{FT}, x_sim::SVector{3, FT}; r_min::FT = eps(FT),
    ) where {FT <: AbstractFloat}
    dr = x_sim - frame.center
    r2 = dr[1] * dr[1] + dr[2] * dr[2] + dr[3] * dr[3]
    r = max(sqrt(r2), r_min)
    inv_r = inv(r)
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r2 * inv_r
    inv_r5 = inv_r3 * inv_r2
    m = frame.dipole_moment
    mdr = m[1] * dr[1] + m[2] * dr[2] + m[3] * dr[3]
    return (FT(3) * mdr * inv_r5) * dr - inv_r3 * m
end

@inline @muladd function _map_dipole_smg_core(
        x_smg::SVector{3, FT}, r_map::FT, ::Val{ComputeJac},
    ) where {FT <: AbstractFloat, ComputeJac}
    x, y, z = x_smg[1], x_smg[2], x_smg[3]
    rho2 = x * x + y * y
    r2 = rho2 + z * z
    r = sqrt(r2)
    z3 = zero(SVector{3, FT})
    z_jac = zero(SMatrix{2, 3, FT, 6})
    if r <= FT(1.0e-12) || r_map <= FT(1.0e-12)
        return (0, z3, zero(FT), zero(FT), z_jac)
    end

    i_hem = z >= zero(FT) ? 1 : -1
    ratio = r_map / r
    xy_ratio = ratio * sqrt(ratio) # (r_map / r)^(3/2)

    xm = xy_ratio * x
    ym = xy_ratio * y
    rho_map2 = xm * xm + ym * ym
    r_map2 = r_map * r_map
    two_pi = FT(2 * pi)

    if rho_map2 > r_map2
        # Field line closes below r_map (L < r_map).
        rho = sqrt(rho2)
        if rho > zero(FT)
            inv_rho_map = r_map / rho
            xm = inv_rho_map * x
            ym = inv_rho_map * y
        else
            xm = r_map
            ym = zero(FT)
        end
        x_map = SVector{3, FT}(xm, ym, zero(FT))
        a = atan(ym, xm)
        phi_map = a < zero(FT) ? a + two_pi : (a == zero(FT) ? zero(FT) : a)
        return (0, x_map, FT(0.5 * pi), phi_map, z_jac)
    end

    zm = FT(i_hem) * sqrt(max(zero(FT), r_map2 - rho_map2))
    x_map = SVector{3, FT}(xm, ym, zm)
    cos_th = clamp(zm / r_map, -one(FT), one(FT))
    theta_map = acos(cos_th)
    a = atan(ym, xm)
    phi_map = a < zero(FT) ? a + two_pi : (a == zero(FT) ? zero(FT) : a)

    jac = z_jac
    if ComputeJac
        rho = sqrt(rho2)
        rho_map = sqrt(rho_map2)
        if abs(zm) >= FT(1.0e-6) * r_map && rho >= FT(1.0e-8) * r
            inv_r2 = inv(r2)
            inv_rho2 = inv(rho2)
            inv_zm = inv(zm)
            z_over_r = z / r
            factor = -(FT(0.5) - FT(1.5) * z_over_r * z_over_r) * inv_zm / rho
            j00 = xm * factor
            j01 = ym * factor
            j02 = -FT(1.5) * (rho_map * inv_zm) * (z * inv_r2)
            j10 = -y * inv_rho2
            j11 = x * inv_rho2
            j12 = zero(FT)
            jac = SMatrix{2, 3, FT, 6}(j00, j10, j01, j11, j02, j12)
        end
    end

    return (i_hem, x_map, theta_map, phi_map, jac)
end

"""
    map_dipole_smg(x_smg::SVector{3, FT}, r_map::FT) -> (i_hem, x_map, theta_map, phi_map)

Map SMG position `x_smg` along the dipole field line ``r = L\\sin^2\\theta`` to radius
`r_map`. Returns `(i_hem, x_map, theta_map, phi_map)` where `i_hem` is `+1` (north), `-1`
(south), or `0` (closed equatorial field line with ``L < r_{\\text{map}}``).
"""
@inline function map_dipole_smg(x_smg::SVector{3, FT}, r_map::FT) where {FT <: AbstractFloat}
    i_hem, x_map, theta_map, phi_map, _ = _map_dipole_smg_core(x_smg, r_map, Val(false))
    return (i_hem, x_map, theta_map, phi_map)
end

"""
    map_dipole_smg_with_jacobian(
        x_smg::SVector{3, FT}, r_map::FT
    ) -> (i_hem, x_map, theta_map, phi_map, jac_smg)

Map SMG position `x_smg` along the dipole field line ``r = L\\sin^2\\theta`` to radius
`r_map`, returning `(i_hem, x_map, theta_map, phi_map, jac_smg)` where
`jac_smg::SMatrix{2, 3, FT, 6}` is the exact closed-form Jacobian
``\\partial(\\theta_m, \\phi_m) / \\partial(x, y, z)`` in SMG coordinates.
"""
@inline function map_dipole_smg_with_jacobian(
        x_smg::SVector{3, FT}, r_map::FT,
    ) where {FT <: AbstractFloat}
    return _map_dipole_smg_core(x_smg, r_map, Val(true))
end

@inline @muladd function _dipole_conductance_tensor_core(
        sigma_P::FT,
        sigma_H::FT,
        cos_theta::FT,
        sin_theta::FT,
        dipole_sign::FT,
        s0::FT,
        use_dip_angle::Bool,
    ) where {FT <: AbstractFloat}
    hall_sign = -dipole_sign
    if !use_dip_angle
        hemi = abs(cos_theta) <= FT(1.0e-12) ? zero(FT) : (cos_theta > zero(FT) ? one(FT) : -one(FT))
        return (sigma_P, hall_sign * hemi * sigma_H, sigma_P)
    end
    sn2 = sin_theta * sin_theta
    cs2 = cos_theta * cos_theta
    cs3 = one(FT) + FT(3) * cs2
    cs4 = sqrt(cs3)
    C = FT(4) * s0 * cs2 + sigma_P * sn2
    inv_C = inv(C)
    s_tt = (s0 * sigma_P * cs3) * inv_C
    s_tp = hall_sign * (FT(2) * s0 * sigma_H * cos_theta * cs4) * inv_C
    s_pp = sigma_P + (sigma_H * sigma_H * sn2) * inv_C
    return (s_tt, s_tp, s_pp)
end

"""
    dipole_conductance_tensor(
        sigma_P::FT,
        sigma_H::FT,
        cos_theta::FT,
        sin_theta::FT,
        dipole_sign::FT;
        sigma_0_ratio = FT(1000),
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
        sigma_P::FT,
        sigma_H::FT,
        cos_theta::FT,
        sin_theta::FT,
        dipole_sign::FT;
        sigma_0_ratio::FT = FT(1000),
        sigma_parallel::Union{Nothing, FT} = nothing,
        use_dip_angle::Bool = true,
    ) where {FT <: AbstractFloat}
    s0 = sigma_parallel === nothing ? (sigma_0_ratio * sigma_P) : max(sigma_parallel, FT(10))
    return _dipole_conductance_tensor_core(
        sigma_P, sigma_H, cos_theta, sin_theta, dipole_sign, s0, use_dip_angle,
    )
end

"""
    dipole_fac_sample_point(
        frame::DipoleFrame{FT},
        theta_i::FT,
        phi::FT,
        r_fac::FT,
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
        frame::DipoleFrame{FT}, theta_i::FT, phi::FT, r_fac::FT,
    ) where {FT <: AbstractFloat}
    sin_i, cos_i = sincos(theta_i)
    r_i = frame.r_iono
    sin_f_sq = (r_fac / r_i) * sin_i * sin_i
    z3 = zero(SVector{3, FT})
    if sin_f_sq >= one(FT)
        return (false, z3, z3, z3, zero(FT), zero(FT))
    end
    sin_f = sqrt(sin_f_sq)
    cos_f = copysign(sqrt(max(one(FT) - sin_f_sq, zero(FT))), cos_i)
    sp, cp = sincos(phi)

    r_hat_fac_smg = SVector{3, FT}(sin_f * cp, sin_f * sp, cos_f)
    r_hat_iono_smg = SVector{3, FT}(sin_i * cp, sin_i * sp, cos_i)

    rot_T = transpose(frame.rot)
    r_hat_fac_sim = rot_T * r_hat_fac_smg
    r_hat_iono_sim = rot_T * r_hat_iono_smg

    x_fac_sim = frame.center + r_fac * r_hat_fac_sim
    x_iono_sim = frame.center + r_i * r_hat_iono_sim

    denom_i = sqrt(one(FT) + FT(3) * cos_i * cos_i)
    inv_denom = inv(denom_i)
    inv_r_i = inv(r_i)
    b_iono_mag = (frame.M0 * (inv_r_i * inv_r_i * inv_r_i)) * denom_i
    br_over_b_iono = frame.dipole_sign * (FT(2) * cos_i) * inv_denom

    return (true, x_fac_sim, x_iono_sim, r_hat_iono_sim, b_iono_mag, br_over_b_iono)
end
