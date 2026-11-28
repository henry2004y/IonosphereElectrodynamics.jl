# ==============================================================================
# Height-Integrated Ionospheric Conductance Models
# ==============================================================================

"""
    AbstractConductanceModel

Abstract supertype for height-integrated ionospheric Pedersen and Hall conductance
models ``(\\Sigma_P, \\Sigma_H)`` evaluated on a 2D spherical thin-shell ionosphere.
"""
abstract type AbstractConductanceModel end

@inline _conductance_eltype(::AbstractConductanceModel) = Union{}
@inline _is_static_conductance(::AbstractConductanceModel) = false

"""
    UniformConductance(sigma_P, sigma_H = zero(sigma_P); FT = nothing) <: AbstractConductanceModel

Spatially uniform height-integrated Pedersen conductance `sigma_P > 0` and Hall
conductance `sigma_H`. Element type `FT` is inferred from arguments when not specified
(defaulting to `Float64` for integer inputs).
"""
struct UniformConductance{FT <: Real} <: AbstractConductanceModel
    sigma_P::FT
    sigma_H::FT
    function UniformConductance{FT}(sigma_P::Real, sigma_H::Real) where {FT <: Real}
        (isfinite(sigma_P) && sigma_P > 0) ||
            throw(ArgumentError("sigma_P must be finite and strictly positive (got $sigma_P)"))
        isfinite(sigma_H) || throw(ArgumentError("sigma_H must be finite (got $sigma_H)"))
        return new{FT}(FT(sigma_P), FT(sigma_H))
    end
end

@inline _conductance_eltype(::UniformConductance{FT}) where {FT} = FT
@inline _is_static_conductance(::UniformConductance) = true

function UniformConductance(
        sigma_P::Real,
        sigma_H::Real = zero(sigma_P);
        FT::Union{Nothing, Type{<:Real}} = nothing,
    )
    T = _infer_ft(FT, sigma_P, sigma_H)
    return UniformConductance{T}(sigma_P, sigma_H)
end

"""
    SolarEUVConductance(;
        sigma_P_day = 10.0,
        sigma_H_day = 15.0,
        sigma_P_night = 1.0,
        sigma_H_night = 0.5,
        sun_dir = (-1.0, 0.0, 0.0),
        exponent = 0.5,
        FT = nothing,
    ) <: AbstractConductanceModel

Dayside solar-EUV conductance model parameterized by solar zenith angle ``\\chi``
(``\\cos\\chi = \\hat{\\mathbf{r}} \\cdot \\hat{\\mathbf{s}}``):
```math
\\Sigma_P(\\chi) = \\sqrt{\\Sigma_{P,\\text{night}}^2 + (\\Sigma_{P,\\text{day}} \\max(0, \\cos\\chi)^\\alpha)^2}, \\quad
\\Sigma_H(\\chi) = \\sqrt{\\Sigma_{H,\\text{night}}^2 + (\\Sigma_{H,\\text{day}} \\max(0, \\cos\\chi)^\\alpha)^2}.
```
"""
struct SolarEUVConductance{FT <: Real} <: AbstractConductanceModel
    sigma_P_day::FT
    sigma_H_day::FT
    sigma_P_night::FT
    sigma_H_night::FT
    sun_dir::SVector{3, FT}
    exponent::FT
end

@inline _conductance_eltype(::SolarEUVConductance{FT}) where {FT} = FT
@inline _is_static_conductance(::SolarEUVConductance) = true

function SolarEUVConductance(
        ;
        sigma_P_day::Union{_Default, Real} = _DEFAULT,
        sigma_H_day::Union{_Default, Real} = _DEFAULT,
        sigma_P_night::Union{_Default, Real} = _DEFAULT,
        sigma_H_night::Union{_Default, Real} = _DEFAULT,
        sun_dir::Union{_Default, _Vec3Input} = _DEFAULT,
        exponent::Union{_Default, Real} = _DEFAULT,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    )
    T = _infer_ft(
        FT, sigma_P_day, sigma_H_day, sigma_P_night, sigma_H_night, sun_dir, exponent,
    )
    sp_day = _val_or(sigma_P_day, T(10))
    sh_day = _val_or(sigma_H_day, T(15))
    sp_ngt = _val_or(sigma_P_night, one(T))
    sh_ngt = _val_or(sigma_H_night, T(0.5))
    s_raw = _val_or(sun_dir, (-one(T), zero(T), zero(T)))
    exp_v = _val_or(exponent, T(0.5))

    (isfinite(sp_day) && sp_day >= 0) ||
        throw(ArgumentError("sigma_P_day must be finite and >= 0 (got $sp_day)"))
    (isfinite(sh_day) && sh_day >= 0) ||
        throw(ArgumentError("sigma_H_day must be finite and >= 0 (got $sh_day)"))
    (isfinite(sp_ngt) && sp_ngt > 0) ||
        throw(ArgumentError("sigma_P_night must be finite and strictly positive (got $sp_ngt)"))
    (isfinite(sh_ngt) && sh_ngt >= 0) ||
        throw(ArgumentError("sigma_H_night must be finite and >= 0 (got $sh_ngt)"))
    (isfinite(exp_v) && exp_v > 0) ||
        throw(ArgumentError("exponent must be finite and > 0 (got $exp_v)"))
    length(s_raw) == 3 || throw(ArgumentError("sun_dir must have length 3"))
    i0 = firstindex(s_raw)
    s = SVector{3, T}(T(s_raw[i0]), T(s_raw[i0 + 1]), T(s_raw[i0 + 2]))
    sn = norm(s)
    sn > zero(T) || throw(ArgumentError("sun_dir must be a nonzero vector"))
    return SolarEUVConductance{T}(
        T(sp_day), T(sh_day), T(sp_ngt), T(sh_ngt),
        s / sn, T(exp_v),
    )
end

"""
    MoenBrekkeEUVConductance(;
        f107 = 150.0,
        sigma_P_night = 1.0,
        sigma_H_night = 0.5,
        sun_dir = (1.0, 0.0, 0.0),
        FT = nothing,
    ) <: AbstractConductanceModel

Moen & Brekke (1993) solar-EUV ionospheric conductance model driven by the ``F_{10.7}``
radio flux index in SI units (Siemens), combined in quadrature with nightside floors:
```math
\\Sigma_{P,\\text{EUV}} = F_{10.7}^{0.49}(0.34\\cos\\chi + 0.93\\sqrt{\\cos\\chi}), \\quad
\\Sigma_{H,\\text{EUV}} = F_{10.7}^{0.53}(0.81\\cos\\chi + 0.54\\sqrt{\\cos\\chi}).
```
"""
struct MoenBrekkeEUVConductance{FT <: Real} <: AbstractConductanceModel
    f107::FT
    f107_ped::FT
    f107_hall::FT
    sigma_P_night::FT
    sigma_H_night::FT
    sun_dir::SVector{3, FT}
end

@inline _conductance_eltype(::MoenBrekkeEUVConductance{FT}) where {FT} = FT
@inline _is_static_conductance(::MoenBrekkeEUVConductance) = true

function MoenBrekkeEUVConductance(
        ;
        f107::Union{_Default, Real} = _DEFAULT,
        sigma_P_night::Union{_Default, Real} = _DEFAULT,
        sigma_H_night::Union{_Default, Real} = _DEFAULT,
        sun_dir::Union{_Default, _Vec3Input} = _DEFAULT,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    )
    T = _infer_ft(FT, f107, sigma_P_night, sigma_H_night, sun_dir)
    sT = _scalar_float_type(T)
    f107_v = _val_or(f107, T(150))
    sp_ngt = _val_or(sigma_P_night, one(T))
    sh_ngt = _val_or(sigma_H_night, T(0.5))
    s_raw = _val_or(sun_dir, (one(T), zero(T), zero(T)))

    (isfinite(f107_v) && f107_v >= 1) ||
        throw(ArgumentError("f107 must be finite and >= 1.0 (got $f107_v)"))
    (isfinite(sp_ngt) && sp_ngt > 0) ||
        throw(ArgumentError("sigma_P_night must be finite and strictly positive (got $sp_ngt)"))
    (isfinite(sh_ngt) && sh_ngt >= 0) ||
        throw(ArgumentError("sigma_H_night must be finite and >= 0 (got $sh_ngt)"))
    length(s_raw) == 3 || throw(ArgumentError("sun_dir must have length 3"))
    i0 = firstindex(s_raw)
    s = SVector{3, T}(T(s_raw[i0]), T(s_raw[i0 + 1]), T(s_raw[i0 + 2]))
    sn = norm(s)
    sn > zero(T) || throw(ArgumentError("sun_dir must be a nonzero vector"))
    f107_ft = T(f107_v)
    return MoenBrekkeEUVConductance{T}(
        f107_ft,
        T(f107_ft^sT(0.49)),
        T(f107_ft^sT(0.53)),
        T(sp_ngt),
        T(sh_ngt),
        s / sn,
    )
end

"""
    AuroralConductance(
        base::AbstractConductanceModel;
        sigma_P_max = 15.0,
        hall_ratio = 1.5,
        fac_scale = 1.0e-6,
        mode = :tanh_sqrt,
        FT = nothing,
    ) <: AbstractConductanceModel

Dynamic upward-FAC auroral precipitation conductance wrapper around any `base`
[`AbstractConductanceModel`](@ref). Enhances ``(\\Sigma_P, \\Sigma_H)`` in quadrature
where outward radial FAC ``J_{\\text{up}} = \\max(0, J_r^{\\text{outward}}) > 0``:
- `mode = :tanh_sqrt` (FLEKS/RIM): ``\\Delta\\Sigma_P = \\Sigma_{P,\\max}\\tanh(\\sqrt{J_{\\text{up}}/J_0})``.
- `mode = :saturating`: ``\\Delta\\Sigma_P = \\Sigma_{P,\\max} J_{\\text{up}} / (J_{\\text{up}} + J_0)``.
"""
struct AuroralConductance{FT <: Real, B <: AbstractConductanceModel} <: AbstractConductanceModel
    base::B
    sigma_P_max::FT
    hall_ratio::FT
    fac_scale::FT
    inv_fac_scale::FT
    mode::Symbol
end

@inline _conductance_eltype(m::AuroralConductance{FT}) where {FT} =
    promote_type(FT, _conductance_eltype(m.base))

AuroralConductance{FT, B}(
    base::B, sigma_P_max::FT, hall_ratio::FT, fac_scale::FT, mode::Symbol,
) where {FT <: Real, B <: AbstractConductanceModel} =
    AuroralConductance{FT, B}(base, sigma_P_max, hall_ratio, fac_scale, inv(fac_scale), mode)

function AuroralConductance(
        base::B;
        sigma_P_max::Union{_Default, Real} = _DEFAULT,
        hall_ratio::Union{_Default, Real} = _DEFAULT,
        fac_scale::Union{_Default, Real} = _DEFAULT,
        mode::Symbol = :tanh_sqrt,
        FT::Union{Nothing, Type{<:Real}} = nothing,
    ) where {B <: AbstractConductanceModel}
    T = _infer_ft(FT, _conductance_eltype(base), sigma_P_max, hall_ratio, fac_scale)
    sT = _scalar_float_type(T)
    sp_max = _val_or(sigma_P_max, T(15))
    h_rat = _val_or(hall_ratio, T(1.5))
    f_scl = _val_or(fac_scale, T(sT(1.0e-6)))

    (isfinite(sp_max) && sp_max >= 0) ||
        throw(ArgumentError("sigma_P_max must be finite and >= 0 (got $sp_max)"))
    (isfinite(h_rat) && h_rat >= 0) ||
        throw(ArgumentError("hall_ratio must be finite and >= 0 (got $h_rat)"))
    (isfinite(f_scl) && f_scl > 0) ||
        throw(ArgumentError("fac_scale must be finite and > 0 (got $f_scl)"))
    mode in (:tanh_sqrt, :saturating) ||
        throw(ArgumentError("mode must be :tanh_sqrt or :saturating (got $mode)"))
    fs = T(f_scl)
    return AuroralConductance{T, B}(
        base, T(sp_max), T(h_rat), fs, inv(fs), mode,
    )
end

"""
    CustomConductance(fn) <: AbstractConductanceModel

User-supplied conductance callback `fn(theta, phi, t) -> (sigma_P, sigma_H)` where
`theta ∈ [0, π]` is dipole colatitude, `phi ∈ [0, 2π)` is dipole longitude, and `t` is
simulation time. `sigma_P` must be finite and strictly positive everywhere.
"""
struct CustomConductance{F} <: AbstractConductanceModel
    fn::F
end

"""
    evaluate_conductance(
        model::AbstractConductanceModel,
        theta::Real,
        phi::Real,
        r_hat,
        t::Real;
        jr_outward::Real = 0,
    ) -> Tuple{T, T}

Evaluate height-integrated Pedersen and Hall conductances `(sigma_P, sigma_H)` for
`model` at dipole colatitude `theta`, longitude `phi`, simulation-frame unit radial
vector `r_hat` (`SVector{3}`, `AbstractVector`, or `NTuple{3}`), simulation time `t`,
and optional outward FAC `jr_outward`.
"""
@inline function evaluate_conductance(
        m::UniformConductance{FT},
        theta::Real,
        phi::Real,
        r_hat::_Vec3Input,
        t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT <: Real}
    return (m.sigma_P, m.sigma_H)
end

@inline @muladd function evaluate_conductance(
        m::SolarEUVConductance{FT},
        theta::Real,
        phi::Real,
        r_hat::_Vec3Input,
        t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT <: Real}
    rh = _as_svec3(r_hat)
    T = float(promote_type(FT, eltype(rh)))
    c = dot(rh, m.sun_dir)
    if c > zero(T)
        f = c^m.exponent
        sp_d = m.sigma_P_day * f
        sh_d = m.sigma_H_day * f
        sp = sqrt(m.sigma_P_night * m.sigma_P_night + sp_d * sp_d)
        sh2 = m.sigma_H_night * m.sigma_H_night + sh_d * sh_d
        sh = sh2 > zero(T) ? sqrt(sh2) : T(m.sigma_H_night)
        return (T(sp), T(sh))
    end
    return (T(m.sigma_P_night), T(m.sigma_H_night))
end

@inline @muladd function evaluate_conductance(
        m::MoenBrekkeEUVConductance{FT},
        theta::Real,
        phi::Real,
        r_hat::_Vec3Input,
        t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT <: Real}
    rh = _as_svec3(r_hat)
    T = float(promote_type(FT, eltype(rh)))
    sT = _scalar_float_type(T)
    c = dot(rh, m.sun_dir)
    if c > zero(T)
        sc = sqrt(c)
        sp_euv = m.f107_ped * (sT(0.34) * c + sT(0.93) * sc)
        sh_euv = m.f107_hall * (sT(0.81) * c + sT(0.54) * sc)
        return (
            T(sqrt(m.sigma_P_night * m.sigma_P_night + sp_euv * sp_euv)),
            T(sqrt(m.sigma_H_night * m.sigma_H_night + sh_euv * sh_euv)),
        )
    end
    return (T(m.sigma_P_night), T(m.sigma_H_night))
end

@inline @muladd function evaluate_conductance(
        m::AuroralConductance{FT},
        theta::Real,
        phi::Real,
        r_hat::_Vec3Input,
        t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT <: Real}
    sp_base, sh_base = evaluate_conductance(
        m.base, theta, phi, r_hat, t; jr_outward = jr_outward,
    )
    T = float(promote_type(FT, typeof(sp_base), typeof(sh_base), typeof(jr_outward)))
    sp = T(sp_base)
    sh = T(sh_base)
    jr = T(jr_outward)
    if jr > zero(T) && m.sigma_P_max > zero(T)
        ratio = jr * m.inv_fac_scale
        f_aur = m.mode === :tanh_sqrt ? tanh(sqrt(ratio)) : (ratio / (ratio + one(T)))
        sp_aur = m.sigma_P_max * f_aur
        sh_aur = m.hall_ratio * sp_aur
        sp = sqrt(sp * sp + sp_aur * sp_aur)
        sh2 = sh * sh + sh_aur * sh_aur
        sh = sh2 > zero(T) ? sqrt(sh2) : sh
    end
    return (sp, sh)
end

@inline function evaluate_conductance(
        m::CustomConductance,
        theta::Real,
        phi::Real,
        r_hat::_Vec3Input,
        t::Real;
        jr_outward::Real = 0,
    )
    sp, sh = m.fn(theta, phi, t)
    T = float(promote_type(typeof(sp), typeof(sh)))
    sp_f = T(sp)
    sh_f = T(sh)
    (isfinite(sp_f) && sp_f > zero(T)) ||
        throw(ArgumentError("CustomConductance returned invalid sigma_P = $sp (must be finite and > 0)"))
    isfinite(sh_f) ||
        throw(ArgumentError("CustomConductance returned non-finite sigma_H = $sh"))
    return (sp_f, sh_f)
end

"""
    electron_mobility_tensor(sigma_p::Real, sigma_h::Real, b_hat; sigma_0::Real = sigma_p) -> SMatrix{3,3}

Construct the anisotropic 3D conductance/mobility tensor
`sigma_0 b bᵀ + sigma_p (I - b bᵀ) + sigma_h (b ×)`. Accepts any length-3 `SVector`,
`AbstractVector`, or `NTuple` for `b_hat` and promotes across generic `Real` types.
"""
@inline @muladd function electron_mobility_tensor(
        sigma_p::Real,
        sigma_h::Real,
        b_hat::_Vec3Input;
        sigma_0::Real = sigma_p,
    )
    bh = _as_svec3(b_hat)
    T = float(promote_type(typeof(sigma_p), typeof(sigma_h), typeof(sigma_0), eltype(bh)))
    bx, by, bz = T(bh[1]), T(bh[2]), T(bh[3])
    sp = T(sigma_p)
    sh = T(sigma_h)
    s0 = T(sigma_0)
    ds = s0 - sp
    return SMatrix{3, 3, T, 9}(
        sp + ds * bx * bx, ds * bx * by + sh * bz, ds * bx * bz - sh * by,
        ds * by * bx - sh * bz, sp + ds * by * by, ds * by * bz + sh * bx,
        ds * bz * bx + sh * by, ds * bz * by - sh * bx, sp + ds * bz * bz,
    )
end
