# ==============================================================================
# Height-Integrated Ionospheric Conductance Models
# ==============================================================================

"""
    AbstractConductanceModel

Abstract supertype for height-integrated ionospheric Pedersen and Hall conductance
models ``(\\Sigma_P, \\Sigma_H)`` evaluated on a 2D spherical thin-shell ionosphere.
"""
abstract type AbstractConductanceModel end

"""
    UniformConductance(sigma_P, sigma_H = 0.0; FT = Float64) <: AbstractConductanceModel

Spatially uniform height-integrated Pedersen conductance `sigma_P > 0` and Hall
conductance `sigma_H`.
"""
struct UniformConductance{FT <: AbstractFloat} <: AbstractConductanceModel
    sigma_P::FT
    sigma_H::FT
    function UniformConductance{FT}(sigma_P::Real, sigma_H::Real) where {FT <: AbstractFloat}
        (isfinite(sigma_P) && sigma_P > 0) ||
            throw(ArgumentError("sigma_P must be finite and strictly positive (got $sigma_P)"))
        isfinite(sigma_H) || throw(ArgumentError("sigma_H must be finite (got $sigma_H)"))
        return new{FT}(FT(sigma_P), FT(sigma_H))
    end
end

UniformConductance(sigma_P::Real, sigma_H::Real = 0.0; FT::Type{<:AbstractFloat} = Float64) =
    UniformConductance{FT}(sigma_P, sigma_H)

"""
    SolarEUVConductance(;
        sigma_P_day = 10.0,
        sigma_H_day = 15.0,
        sigma_P_night = 1.0,
        sigma_H_night = 0.5,
        sun_dir = (-1.0, 0.0, 0.0),
        exponent = 0.5,
        FT = Float64,
    ) <: AbstractConductanceModel

Dayside solar-EUV conductance model parameterized by solar zenith angle ``\\chi``
(``\\cos\\chi = \\hat{\\mathbf{r}} \\cdot \\hat{\\mathbf{s}}``):
```math
\\Sigma_P(\\chi) = \\sqrt{\\Sigma_{P,\\text{night}}^2 + (\\Sigma_{P,\\text{day}} \\max(0, \\cos\\chi)^\\alpha)^2}, \\quad
\\Sigma_H(\\chi) = \\sqrt{\\Sigma_{H,\\text{night}}^2 + (\\Sigma_{H,\\text{day}} \\max(0, \\cos\\chi)^\\alpha)^2}.
```
"""
struct SolarEUVConductance{FT <: AbstractFloat} <: AbstractConductanceModel
    sigma_P_day::FT
    sigma_H_day::FT
    sigma_P_night::FT
    sigma_H_night::FT
    sun_dir::SVector{3, FT}
    exponent::FT
end

function SolarEUVConductance(;
        sigma_P_day::Real = 10.0,
        sigma_H_day::Real = 15.0,
        sigma_P_night::Real = 1.0,
        sigma_H_night::Real = 0.5,
        sun_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (-1.0, 0.0, 0.0),
        exponent::Real = 0.5,
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(sigma_P_day) && sigma_P_day >= 0) ||
        throw(ArgumentError("sigma_P_day must be finite and >= 0 (got $sigma_P_day)"))
    (isfinite(sigma_H_day) && sigma_H_day >= 0) ||
        throw(ArgumentError("sigma_H_day must be finite and >= 0 (got $sigma_H_day)"))
    (isfinite(sigma_P_night) && sigma_P_night > 0) ||
        throw(ArgumentError("sigma_P_night must be finite and strictly positive (got $sigma_P_night)"))
    (isfinite(sigma_H_night) && sigma_H_night >= 0) ||
        throw(ArgumentError("sigma_H_night must be finite and >= 0 (got $sigma_H_night)"))
    (isfinite(exponent) && exponent > 0) ||
        throw(ArgumentError("exponent must be finite and > 0 (got $exponent)"))
    length(sun_dir) == 3 || throw(ArgumentError("sun_dir must have length 3"))
    s = SVector{3, FT}(FT(sun_dir[1]), FT(sun_dir[2]), FT(sun_dir[3]))
    sn = norm(s)
    sn > zero(FT) || throw(ArgumentError("sun_dir must be a nonzero vector"))
    return SolarEUVConductance{FT}(
        FT(sigma_P_day), FT(sigma_H_day), FT(sigma_P_night), FT(sigma_H_night),
        s / sn, FT(exponent),
    )
end

"""
    MoenBrekkeEUVConductance(;
        f107 = 150.0,
        sigma_P_night = 1.0,
        sigma_H_night = 0.5,
        sun_dir = (1.0, 0.0, 0.0),
        FT = Float64,
    ) <: AbstractConductanceModel

Moen & Brekke (1993) solar-EUV ionospheric conductance model driven by the ``F_{10.7}``
radio flux index in SI units (Siemens), combined in quadrature with nightside floors:
```math
\\Sigma_{P,\\text{EUV}} = F_{10.7}^{0.49}(0.34\\cos\\chi + 0.93\\sqrt{\\cos\\chi}), \\quad
\\Sigma_{H,\\text{EUV}} = F_{10.7}^{0.53}(0.81\\cos\\chi + 0.54\\sqrt{\\cos\\chi}).
```
"""
struct MoenBrekkeEUVConductance{FT <: AbstractFloat} <: AbstractConductanceModel
    f107::FT
    f107_ped::FT
    f107_hall::FT
    sigma_P_night::FT
    sigma_H_night::FT
    sun_dir::SVector{3, FT}
end

function MoenBrekkeEUVConductance(;
        f107::Real = 150.0,
        sigma_P_night::Real = 1.0,
        sigma_H_night::Real = 0.5,
        sun_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (1.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(f107) && f107 >= 1.0) ||
        throw(ArgumentError("f107 must be finite and >= 1.0 (got $f107)"))
    (isfinite(sigma_P_night) && sigma_P_night > 0) ||
        throw(ArgumentError("sigma_P_night must be finite and strictly positive (got $sigma_P_night)"))
    (isfinite(sigma_H_night) && sigma_H_night >= 0) ||
        throw(ArgumentError("sigma_H_night must be finite and >= 0 (got $sigma_H_night)"))
    length(sun_dir) == 3 || throw(ArgumentError("sun_dir must have length 3"))
    s = SVector{3, FT}(FT(sun_dir[1]), FT(sun_dir[2]), FT(sun_dir[3]))
    sn = norm(s)
    sn > zero(FT) || throw(ArgumentError("sun_dir must be a nonzero vector"))
    f107_ft = FT(f107)
    return MoenBrekkeEUVConductance{FT}(
        f107_ft,
        f107_ft^FT(0.49),
        f107_ft^FT(0.53),
        FT(sigma_P_night),
        FT(sigma_H_night),
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
        FT = Float64,
    ) <: AbstractConductanceModel

Dynamic upward-FAC auroral precipitation conductance wrapper around any `base`
[`AbstractConductanceModel`](@ref). Enhances ``(\\Sigma_P, \\Sigma_H)`` in quadrature
where outward radial FAC ``J_{\\text{up}} = \\max(0, J_r^{\\text{outward}}) > 0``:
- `mode = :tanh_sqrt` (FLEKS/RIM): ``\\Delta\\Sigma_P = \\Sigma_{P,\\max}\\tanh(\\sqrt{J_{\\text{up}}/J_0})``.
- `mode = :saturating`: ``\\Delta\\Sigma_P = \\Sigma_{P,\\max} J_{\\text{up}} / (J_{\\text{up}} + J_0)``.
"""
struct AuroralConductance{FT <: AbstractFloat, B <: AbstractConductanceModel} <: AbstractConductanceModel
    base::B
    sigma_P_max::FT
    hall_ratio::FT
    fac_scale::FT
    mode::Symbol
end

function AuroralConductance(
        base::B;
        sigma_P_max::Real = 15.0,
        hall_ratio::Real = 1.5,
        fac_scale::Real = 1.0e-6,
        mode::Symbol = :tanh_sqrt,
        FT::Type{<:AbstractFloat} = Float64,
    ) where {B <: AbstractConductanceModel}
    (isfinite(sigma_P_max) && sigma_P_max >= 0) ||
        throw(ArgumentError("sigma_P_max must be finite and >= 0 (got $sigma_P_max)"))
    (isfinite(hall_ratio) && hall_ratio >= 0) ||
        throw(ArgumentError("hall_ratio must be finite and >= 0 (got $hall_ratio)"))
    (isfinite(fac_scale) && fac_scale > 0) ||
        throw(ArgumentError("fac_scale must be finite and > 0 (got $fac_scale)"))
    mode in (:tanh_sqrt, :saturating) ||
        throw(ArgumentError("mode must be :tanh_sqrt or :saturating (got $mode)"))
    return AuroralConductance{FT, B}(
        base, FT(sigma_P_max), FT(hall_ratio), FT(fac_scale), mode,
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
        r_hat::SVector{3, FT},
        t::Real;
        jr_outward::Real = zero(FT),
    ) -> Tuple{FT, FT}

Evaluate height-integrated Pedersen and Hall conductances `(sigma_P, sigma_H)` for
`model` at dipole colatitude `theta`, longitude `phi`, simulation-frame unit radial
vector `r_hat`, simulation time `t`, and optional outward FAC `jr_outward`.
"""
@inline function evaluate_conductance(
        m::UniformConductance{FT}, theta::Real, phi::Real, r_hat::SVector{3, FT}, t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT}
    return (m.sigma_P, m.sigma_H)
end

@inline function evaluate_conductance(
        m::SolarEUVConductance{FT}, theta::Real, phi::Real, r_hat::SVector{3, FT}, t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT}
    c = max(zero(FT), dot(r_hat, m.sun_dir))
    f = c > zero(FT) ? c^m.exponent : zero(FT)
    sp = hypot(m.sigma_P_night, m.sigma_P_day * f)
    sh = hypot(m.sigma_H_night, m.sigma_H_day * f)
    return (sp, sh)
end

@inline function evaluate_conductance(
        m::MoenBrekkeEUVConductance{FT}, theta::Real, phi::Real, r_hat::SVector{3, FT}, t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT}
    c = max(zero(FT), dot(r_hat, m.sun_dir))
    if c > zero(FT)
        sc = sqrt(c)
        sp_euv = m.f107_ped * (FT(0.34) * c + FT(0.93) * sc)
        sh_euv = m.f107_hall * (FT(0.81) * c + FT(0.54) * sc)
        return (hypot(m.sigma_P_night, sp_euv), hypot(m.sigma_H_night, sh_euv))
    end
    return (m.sigma_P_night, m.sigma_H_night)
end

@inline function evaluate_conductance(
        m::AuroralConductance{FT}, theta::Real, phi::Real, r_hat::SVector{3, FT}, t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT}
    sp_base, sh_base = evaluate_conductance(
        m.base, theta, phi, r_hat, t; jr_outward = jr_outward,
    )
    sp = FT(sp_base)
    sh = FT(sh_base)
    j_up = max(zero(FT), FT(jr_outward))
    if j_up > zero(FT) && m.sigma_P_max > zero(FT)
        ratio = j_up / m.fac_scale
        f_aur = m.mode === :tanh_sqrt ? tanh(sqrt(ratio)) : (ratio / (ratio + one(FT)))
        sp_aur = m.sigma_P_max * f_aur
        sh_aur = m.hall_ratio * sp_aur
        sp = hypot(sp, sp_aur)
        sh = hypot(sh, sh_aur)
    end
    return (sp, sh)
end

@inline function evaluate_conductance(
        m::CustomConductance, theta::Real, phi::Real, r_hat::SVector{3, FT}, t::Real;
        jr_outward::Real = zero(FT),
    ) where {FT}
    sp, sh = m.fn(theta, phi, t)
    sp_f = FT(sp)
    sh_f = FT(sh)
    (isfinite(sp_f) && sp_f > zero(FT)) ||
        throw(ArgumentError("CustomConductance returned invalid sigma_P = $sp (must be finite and > 0)"))
    isfinite(sh_f) ||
        throw(ArgumentError("CustomConductance returned non-finite sigma_H = $sh"))
    return (sp_f, sh_f)
end
