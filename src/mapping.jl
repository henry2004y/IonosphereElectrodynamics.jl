# ==============================================================================
# GPU/KA-Safe 3D Field View & Analytic Dipole Potential/Convection Evaluation
# ==============================================================================

"""
    IonosphereFieldView{FT <: AbstractFloat, Mat <: AbstractMatrix{FT}}

Lightweight, `Adapt.adapt`-friendly view of solved ionospheric potential gradients,
pole gradients, dipole frame, and planetary corotation angular velocity for 3D CPU/GPU
boundary evaluation kernels.
"""
struct IonosphereFieldView{FT <: AbstractFloat, Mat <: AbstractMatrix{FT}}
    frame::DipoleFrame{FT}
    n_theta::Int
    n_phi::Int
    dtheta::FT
    dphi::FT
    lat_boundary_rad::FT
    omega_corot::SVector{3, FT}
    dphi_dtheta::Mat
    dphi_dphi::Mat
    pole_grad_N::SVector{2, FT}
    pole_grad_S::SVector{2, FT}
end

function Adapt.adapt_structure(to, v::IonosphereFieldView{FT}) where {FT}
    dth_dev = Adapt.adapt(to, v.dphi_dtheta)
    dph_dev = Adapt.adapt(to, v.dphi_dphi)
    return IonosphereFieldView{FT, typeof(dth_dev)}(
        v.frame,
        v.n_theta,
        v.n_phi,
        v.dtheta,
        v.dphi,
        v.lat_boundary_rad,
        v.omega_corot,
        dth_dev,
        dph_dev,
        v.pole_grad_N,
        v.pole_grad_S,
    )
end

"""
    ionosphere_field_view(solver::IonosphereSolver) -> IonosphereFieldView

Create an [`IonosphereFieldView`](@ref) referencing `solver`'s solved gradient arrays.
"""
function ionosphere_field_view(solver::IonosphereSolver{FT}) where {FT}
    return IonosphereFieldView{FT, Matrix{FT}}(
        solver.frame,
        solver.n_theta,
        solver.n_phi,
        solver.dtheta,
        solver.dphi,
        solver.lat_boundary_rad,
        solver.config.omega_corot,
        solver.dphi_dtheta,
        solver.dphi_dphi,
        solver.pole_grad_N,
        solver.pole_grad_S,
    )
end

@inline function _eval_convection_E_smg(
        v::IonosphereFieldView{FT}, x_smg::SVector{3, FT},
    ) where {FT <: AbstractFloat}
    r2 = dot(x_smg, x_smg)
    r = sqrt(r2)
    r <= FT(1.0e-12) && return zero(SVector{3, FT})

    rho2 = x_smg[1] * x_smg[1] + x_smg[2] * x_smg[2]
    rho = sqrt(rho2)
    if rho < FT(1.0e-8) * r
        ratio = v.frame.r_iono / r
        scale = -(ratio * sqrt(ratio))
        pg = x_smg[3] >= zero(FT) ? v.pole_grad_N : v.pole_grad_S
        return SVector{3, FT}(scale * pg[1], scale * pg[2], zero(FT))
    end

    i_hem, _, theta, phi, jac_smg = map_dipole_smg_with_jacobian(x_smg, v.frame.r_iono)
    i_hem == 0 && return zero(SVector{3, FT})
    mag_lat = abs(FT(0.5 * pi) - theta)
    mag_lat <= v.lat_boundary_rad && return zero(SVector{3, FT})

    t_norm = theta / v.dtheta + one(FT)
    p_norm = mod(phi, FT(2 * pi)) / v.dphi
    i0 = clamp(floor(Int, t_norm), 1, v.n_theta - 1)
    i1 = i0 + 1
    wt = clamp(t_norm - FT(i0), zero(FT), one(FT))

    j0_raw = floor(Int, p_norm)
    j0 = mod(j0_raw, v.n_phi) + 1
    j1 = j0 == v.n_phi ? 1 : j0 + 1
    wp = p_norm - FT(j0_raw)

    @inbounds begin
        dt00 = v.dphi_dtheta[i0, j0]
        dt01 = v.dphi_dtheta[i0, j1]
        dt10 = v.dphi_dtheta[i1, j0]
        dt11 = v.dphi_dtheta[i1, j1]
        dp00 = v.dphi_dphi[i0, j0]
        dp01 = v.dphi_dphi[i0, j1]
        dp10 = v.dphi_dphi[i1, j0]
        dp11 = v.dphi_dphi[i1, j1]
    end
    dPdT = (one(FT) - wt) * ((one(FT) - wp) * dt00 + wp * dt01) +
        wt * ((one(FT) - wp) * dt10 + wp * dt11)
    dPdP = (one(FT) - wt) * ((one(FT) - wp) * dp00 + wp * dp01) +
        wt * ((one(FT) - wp) * dp10 + wp * dp11)

    return SVector{3, FT}(
        -(dPdT * jac_smg[1, 1] + dPdP * jac_smg[2, 1]),
        -(dPdT * jac_smg[1, 2] + dPdP * jac_smg[2, 2]),
        -(dPdT * jac_smg[1, 3] + dPdP * jac_smg[2, 3]),
    )
end

"""
    eval_ionosphere_E(
        view::IonosphereFieldView{FT},
        x_sim::SVector{3, FT};
        B0::Union{Nothing, SVector{3, FT}} = nothing,
    ) -> SVector{3, FT}

Evaluate the mapped 3D electric field ``\\mathbf{E} = -\\nabla\\Phi + \\mathbf{E}_{\\text{corot}}``
at simulation position `x_sim`, satisfying ``\\mathbf{E}\\cdot\\mathbf{B}_0 \\equiv 0``
via the exact dipole Jacobian (with smooth ``m=1`` polar-axis regularity at ``\\rho = 0``).
When `B0` is supplied and `view.omega_corot` is non-zero, adds the exact 3D planetary
corotation electric field ``-(\\boldsymbol{\\Omega}_{\\text{corot}} \\times \\Delta\\mathbf{r}) \\times \\mathbf{B}_0``.
"""
@inline function eval_ionosphere_E(
        v::IonosphereFieldView{FT},
        x_sim::SVector{3, FT};
        B0::Union{Nothing, SVector{3, FT}} = nothing,
    ) where {FT <: AbstractFloat}
    dr = x_sim - v.frame.center
    x_smg = v.frame.rot * dr
    e_smg = _eval_convection_E_smg(v, x_smg)
    e_sim = transpose(v.frame.rot) * e_smg
    if B0 !== nothing && !iszero(v.omega_corot)
        u_cor = cross(v.omega_corot, dr)
        e_sim -= cross(u_cor, B0)
    end
    return e_sim
end

@inline eval_ionosphere_E(
    solver::IonosphereSolver{FT}, x_sim::SVector{3, FT};
    B0::Union{Nothing, SVector{3, FT}} = nothing,
) where {FT} = eval_ionosphere_E(ionosphere_field_view(solver), x_sim; B0 = B0)

"""
    eval_ionosphere_u_exb(
        view::IonosphereFieldView{FT},
        x_sim::SVector{3, FT},
        B0::SVector{3, FT};
        u_max::FT = FT(Inf),
    ) -> SVector{3, FT}

Evaluate the mapped magnetospheric convection velocity
``\\mathbf{u} = (\\mathbf{E}_{\\text{conv}} \\times \\mathbf{B}_0) / |\\mathbf{B}_0|^2 + \\boldsymbol{\\Omega}_{\\text{corot}} \\times \\Delta\\mathbf{r}``
at simulation position `x_sim`, optionally capped at `|u| <= u_max`.
"""
@inline function eval_ionosphere_u_exb(
        v::IonosphereFieldView{FT},
        x_sim::SVector{3, FT},
        B0::SVector{3, FT};
        u_max::FT = FT(Inf),
    ) where {FT <: AbstractFloat}
    dr = x_sim - v.frame.center
    x_smg = v.frame.rot * dr
    e_smg = _eval_convection_E_smg(v, x_smg)
    e_conv = transpose(v.frame.rot) * e_smg
    b2 = dot(B0, B0)
    u_conv = b2 > FT(1.0e-30) ? (cross(e_conv, B0) / b2) : zero(SVector{3, FT})
    u = u_conv + cross(v.omega_corot, dr)
    um = norm(u)
    if um > u_max
        u = u * (u_max / um)
    end
    return u
end

@inline eval_ionosphere_u_exb(
    solver::IonosphereSolver{FT}, x_sim::SVector{3, FT}, B0::SVector{3, FT};
    u_max::Real = solver.config.u_max,
) where {FT} = eval_ionosphere_u_exb(ionosphere_field_view(solver), x_sim, B0; u_max = FT(u_max))
