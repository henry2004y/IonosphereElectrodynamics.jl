# ==============================================================================
# GPU/KA-Safe 3D Field View & Analytic Dipole Potential/Convection Evaluation
# ==============================================================================

"""
    IonosphereFieldView{FT <: Real, Mat <: AbstractMatrix{FT}}

Lightweight, `Adapt.adapt`-friendly view of solved ionospheric potential gradients,
pole gradients, dipole frame, and planetary corotation angular velocity for 3D CPU/GPU
boundary evaluation kernels.
"""
struct IonosphereFieldView{FT <: Real, Mat <: AbstractMatrix{FT}}
    frame::DipoleFrame{FT}
    n_theta::Int
    n_phi::Int
    dtheta::FT
    dphi::FT
    inv_dtheta::FT
    inv_dphi::FT
    lat_boundary_rad::FT
    omega_corot::SVector{3, FT}
    dphi_dtheta::Mat
    dphi_dphi::Mat
    pole_grad_N::SVector{2, FT}
    pole_grad_S::SVector{2, FT}
end

IonosphereFieldView{FT, Mat}(
    frame::DipoleFrame{FT},
    n_theta::Int,
    n_phi::Int,
    dtheta::FT,
    dphi::FT,
    lat_boundary_rad::FT,
    omega_corot::SVector{3, FT},
    dphi_dtheta::Mat,
    dphi_dphi::Mat,
    pole_grad_N::SVector{2, FT},
    pole_grad_S::SVector{2, FT},
) where {FT <: Real, Mat <: AbstractMatrix{FT}} = IonosphereFieldView{FT, Mat}(
    frame, n_theta, n_phi, dtheta, dphi, inv(dtheta), inv(dphi),
    lat_boundary_rad, omega_corot, dphi_dtheta, dphi_dphi, pole_grad_N, pole_grad_S,
)

function Adapt.adapt_structure(to, v::IonosphereFieldView{FT}) where {FT}
    dth_dev = Adapt.adapt(to, v.dphi_dtheta)
    dph_dev = Adapt.adapt(to, v.dphi_dphi)
    return IonosphereFieldView{FT, typeof(dth_dev)}(
        v.frame,
        v.n_theta,
        v.n_phi,
        v.dtheta,
        v.dphi,
        v.inv_dtheta,
        v.inv_dphi,
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
    ionosphere_field_view(solver::IonosphereSolver, phi_pot::AbstractMatrix{<:Real}) -> IonosphereFieldView

Create an [`IonosphereFieldView`](@ref) referencing `solver`'s solved gradient arrays, or
purely compute gradients from an explicit potential matrix `phi_pot` (e.g., returned by
[`solve_ionosphere`](@ref)) without mutating `solver`.
"""
function ionosphere_field_view(solver::IonosphereSolver{FT}) where {FT}
    return IonosphereFieldView{FT, Matrix{FT}}(
        solver.frame,
        solver.n_theta,
        solver.n_phi,
        solver.dtheta,
        solver.dphi,
        solver.inv_dtheta,
        solver.inv_dphi,
        solver.lat_boundary_rad,
        solver.config.omega_corot,
        solver.dphi_dtheta,
        solver.dphi_dphi,
        solver.pole_grad_N,
        solver.pole_grad_S,
    )
end

function ionosphere_field_view(
        solver::IonosphereSolver{FT}, phi_pot::AbstractMatrix{<:Real},
    ) where {FT <: Real}
    size(phi_pot) == (solver.n_theta, solver.n_phi) ||
        throw(DimensionMismatch("phi_pot size $(size(phi_pot)) != expected $((solver.n_theta, solver.n_phi))"))
    dth, dph, pg_N, pg_S = _compute_gradients_outofplace(solver, phi_pot)
    T = eltype(dth)
    return IonosphereFieldView{T, Matrix{T}}(
        _convert_frame(T, solver.frame),
        solver.n_theta,
        solver.n_phi,
        T(solver.dtheta),
        T(solver.dphi),
        T(solver.inv_dtheta),
        T(solver.inv_dphi),
        T(solver.lat_boundary_rad),
        SVector{3, T}(solver.config.omega_corot),
        dth,
        dph,
        pg_N,
        pg_S,
    )
end

@inline @muladd function _eval_convection_E_smg(
        v::IonosphereFieldView{FTv}, x_smg::SVector{3, FTx},
    ) where {FTv <: Real, FTx <: Real}
    T = float(promote_type(FTv, FTx))
    sT = _scalar_float_type(T)
    x = T(x_smg[1])
    y = T(x_smg[2])
    z = T(x_smg[3])
    rho2 = x * x + y * y
    r2 = rho2 + z * z
    eps_r2 = T(max(floatmin(sT), sT(1.0e-24)))
    r2 <= eps_r2 && return zero(SVector{3, T})

    eps_rho2 = T(max(floatmin(sT), sT(1.0e-16)))
    if rho2 < eps_rho2 * r2
        r = sqrt(r2)
        ratio = T(v.frame.r_iono) / r
        scale = -(ratio * sqrt(ratio))
        pg = z >= zero(T) ? v.pole_grad_N : v.pole_grad_S
        return SVector{3, T}(scale * T(pg[1]), scale * T(pg[2]), zero(T))
    end

    i_hem, _, theta, phi, jac_smg = _map_dipole_smg_core(
        SVector{3, T}(x, y, z), T(v.frame.r_iono), Val(true),
    )
    i_hem == 0 && return zero(SVector{3, T})
    mag_lat = abs(T(0.5) * _pi(T) - theta)
    mag_lat <= T(v.lat_boundary_rad) && return zero(SVector{3, T})

    t_norm = theta * T(v.inv_dtheta) + one(T)
    p_norm = phi * T(v.inv_dphi)
    i0 = clamp(_floor_int(t_norm), 1, v.n_theta - 1)
    i1 = i0 + 1
    wt = clamp(t_norm - T(i0), zero(T), one(T))

    j0_raw = _floor_int(p_norm)
    j0 = mod(j0_raw, v.n_phi) + 1
    j1 = j0 == v.n_phi ? 1 : j0 + 1
    wp = p_norm - T(j0_raw)

    @inbounds begin
        dt00 = T(v.dphi_dtheta[i0, j0])
        dt01 = T(v.dphi_dtheta[i0, j1])
        dt10 = T(v.dphi_dtheta[i1, j0])
        dt11 = T(v.dphi_dtheta[i1, j1])
        dp00 = T(v.dphi_dphi[i0, j0])
        dp01 = T(v.dphi_dphi[i0, j1])
        dp10 = T(v.dphi_dphi[i1, j0])
        dp11 = T(v.dphi_dphi[i1, j1])
    end
    dt0 = dt00 + wp * (dt01 - dt00)
    dt1 = dt10 + wp * (dt11 - dt10)
    dPdT = dt0 + wt * (dt1 - dt0)
    dp0 = dp00 + wp * (dp01 - dp00)
    dp1 = dp10 + wp * (dp11 - dp10)
    dPdP = dp0 + wt * (dp1 - dp0)

    return SVector{3, T}(
        -(dPdT * jac_smg[1, 1] + dPdP * jac_smg[2, 1]),
        -(dPdT * jac_smg[1, 2] + dPdP * jac_smg[2, 2]),
        -(dPdT * jac_smg[1, 3] + dPdP * jac_smg[2, 3]),
    )
end

"""
    eval_ionosphere_E(
        view::IonosphereFieldView,
        x_sim;
        B0 = nothing,
    ) -> SVector{3}

Evaluate the mapped 3D electric field ``\\mathbf{E} = -\\nabla\\Phi + \\mathbf{E}_{\\text{corot}}``
at simulation position `x_sim` (`SVector{3}`, `AbstractVector`, or `NTuple{3}`), satisfying
``\\mathbf{E}\\cdot\\mathbf{B}_0 \\equiv 0`` via the exact dipole Jacobian (with smooth
``m=1`` polar-axis regularity at ``\\rho = 0``). When `B0` is supplied and
`view.omega_corot` is non-zero, adds the exact 3D planetary corotation electric field
``-(\\boldsymbol{\\Omega}_{\\text{corot}} \\times \\Delta\\mathbf{r}) \\times \\mathbf{B}_0``.
"""
@inline @muladd function eval_ionosphere_E(
        v::IonosphereFieldView{FTv},
        x_sim::_Vec3Input;
        B0::Union{Nothing, _Vec3Input} = nothing,
    ) where {FTv <: Real}
    xs = _as_svec3(x_sim)
    dr = xs - v.frame.center
    x_smg = v.frame.rot * dr
    e_smg = _eval_convection_E_smg(v, x_smg)
    e_sim = transpose(v.frame.rot) * e_smg
    if B0 === nothing
        return e_sim
    end
    b0 = _as_svec3(B0)
    T = float(promote_type(eltype(e_sim), eltype(b0)))
    e_out = SVector{3, T}(e_sim)
    if !iszero(v.omega_corot)
        u_cor = cross(SVector{3, T}(v.omega_corot), SVector{3, T}(dr))
        e_out -= cross(u_cor, SVector{3, T}(b0))
    end
    return e_out
end

@inline eval_ionosphere_E(
    solver::IonosphereSolver, x_sim::_Vec3Input;
    B0::Union{Nothing, _Vec3Input} = nothing,
) = eval_ionosphere_E(ionosphere_field_view(solver), x_sim; B0 = B0)

"""
    eval_ionosphere_u_exb(
        view::IonosphereFieldView,
        x_sim,
        B0;
        u_max = Inf,
    ) -> SVector{3}

Evaluate the mapped magnetospheric convection velocity
``\\mathbf{u} = (\\mathbf{E}_{\\text{conv}} \\times \\mathbf{B}_0) / |\\mathbf{B}_0|^2 + \\boldsymbol{\\Omega}_{\\text{corot}} \\times \\Delta\\mathbf{r}``
at simulation position `x_sim`, optionally capped at `|u| <= u_max`.
"""
@inline @muladd function eval_ionosphere_u_exb(
        v::IonosphereFieldView{FTv},
        x_sim::_Vec3Input,
        B0::_Vec3Input;
        u_max::Real = _inf(FTv),
    ) where {FTv <: Real}
    xs = _as_svec3(x_sim)
    b0_raw = _as_svec3(B0)
    T = float(promote_type(FTv, eltype(xs), eltype(b0_raw), typeof(u_max)))
    dr = SVector{3, T}(
        xs[1] - v.frame.center[1],
        xs[2] - v.frame.center[2],
        xs[3] - v.frame.center[3],
    )
    rot = SMatrix{3, 3, T, 9}(v.frame.rot)
    x_smg = rot * dr
    e_smg = _eval_convection_E_smg(v, x_smg)
    e_conv = transpose(rot) * SVector{3, T}(e_smg)
    b0 = SVector{3, T}(b0_raw)
    b2 = b0[1] * b0[1] + b0[2] * b0[2] + b0[3] * b0[3]
    u_conv = b2 > _tiny(T) ? (cross(e_conv, b0) / b2) : zero(SVector{3, T})
    u = u_conv + cross(SVector{3, T}(v.omega_corot), dr)
    u_max_t = T(u_max)
    u2 = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
    if isfinite(u_max_t) && u2 > u_max_t * u_max_t
        u = u * (u_max_t / sqrt(u2))
    end
    return u
end

@inline eval_ionosphere_u_exb(
    solver::IonosphereSolver, x_sim::_Vec3Input, B0::_Vec3Input;
    u_max::Real = solver.config.u_max,
) = eval_ionosphere_u_exb(ionosphere_field_view(solver), x_sim, B0; u_max = u_max)
