module IonosphereElectrodynamicsChainRulesCoreExt

using ChainRulesCore:
    AbstractZero, ChainRulesCore, NoTangent, ProjectTo, Tangent, ZeroTangent, unthunk
using IonosphereElectrodynamics:
    IonosphereSolver,
    _compute_gradients_vjp,
    _conductance_depends_on_jr,
    _copy_op,
    _is_static_conductance,
    _solve_ionosphere_jvp,
    _solve_ionosphere_stateless,
    _solve_ionosphere_vjp,
    ionosphere_field_view,
    solve_ionosphere

function _snapshot_forward(
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real},
        t::FT,
    ) where {FT <: AbstractFloat}
    phi_out = _solve_ionosphere_stateless(solver, jr_in, t)
    if _is_static_conductance(solver.config.conductance)
        return (phi_out, solver.op_N, solver.op_S, nothing, nothing, nothing)
    end
    op_N_snap = _copy_op(solver.op_N)
    op_S_snap = _copy_op(solver.op_S)
    x_N_snap = copy(solver.ws_N.x)
    x_S_snap = copy(solver.ws_S.x)
    jr_s_snap = _conductance_depends_on_jr(solver.config.conductance) ?
        copy(solver.jr_outward) : nothing
    return (phi_out, op_N_snap, op_S_snap, x_N_snap, x_S_snap, jr_s_snap)
end

function ChainRulesCore.rrule(
        ::typeof(solve_ionosphere),
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real};
        t::Real = zero(FT),
    ) where {FT <: AbstractFloat}
    t_ft = FT(t)
    phi_out, op_N_snap, op_S_snap, x_N_snap, x_S_snap, jr_s_snap =
        _snapshot_forward(solver, jr_in, t_ft)
    proj_jr = ProjectTo(jr_in)

    function solve_ionosphere_pullback(dphi_raw)
        dphi = unthunk(dphi_raw)
        if dphi isa AbstractZero
            return (NoTangent(), NoTangent(), ZeroTangent())
        end
        bar_jr = _solve_ionosphere_vjp(
            solver, dphi, t_ft;
            op_N = op_N_snap,
            op_S = op_S_snap,
            x_N = x_N_snap,
            x_S = x_S_snap,
            jr_s = jr_s_snap,
        )
        return (NoTangent(), NoTangent(), proj_jr(bar_jr))
    end
    return phi_out, solve_ionosphere_pullback
end

function ChainRulesCore.frule(
        (_, _, djr_raw),
        ::typeof(solve_ionosphere),
        solver::IonosphereSolver{FT},
        jr_in::AbstractMatrix{<:Real};
        t::Real = zero(FT),
    ) where {FT <: AbstractFloat}
    t_ft = FT(t)
    phi_out, op_N_snap, op_S_snap, x_N_snap, x_S_snap, jr_s_snap =
        _snapshot_forward(solver, jr_in, t_ft)
    djr = unthunk(djr_raw)
    if djr isa AbstractZero
        return phi_out, ZeroTangent()
    end
    dphi = _solve_ionosphere_jvp(
        solver, djr, t_ft;
        op_N = op_N_snap,
        op_S = op_S_snap,
        x_N = x_N_snap,
        x_S = x_S_snap,
        jr_s = jr_s_snap,
    )
    return phi_out, dphi
end

@inline _tangent_field(t::AbstractZero, ::Symbol) = nothing
@inline function _tangent_field(t::Tangent, s::Symbol)
    v = unthunk(getproperty(t, s))
    return v isa AbstractZero ? nothing : v
end
@inline function _tangent_field(t::NamedTuple, s::Symbol)
    haskey(t, s) || return nothing
    v = unthunk(getproperty(t, s))
    return v isa AbstractZero ? nothing : v
end

function ChainRulesCore.rrule(
        ::typeof(ionosphere_field_view),
        solver::IonosphereSolver{FT},
        phi_pot::AbstractMatrix{<:Real},
    ) where {FT <: AbstractFloat}
    v = ionosphere_field_view(solver, phi_pot)
    proj_phi = ProjectTo(phi_pot)

    function ionosphere_field_view_pullback(dv_raw)
        dv = unthunk(dv_raw)
        if dv isa AbstractZero
            return (NoTangent(), NoTangent(), ZeroTangent())
        end
        bar_dth = _tangent_field(dv, :dphi_dtheta)
        bar_dph = _tangent_field(dv, :dphi_dphi)
        bar_pg_N = _tangent_field(dv, :pole_grad_N)
        bar_pg_S = _tangent_field(dv, :pole_grad_S)
        if bar_dth === nothing && bar_dph === nothing &&
                bar_pg_N === nothing && bar_pg_S === nothing
            return (NoTangent(), NoTangent(), ZeroTangent())
        end
        bar_phi = _compute_gradients_vjp(solver, bar_dth, bar_dph, bar_pg_N, bar_pg_S)
        return (NoTangent(), NoTangent(), proj_phi(bar_phi))
    end
    return v, ionosphere_field_view_pullback
end

end # module IonosphereElectrodynamicsChainRulesCoreExt
