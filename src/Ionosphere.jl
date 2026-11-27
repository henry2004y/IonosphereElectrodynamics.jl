"""
    Ionosphere

Standalone, zero-allocation 2D spherical magnetosphere–ionosphere (M–I) coupling solver.
Solves the height-integrated ionospheric potential equation
``\\nabla_h \\cdot (\\boldsymbol{\\Sigma} \\cdot \\nabla_h \\Phi) = J_r^{\\text{outward}}``
on both hemispheres in dipole SMG coordinates and maps potentials, electric fields, and
``\\mathbf{E}\\times\\mathbf{B}_0`` convection velocities along dipole field lines.
"""
module Ionosphere

using Adapt: Adapt
using LinearAlgebra: cross, dot, norm
using PrecompileTools: @compile_workload, @setup_workload
using StaticArrays: SMatrix, SVector

include("conductance.jl")
include("dipole.jl")
include("solver.jl")
include("mapping.jl")

export AbstractConductanceModel,
    UniformConductance,
    SolarEUVConductance,
    MoenBrekkeEUVConductance,
    AuroralConductance,
    CustomConductance,
    evaluate_conductance,
    electron_mobility_tensor,
    DipoleFrame,
    build_dipole_frame,
    dipole_B0,
    map_dipole_smg,
    map_dipole_smg_with_jacobian,
    dipole_conductance_tensor,
    dipole_fac_sample_point,
    IonosphereConfig,
    IonosphereSolver,
    solve_ionosphere!,
    reset_ionosphere!,
    cross_polar_cap_potential,
    write_ionosphere_diagnostics,
    IonosphereFieldView,
    ionosphere_field_view,
    eval_ionosphere_E,
    eval_ionosphere_u_exb

@setup_workload begin
    @compile_workload begin
        cfg = IonosphereConfig(
            ;
            n_theta_hem = 9,
            n_phi = 16,
            conductance = AuroralConductance(MoenBrekkeEUVConductance()),
            tol = 1.0e-8,
        )
        sol = IonosphereSolver(cfg; r_iono = 1.0, dipole_moment = (0.0, 0.0, -1.0))
        jr = [sin(sol.theta[i]) * sin(sol.phi[j]) for i in 1:sol.n_theta, j in 1:sol.n_phi]
        _ = solve_ionosphere!(sol, jr)
        _ = cross_polar_cap_potential(sol)
        x = SVector(0.0, 0.2, 2.0)
        b0 = dipole_B0(sol.frame, x)
        _ = eval_ionosphere_E(sol, x; B0 = b0)
        _ = eval_ionosphere_u_exb(sol, x, b0)
        _ = electron_mobility_tensor(1.0, 0.5, SVector(0.0, 0.0, 1.0))
    end
end

end # module Ionosphere
