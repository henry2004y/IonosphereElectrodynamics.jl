# Ionosphere.jl

Standalone, zero-allocation 2D spherical magnetosphere–ionosphere (M–I) electrostatic coupling solver for space plasma simulations.

## Public API

- **Conductance & mobility models**: `AbstractConductanceModel`, `UniformConductance`, `SolarEUVConductance`, `MoenBrekkeEUVConductance`, `AuroralConductance`, `CustomConductance`, `evaluate_conductance`, `electron_mobility_tensor`.
- **Dipole frame & mapping**: `DipoleFrame`, `build_dipole_frame`, `dipole_B0`, `map_dipole_smg`, `map_dipole_smg_with_jacobian`, `dipole_conductance_tensor`, `dipole_fac_sample_point`.
- **Solver & configuration**: `IonosphereConfig`, `IonosphereSolver`, `solve_ionosphere!`, `reset_ionosphere!`, `cross_polar_cap_potential`, `write_ionosphere_diagnostics`.
- **3D field/velocity evaluation (`Adapt`-friendly)**: `IonosphereFieldView`, `ionosphere_field_view`, `eval_ionosphere_E`, `eval_ionosphere_u_exb`.

## Examples

- `examples/fleks_analytic_polar.jl`: reproduces the 6-panel FLEKS analytic Region 1 + Region 2 Birkeland current benchmark with Moen–Brekke EUV + Auroral conductance (`CPCP ≈ 128.2 kV`, requires `CairoMakie`).
