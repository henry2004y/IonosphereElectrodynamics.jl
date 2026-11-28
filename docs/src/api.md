# API Reference

## Package Overview

```@docs
IonosphereElectrodynamics
```

## Conductance & Mobility Models

```@docs
AbstractConductanceModel
UniformConductance
SolarEUVConductance
MoenBrekkeEUVConductance
AuroralConductance
CustomConductance
evaluate_conductance
electron_mobility_tensor
```

## Dipole Frame & Field-Line Mapping

```@docs
DipoleFrame
build_dipole_frame
dipole_B0
map_dipole_smg
map_dipole_smg_with_jacobian
dipole_conductance_tensor
dipole_fac_sample_point
```

## Solver, Configuration & Diagnostics

```@docs
IonosphereConfig
IonosphereSolver
solve_ionosphere!
solve_ionosphere
reset_ionosphere!
cross_polar_cap_potential
write_ionosphere_diagnostics
```

## 3D Field & Convection Evaluation

```@docs
IonosphereFieldView
ionosphere_field_view
eval_ionosphere_E
eval_ionosphere_u_exb
```

