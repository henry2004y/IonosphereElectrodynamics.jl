# IonosphereElectrodynamics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://henry2004y.github.io/IonosphereElectrodynamics.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://henry2004y.github.io/IonosphereElectrodynamics.jl/dev)
[![CI](https://github.com/henry2004y/IonosphereElectrodynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/henry2004y/IonosphereElectrodynamics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/henry2004y/IonosphereElectrodynamics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/henry2004y/IonosphereElectrodynamics.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone, zero-allocation 2D spherical magnetosphere–ionosphere (M–I) electrostatic coupling solver for space plasma simulations.

Solves the height-integrated ionospheric Ohm's law on both hemispheres in dipole Solar Magnetic (SMG) coordinates:

$$\nabla_h \cdot (\boldsymbol{\Sigma} \cdot \nabla_h \Phi) = J_r^{\text{outward}}$$

using a 5-point spherical finite-difference stencil and pre-allocated ILU(0)-preconditioned BiCGSTAB with zero steady-state heap allocations, then maps potentials, electric fields ($\mathbf{E} \cdot \mathbf{B}_0 \equiv 0$), and $\mathbf{E} \times \mathbf{B}_0$ convection velocities along analytic dipole field lines.

## Key Features

- **Conductance & Mobility Models**: `UniformConductance`, `SolarEUVConductance`, `MoenBrekkeEUVConductance` ($F_{10.7}$-driven solar EUV), `AuroralConductance` (dynamic upward-FAC precipitation wrapper), `CustomConductance` callbacks, and anisotropic 3D `electron_mobility_tensor`.
- **Dipole SMG Frame & Analytic Mapping**: `DipoleFrame` (`build_dipole_frame`) with arbitrary dipole tilt and offset center, analytic `dipole_B0`, closed-form field-line mapping and exact $2 \times 3$ Jacobian (`map_dipole_smg`, `map_dipole_smg_with_jacobian`), equatorial Cowling enhancement (`dipole_conductance_tensor`), and FAC shell mapping (`dipole_fac_sample_point`).
- **Zero-Allocation Hemispheric Solver**: Pre-allocated `IonosphereSolver` (`solve_ionosphere!`, `reset_ionosphere!`) supporting Dirichlet and Neumann low-latitude boundary conditions, FAC neutralization/smoothing, planetary corotation (`omega_corot`), cross-polar-cap potential (`cross_polar_cap_potential`), and FLEKS-compatible ASCII diagnostics (`write_ionosphere_diagnostics`).
- **GPU/`Adapt`-Friendly 3D Field Views**: Lightweight `IonosphereFieldView` (`ionosphere_field_view`) supporting `eval_ionosphere_E` and `eval_ionosphere_u_exb` with smooth $m = 1$ polar-axis regularity at $\rho = 0$ and optional corotation electric fields.

## Installation

```julia
using Pkg
Pkg.add("IonosphereElectrodynamics")
```

Or for local development:

```julia
using Pkg
Pkg.develop(path = "/path/to/IonosphereElectrodynamics.jl")
```

## Quick Start

```julia
using IonosphereElectrodynamics
using StaticArrays

# 1. Configure Moen-Brekke solar EUV + auroral conductance on a 2D spherical grid
cond = AuroralConductance(
    MoenBrekkeEUVConductance(; f107 = 150.0, sun_dir = (1.0, 0.0, 0.0));
    sigma_P_max = 15.0,
    hall_ratio = 1.5,
    fac_scale = 1.0e-6,
)
cfg = IonosphereConfig(;
    n_theta_hem = 33,
    n_phi = 64,
    lat_boundary_deg = 20.0,
    conductance = cond,
    tol = 1.0e-8,
)

r_planet = 6.371e6
sol = IonosphereSolver(
    cfg;
    r_iono = 1.017 * r_planet,
    dipole_moment = (0.0, 0.0, -3.0e-5 * r_planet^3),
)

# 2. Prescribe an analytic Region 1 + Region 2 field-aligned current pattern
th_r1, th_r2, sig = deg2rad(18.0), deg2rad(24.0), deg2rad(2.5)
jr = [
    let th_hem = sol.theta[i] <= 0.5 * pi ? sol.theta[i] : pi - sol.theta[i]
        r1 = exp(-0.5 * ((th_hem - th_r1) / sig)^2)
        r2 = exp(-0.5 * ((th_hem - th_r2) / sig)^2)
        (1.2e-6 * r1 - 0.6e-6 * r2) * sin(sol.phi[j])
    end for i in 1:sol.n_theta, j in 1:sol.n_phi
]

# 3. Solve both hemispheres and inspect cross-polar-cap potential
converged = solve_ionosphere!(sol, jr)
cpcp = cross_polar_cap_potential(sol)

# 4. Evaluate mapped 3D electric field and E x B velocity at an inner boundary point
view = ionosphere_field_view(sol)
x = SVector(0.0, 0.2 * r_planet, 3.0 * r_planet)
b0 = dipole_B0(sol.frame, x)
e_vec = eval_ionosphere_E(view, x; B0 = b0)
u_exb = eval_ionosphere_u_exb(view, x, b0)
```

## Examples

- [`examples/fleks_analytic_polar.jl`](examples/fleks_analytic_polar.jl): Reproduces the 6-panel FLEKS analytic Region 1 + Region 2 Birkeland current benchmark with Moen–Brekke EUV + Auroral conductance (`CPCP ≈ 128.2 kV`, requires `CairoMakie`):

```bash
julia --project=/path/to/env-with-CairoMakie examples/fleks_analytic_polar.jl
```

## Public API

- **Conductance & mobility models**: `AbstractConductanceModel`, `UniformConductance`, `SolarEUVConductance`, `MoenBrekkeEUVConductance`, `AuroralConductance`, `CustomConductance`, `evaluate_conductance`, `electron_mobility_tensor`.
- **Dipole frame & mapping**: `DipoleFrame`, `build_dipole_frame`, `dipole_B0`, `map_dipole_smg`, `map_dipole_smg_with_jacobian`, `dipole_conductance_tensor`, `dipole_fac_sample_point`.
- **Solver & configuration**: `IonosphereConfig`, `IonosphereSolver`, `solve_ionosphere!`, `reset_ionosphere!`, `cross_polar_cap_potential`, `write_ionosphere_diagnostics`.
- **3D field/velocity evaluation (`Adapt`-friendly)**: `IonosphereFieldView`, `ionosphere_field_view`, `eval_ionosphere_E`, `eval_ionosphere_u_exb`.

## Testing & Benchmarking

Run the unit, convergence, and `Aqua.jl` / `ExplicitImports.jl` quality test suite:

```bash
julia --project=IonosphereElectrodynamics.jl -e 'using Pkg; Pkg.test()'
```

Benchmarks are defined in [`benchmark/benchmarks.jl`](benchmark/benchmarks.jl) using `BenchmarkTools.jl` and tracked across revisions with [`AirspeedVelocity.jl`](https://github.com/MilesCranmer/AirspeedVelocity.jl):

```bash
# Compare the current working tree against main using the AirspeedVelocity CLI
cd IonosphereElectrodynamics.jl && benchpkg IonosphereElectrodynamics --rev=main,dirty
```

Or from Julia:

```julia
using AirspeedVelocity

AirspeedVelocity.benchmark("IonosphereElectrodynamics", ["main", "dirty"]; path = "/path/to/IonosphereElectrodynamics.jl")
```

## License

`IonosphereElectrodynamics.jl` is licensed under the [MIT License](LICENSE).
