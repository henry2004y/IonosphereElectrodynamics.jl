```@raw html
---
layout: home

hero:
  name: "IonosphereElectrodynamics.jl"
  text: "2D Spherical M–I Coupling Solver"
  tagline: "Zero-allocation ionospheric Ohm's law solver, conductance models, and analytic dipole field-line mapping for space plasma simulations."
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: Formulation & Physics
      link: /formulation
    - theme: alt
      text: API Reference
      link: /api

features:
  - title: Conductance & Mobility Models
    details: Uniform, cosine solar-EUV, Moen–Brekke F10.7 EUV, upward-FAC auroral precipitation (tanh-sqrt and saturating), custom callbacks, and 3D anisotropic electron mobility tensors.
  - title: Analytic Dipole Mapping & Jacobian
    details: Arbitrary-tilt, offset DipoleFrame in Solar Magnetic (SMG) coordinates with closed-form field-line mapping, exact 2×3 Jacobian ensuring E ⋅ B0 ≡ 0, and Goodman/Ridley Cowling conductance tensors.
  - title: Zero-Allocation Hemispheric Solver
    details: Pre-allocated 5-point spherical finite-difference solver with wavefront-reordered SIMD ILU(0)-BiCGSTAB, Dirichlet or Neumann low-latitude BCs, FAC smoothing/neutralization, and zero steady-state heap allocations.
  - title: GPU / Adapt-Friendly 3D Field Views
    details: Lightweight isbits-compatible IonosphereFieldView for evaluating mapped electric fields and E × B convection velocities on CPU or GPU with smooth m = 1 polar-axis regularity and corotation.
---
```

# Overview

**IonosphereElectrodynamics.jl** is a standalone, pure-Julia solver for two-dimensional spherical
magnetosphere–ionosphere (M–I) electrostatic coupling. It solves the height-integrated
ionospheric Ohm's law on both hemispheres in dipole Solar Magnetic (SMG) coordinates:

```math
\nabla_h \cdot \bigl(\boldsymbol{\Sigma} \cdot \nabla_h \Phi\bigr) = J_r^{\text{outward}}
```

and maps electric potentials, electric fields ($\mathbf{E} \cdot \mathbf{B}_0 \equiv 0$),
and $\mathbf{E} \times \mathbf{B}_0$ convection velocities along analytic dipole field
lines between the ionospheric shell and magnetospheric boundaries.

## Quick Example

```julia
using IonosphereElectrodynamics
using StaticArrays

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

th_r1, th_r2, sig = deg2rad(18.0), deg2rad(24.0), deg2rad(2.5)
jr = [
    let th_hem = sol.theta[i] <= 0.5 * pi ? sol.theta[i] : pi - sol.theta[i]
        r1 = exp(-0.5 * ((th_hem - th_r1) / sig)^2)
        r2 = exp(-0.5 * ((th_hem - th_r2) / sig)^2)
        (1.2e-6 * r1 - 0.6e-6 * r2) * sin(sol.phi[j])
    end for i in 1:sol.n_theta, j in 1:sol.n_phi
]

solve_ionosphere!(sol, jr)
cpcp = cross_polar_cap_potential(sol)

view = ionosphere_field_view(sol)
x = SVector(0.0, 0.2 * r_planet, 3.0 * r_planet)
b0 = dipole_B0(sol.frame, x)
e_vec = eval_ionosphere_E(view, x; B0 = b0)
u_exb = eval_ionosphere_u_exb(view, x, b0)
```

