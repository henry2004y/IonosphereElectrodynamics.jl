# Getting Started

## Installation

Install `IonosphereElectrodynamics.jl` from the Julia package manager:

```julia
using Pkg
Pkg.add("IonosphereElectrodynamics")
```

For local development against a checkout:

```julia
using Pkg
Pkg.develop(path = "/path/to/IonosphereElectrodynamics.jl")
```

## Coordinate & Sign Conventions

`IonosphereElectrodynamics.jl` operates in an orthonormal **Solar Magnetic (SMG)** dipole frame
[`DipoleFrame`](@ref) constructed from any non-zero 3D `dipole_moment` and optional
offset `center`:

- **North / South SMG axis (`e3`)**: Always oriented toward the simulation northern
  hemisphere (`e3[3] >= 0`). `frame.dipole_sign` records the polarity along `e3`:
  `-1` for Earth-like polarity (dipole moment pointing southward, `m ⋅ e3 < 0`),
  `+1` for reversed polarity.
- **Spherical grid**: Colatitude $\theta \in [0, \pi]$ spans `n_theta = 2 * n_theta_hem - 1`
  points from the north magnetic pole ($\theta = 0$) through the magnetic equator
  ($\theta = \pi/2$) to the south magnetic pole ($\theta = \pi$). Longitude
  $\phi \in [0, 2\pi)$ spans `n_phi` uniform points starting at $\phi = 0$ along `e1`.
- **Outward radial FAC (`jr_outward`)**: Positive values denote current flowing radially
  outward from the ionosphere (upward FAC, electron precipitation region); negative values
  denote downward FAC.

## Walkthrough

The example below configures a Moen–Brekke solar EUV + auroral conductance model, solves
both hemispheres for an analytic Region 1 + Region 2 field-aligned current pattern, and
evaluates the mapped 3D electric field and $\mathbf{E} \times \mathbf{B}_0$ drift:

````@example getting_started
using IonosphereElectrodynamics
using LinearAlgebra: dot, norm
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

converged = solve_ionosphere!(sol, jr)
cpcp = cross_polar_cap_potential(sol)

view = ionosphere_field_view(sol)
x = SVector(0.0, 0.2 * r_planet, 3.0 * r_planet)
b0 = dipole_B0(sol.frame, x)
e_vec = eval_ionosphere_E(view, x; B0 = b0)
u_exb = eval_ionosphere_u_exb(view, x, b0)

println("converged = ", converged)
println("CPCP_N    = ", round(cpcp.north * 1.0e-3; sigdigits = 4), " kV")
println("CPCP_S    = ", round(cpcp.south * 1.0e-3; sigdigits = 4), " kV")
println("E         = ", round.(e_vec .* 1.0e3; sigdigits = 4), " mV/m")
println("|E⋅B0| / (|E||B0|) = ", round(abs(dot(e_vec, b0)) / (norm(e_vec) * norm(b0)); sigdigits = 2))
println("u_exb     = ", round.(u_exb; sigdigits = 4), " m/s")
````

## Diagnostics & Solver Reset

Use [`write_ionosphere_diagnostics`](@ref) to export the 2D grid
(`theta_deg phi_deg Jr SigmaP SigmaH Phi Eth Eph`) to a plain ASCII file, and
[`reset_ionosphere!`](@ref) to clear stored potentials, FAC buffers, and iteration
diagnostics back to zero without reallocating.

