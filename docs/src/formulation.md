# Formulation & Physics

## Height-Integrated Ionospheric Ohm's Law

`IonosphereElectrodynamics.jl` solves the steady-state current continuity equation on the spherical
ionospheric thin shell of radius $R_I$ in each hemisphere independently:

```math
\nabla_h \cdot \bigl(\boldsymbol{\Sigma} \cdot \nabla_h \Phi\bigr) = J_r^{\text{outward}}
```

where $\Phi(\theta, \phi)$ is the electrostatic potential, $J_r^{\text{outward}}$ is the
radial outward field-aligned current density at $r = R_I$, and $\boldsymbol{\Sigma}$ is the
$2 \times 2$ height-integrated spherical conductance tensor in $(\theta, \phi)$ coordinates.

In expanded spherical coordinates, the elliptic operator acting on $\Phi$ is

```math
\frac{1}{R_I^2 \sin^2\theta}\left[
    \sin^2\theta\,\Sigma_{\theta\theta}\,\frac{\partial^2 \Phi}{\partial\theta^2}
    + \left(\sin\theta\cos\theta\,\Sigma_{\theta\theta}
        + \sin^2\theta\,\frac{\partial\Sigma_{\theta\theta}}{\partial\theta}
        - \sin\theta\,\frac{\partial\Sigma_{\theta\phi}}{\partial\phi}\right)\frac{\partial\Phi}{\partial\theta}
    + \Sigma_{\phi\phi}\,\frac{\partial^2 \Phi}{\partial\phi^2}
    + \left(\sin\theta\,\frac{\partial\Sigma_{\theta\phi}}{\partial\theta}
        + \frac{\partial\Sigma_{\phi\phi}}{\partial\phi}\right)\frac{\partial\Phi}{\partial\phi}
\right] = J_r^{\text{outward}}.
```

## Conductance & Mobility Models

All conductance models subtype [`AbstractConductanceModel`](@ref) and are evaluated via
[`evaluate_conductance`](@ref):

- [`UniformConductance`](@ref): Spatially constant $(\Sigma_P, \Sigma_H)$.
- [`SolarEUVConductance`](@ref): Dayside cosine zenith-angle model combining nightside floors
  in quadrature with $\Sigma_{\text{day}} \max(0, \hat{\mathbf{r}} \cdot \hat{\mathbf{s}})^\alpha$.
- [`MoenBrekkeEUVConductance`](@ref): Empirical Moen & Brekke (1993) parameterization driven
  by the $F_{10.7}$ radio flux index:
  ```math
  \Sigma_{P,\text{EUV}} = F_{10.7}^{0.49}\bigl(0.34\cos\chi + 0.93\sqrt{\cos\chi}\bigr), \quad
  \Sigma_{H,\text{EUV}} = F_{10.7}^{0.53}\bigl(0.81\cos\chi + 0.54\sqrt{\cos\chi}\bigr).
  ```
- [`AuroralConductance`](@ref): Wrapper around any base conductance model adding dynamic
  upward-FAC ($J_{\text{up}} = \max(0, J_r^{\text{outward}})$) precipitation in quadrature
  using either `:tanh_sqrt` or `:saturating` profiles.
- [`CustomConductance`](@ref): User-supplied callback `fn(theta, phi, t) -> (sigma_P, sigma_H)`.

For 3D anisotropic plasma closures, [`electron_mobility_tensor`](@ref) constructs the
$3 \times 3$ `SMatrix` tensor $\sigma_0 \hat{\mathbf{b}}\hat{\mathbf{b}}^\mathsf{T} + \sigma_P (\mathbf{I} - \hat{\mathbf{b}}\hat{\mathbf{b}}^\mathsf{T}) + \sigma_H (\hat{\mathbf{b}} \times)$:

````@example conductance_demo
using IonosphereElectrodynamics
using StaticArrays

unif = UniformConductance(5.0, 2.0)
mb = MoenBrekkeEUVConductance(; f107 = 150.0, sun_dir = (1.0, 0.0, 0.0))
aur = AuroralConductance(mb; sigma_P_max = 15.0, hall_ratio = 1.5, fac_scale = 1.0e-6)

r_day = SVector(1.0, 0.0, 0.0)
r_night = SVector(-1.0, 0.0, 0.0)
sp_u, sh_u = evaluate_conductance(unif, 0.5, 0.0, r_day, 0.0)
sp_day, sh_day = evaluate_conductance(mb, 0.5 * pi, 0.0, r_day, 0.0)
sp_night, sh_night = evaluate_conductance(aur, 0.5 * pi, pi, r_night, 0.0; jr_outward = 1.0e-6)

M = electron_mobility_tensor(2.0, 3.0, SVector(0.0, 0.0, 1.0); sigma_0 = 5.0)
s_pole = dipole_conductance_tensor(sp_day, sh_day, cos(0.2), sin(0.2), -1.0)
s_eq = dipole_conductance_tensor(sp_day, sh_day, cos(0.49 * pi), sin(0.49 * pi), -1.0)

println("Uniform:     (ΣP, ΣH) = ", (sp_u, sh_u))
println("Dayside EUV: (ΣP, ΣH) = ", round.((sp_day, sh_day); sigdigits = 4))
println("Night+FAC:   (ΣP, ΣH) = ", round.((sp_night, sh_night); sigdigits = 4))
println("Mobility M * z_hat    = ", M * SVector(0.0, 0.0, 1.0))
println("Dipole tensor (polar) = ", round.(s_pole; sigdigits = 4))
println("Dipole tensor (eq)    = ", round.(s_eq; sigdigits = 4))
````

## Dipole Spherical Conductance Tensor

Following Goodman (1995) and Ridley et al. (2004), [`dipole_conductance_tensor`](@ref)
maps local Pedersen and Hall conductances $(\Sigma_P, \Sigma_H)$ onto the spherical
ionospheric shell with finite parallel conductance $\Sigma_0$:

```math
C = 4\Sigma_0\cos^2\theta + \Sigma_P\sin^2\theta,
```
```math
\Sigma_{\theta\theta} = \frac{\Sigma_0 \Sigma_P (1 + 3\cos^2\theta)}{C}, \quad
\Sigma_{\theta\phi} = -\text{dipole\_sign}\cdot\frac{2\Sigma_0 \Sigma_H \cos\theta \sqrt{1 + 3\cos^2\theta}}{C}, \quad
\Sigma_{\phi\phi} = \Sigma_P + \frac{\Sigma_H^2 \sin^2\theta}{C}.
```

This captures equatorial Cowling enhancement in $\Sigma_{\phi\phi}$ near $\theta = \pi/2$
and automatically flips the Hall cross-term sign when dipole polarity reverses
(`dipole_sign = +1` vs `-1`). Setting `use_dip_angle = false` simplifies to
$(\Sigma_P, -\text{dipole\_sign}\cdot \operatorname{sgn}(\cos\theta)\Sigma_H, \Sigma_P)$.

## Analytic Dipole Mapping, Exact Jacobian & Polar Regularity

Along an axisymmetric dipole field line $r = L \sin^2\theta$ in SMG coordinates:

- [`map_dipole_smg`](@ref) maps a 3D SMG point $\mathbf{x}_{\text{smg}}$ to radius
  $r_{\text{map}}$, returning hemisphere index `i_hem ∈ {+1, -1, 0}` (`0` when the field
  line closes below $r_{\text{map}}$), mapped Cartesian coordinates, and $(\theta_m, \phi_m)$.
- [`map_dipole_smg_with_jacobian`](@ref) additionally computes the exact closed-form
  $2 \times 3$ Jacobian $\mathbf{J}_{\text{smg}} = \partial(\theta_m, \phi_m)/\partial(x, y, z)$.

Evaluating the mapped 3D convection electric field via

```math
\mathbf{E}_{\text{conv}}(\mathbf{x}_{\text{smg}}) = -\mathbf{J}_{\text{smg}}^\mathsf{T}
\begin{bmatrix} \partial\Phi/\partial\theta \\ \partial\Phi/\partial\phi \end{bmatrix}
```

guarantees $\mathbf{E}_{\text{conv}} \cdot \mathbf{B}_0 \equiv 0$ to machine precision
because both $\theta_m$ and $\phi_m$ are constant along dipole field lines.

Near the magnetic poles ($\rho = \sqrt{x_{\text{smg}}^2 + y_{\text{smg}}^2} < 10^{-8} r$),
where spherical coordinates have a coordinate singularity, the solver projects
$\partial\Phi/\partial\theta$ at the pole onto the $m = 1$ Fourier mode
(`pole_grad_N`, `pole_grad_S`) and evaluates $\mathbf{E}_{\text{conv}}$ smoothly with
dipole radial scaling $(R_I / r)^{3/2}$.

When `omega_corot` is non-zero in [`IonosphereConfig`](@ref), [`eval_ionosphere_E`](@ref)
adds $-(\boldsymbol{\Omega}_{\text{corot}} \times \Delta\mathbf{r}) \times \mathbf{B}_0$ and
[`eval_ionosphere_u_exb`](@ref) adds $\boldsymbol{\Omega}_{\text{corot}} \times \Delta\mathbf{r}$
directly in 3D, avoiding spherical finite-difference spikes near the low-latitude boundary.

````@example dipole_mapping_demo
using IonosphereElectrodynamics
using StaticArrays

frame = build_dipole_frame((0.0, 0.0, -3.0e-5); r_iono = 1.017)
x_sim = SVector(1.5, 0.5, 2.0)
b0 = dipole_B0(frame, x_sim)
x_smg = frame.rot * (x_sim - frame.center)
i_hem, x_map, th_map, ph_map, jac = map_dipole_smg_with_jacobian(x_smg, frame.r_iono)

println("i_hem      = ", i_hem, " (", i_hem == 1 ? "North" : i_hem == -1 ? "South" : "Closed", ")")
println("footpoint  = (θ = ", round(rad2deg(th_map); sigdigits = 4), "°, φ = ", round(rad2deg(ph_map); sigdigits = 4), "°)")
println("x_map      = ", round.(x_map; sigdigits = 4))
println("B0         = ", round.(b0; sigdigits = 4))
println("size(jac)  = ", size(jac), ", ∂θ/∂z = ", round(jac[1, 3]; sigdigits = 4))
````

## Numerical Method & Zero-Allocation Solver

Each hemisphere's active grid rows (`lat_boundary_deg` to the pole) are discretized with
second-order central 5-point finite differences:

- **Polar boundary**: Average of the adjacent colatitude ring ($i = 2$ for North, $n_\theta - 1$ for South).
- **Azimuthal boundary**: Periodic in $\phi \in [0, 2\pi)$.
- **Low-latitude boundary**: Either homogeneous Dirichlet (`low_lat_bc = :dirichlet`, $\Phi = 0$)
  or homogeneous Neumann (`low_lat_bc = :neumann`, $\partial\Phi/\partial\theta = 0$ with a
  single pinned gauge node and active-row FAC neutralization).

The linear system is reordered along anti-diagonal wavefronts (`i + j = d + 1`) so both
the incomplete LU factorization `ILU(0)` and forward/backward triangular sweeps vectorize
with `@simd ivdep` inside preconditioned BiCGSTAB. All workspaces are pre-allocated inside
[`IonosphereSolver`](@ref), giving `0 B` heap allocations during steady-state calls to
[`solve_ionosphere!`](@ref).
