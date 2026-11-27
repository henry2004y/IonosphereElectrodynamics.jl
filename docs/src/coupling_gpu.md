# Coupling & GPU Usage

## Two-Way Magnetosphere–Ionosphere Coupling

In a coupled 3D magnetospheric simulation with an inner boundary at $r_{\text{ib}}$ and a
field-aligned current extraction shell at $r_{\text{fac}} \ge r_{\text{ib}} \ge R_I$:

1. **Extract FACs on the ionosphere grid**: For each active `(i, j)` node on `sol.theta`
   and `sol.phi`, call [`dipole_fac_sample_point`](@ref) to trace along the dipole field
   line from $R_I$ to $r_{\text{fac}}$, sample $\mathbf{J}_\parallel = \mathbf{J} \cdot \hat{\mathbf{b}}$
   from the 3D magnetospheric state at `x_fac`, and project to the ionospheric shell using
   flux conservation:
   ```math
   J_r^{\text{outward}} = J_\parallel \frac{|\mathbf{B}_{0,\text{iono}}|}{|\mathbf{B}_{\text{tot}}(\mathbf{x}_{\text{fac}})|}
   \frac{\mathbf{B}_{0,\text{iono}} \cdot \hat{\mathbf{r}}_{\text{iono}}}{|\mathbf{B}_{0,\text{iono}}|}.
   ```
2. **Solve the 2D potential equation**: Call [`solve_ionosphere!`](@ref) with optional
   Laplacian FAC smoothing (`n_smooth_fac`), hemispheric neutralization
   (`neutralize_fac`), and under-relaxation (`relax_alpha`).
3. **Evaluate mapped fields at $r \le r_{\text{ib}}$**: Construct a lightweight
   [`IonosphereFieldView`](@ref) via [`ionosphere_field_view`](@ref) and query
   [`eval_ionosphere_E`](@ref) and [`eval_ionosphere_u_exb`](@ref) inside inner-boundary
   or guard-cell kernels.

````@example fac_mapping_demo
using IonosphereElectrodynamics

frame = build_dipole_frame((0.0, 0.0, -3.0e-5); r_iono = 1.017)
r_fac = 2.5

for th_deg in (15.0, 25.0, 65.0)
    valid, x_fac, _, _, b_iono, br_over_b = dipole_fac_sample_point(
        frame, deg2rad(th_deg), deg2rad(90.0), r_fac,
    )
    if valid
        println(
            "θ = ", th_deg, "° -> x_fac = ", round.(x_fac; sigdigits = 4),
            ", |B0_iono| = ", round(b_iono; sigdigits = 4),
            ", Br/B = ", round(br_over_b; sigdigits = 4),
        )
    else
        println("θ = ", th_deg, "° -> closed below r_fac = ", r_fac)
    end
end
````

## GPU / `Adapt.jl` & `Float32` Support

[`IonosphereFieldView`](@ref) implements `Adapt.adapt_structure`, allowing a host solver's
gradient arrays to be transferred cleanly to GPU device memory (e.g., CUDA/AMDGPU/Metal via
`Adapt.adapt(backend, ionosphere_field_view(sol))`) while keeping [`DipoleFrame`](@ref) and
scalar metadata `isbits`. Both `eval_ionosphere_E` and `eval_ionosphere_u_exb` are `@inline`,
type-stable, and allocation-free for `Float64` and `Float32`:

````@example float32_demo
using IonosphereElectrodynamics
using StaticArrays

cfg32 = IonosphereConfig(;
    n_theta_hem = 17,
    n_phi = 32,
    conductance = AuroralConductance(SolarEUVConductance(; FT = Float32); FT = Float32),
    FT = Float32,
)
sol32 = IonosphereSolver(cfg32; r_iono = 1.0f0, dipole_moment = (0.0f0, 0.0f0, -3.0f-5))
jr32 = [1.0f-6 * sin(sol32.theta[i])^2 * sin(sol32.phi[j]) for i in 1:sol32.n_theta, j in 1:sol32.n_phi]

ok = solve_ionosphere!(sol32, jr32)
cpcp32 = cross_polar_cap_potential(sol32)
view32 = ionosphere_field_view(sol32)
x32 = SVector(0.5f0, 0.4f0, 1.5f0)
b32 = dipole_B0(sol32.frame, x32)
e32 = eval_ionosphere_E(view32, x32; B0 = b32)
u32 = eval_ionosphere_u_exb(view32, x32, b32)

println("ok    = ", ok)
println("cpcp  = (north = ", round(cpcp32.north; sigdigits = 4), ", south = ", round(cpcp32.south; sigdigits = 4), ")")
println("e32   = ", round.(e32; sigdigits = 4), " (", typeof(e32), ")")
println("u32   = ", round.(u32; sigdigits = 4), " (", typeof(u32), ")")
````

## Analytic R1 + R2 Polar Benchmark

The standalone script [`examples/fleks_analytic_polar.jl`](https://github.com/henry2004y/IonosphereElectrodynamics.jl/blob/main/examples/fleks_analytic_polar.jl)
reproduces the FLEKS analytic Region 1 + Region 2 Birkeland current benchmark on a
`91 × 180` spherical grid (`lat_boundary_deg = 20°`) with Moen–Brekke solar EUV
($F_{10.7} = 150$, sun along `+x`) and `:tanh_sqrt` auroral precipitation conductance,
yielding a cross-polar-cap potential of $\text{CPCP} \approx 128.2\,\text{kV}$ in both
hemispheres, dawn-positive / dusk-negative potential cells shifted by Hall conductance, and
antisunward $\mathbf{E} \times \mathbf{B}_0$ convection across both poles:

![Analytic R1+R2 Polar Ionosphere Benchmark](assets/fleks_analytic_polar.png)

To run the example locally and save the figure:

```bash
julia --project=/path/to/env-with-CairoMakie examples/fleks_analytic_polar.jl
```

## Performance Benchmarks

The [`benchmark/benchmarks.jl`](https://github.com/henry2004y/IonosphereElectrodynamics.jl/blob/main/benchmark/benchmarks.jl)
suite tracks conductance evaluation, dipole mapping, 3D field/velocity queries, and warm/cold
2D ILU(0)-BiCGSTAB solves using `BenchmarkTools.jl` and `AirspeedVelocity.jl`:

```bash
cd IonosphereElectrodynamics.jl && benchpkg IonosphereElectrodynamics --rev=main,dirty
```
