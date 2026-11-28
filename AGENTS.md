See [README.md](./README.md) for the package overview, [docs/src/api.md](./docs/src/api.md) for the public API, [docs/src/getting_started.md](./docs/src/getting_started.md) for coordinate/sign conventions and walkthroughs, and [docs/src/formulation.md](./docs/src/formulation.md) for equations and numerical details.

# Code, invariants, and commits
- Format Julia code with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (checked by [`.github/workflows/Format.yml`](./.github/workflows/Format.yml)), keeping lines within 92 characters.
- Avoid `Any` in function signatures, use an explicit `;` before keyword arguments in calls, and use explicit imports (`using Pkg: sym`) — `ExplicitImports.jl` checks for no implicit or stale explicit imports.
- Preserve `0 B` steady-state heap allocations and `@inferred` type stability for both `Float64` and `Float32` on solver and field-evaluation hot paths. Keep dipole frames and 3D field views `isbits`/`Adapt`-safe, and keep non-mutated fields of mutable solver structs marked `const`.
- Every exported symbol must have a docstring (checked in [`test/runtests.jl`](./test/runtests.jl)) and appear in [`docs/src/api.md`](./docs/src/api.md) (`checkdocs = :exports`, `warnonly = false` in [`docs/make.jl`](./docs/make.jl)). Keep the `@compile_workload` in [`src/IonosphereElectrodynamics.jl`](./src/IonosphereElectrodynamics.jl) in sync with API changes. For internal code, add comments or docstrings only when behavior is not obvious from names and types.
- Commit titles follow `component: brief summary` (e.g., `solver:`, `docs:`, `test:`, `perf:`, `fix:`) with a brief prose body when needed; link external issues/PRs as `owner/repo#123`.
- Respect manual user edits to code you wrote; explain concerns and ask before reintroducing modified or deleted code.

# Dependencies and environments
- Never edit `Manifest.toml` manually. Manage dependencies with `Pkg.add` and `Pkg.rm`; edit [`Project.toml`](./Project.toml) directly only for `[compat]` (keeping `Aqua.test_all` checks passing; minimum supported Julia version is `1.11`).
- Tests run via root `[targets]` in [`Project.toml`](./Project.toml); [`docs/Project.toml`](./docs/Project.toml) and [`benchmark/Project.toml`](./benchmark/Project.toml) wire the root package via `[sources]`.

# Testing, docs, and benchmarks
- Full test suite (`Aqua.jl`, `ExplicitImports.jl`, docstring coverage, MMS convergence, FLEKS R1+R2 benchmark, `Float32`, `@inferred`, and `0 B` allocation checks):
  `julia --project=. -e 'using Pkg; Pkg.test()'`
- Wrap warmup + `@allocated` checks in helper functions (as in [`test/runtests.jl`](./test/runtests.jl)) to avoid top-level/`@testset` boxing on Julia 1.11.
- Build docs locally (also runs [`examples/fleks_analytic_polar.jl`](./examples/fleks_analytic_polar.jl) and all `@example` blocks):
  `julia --project=docs -e 'using Pkg; Pkg.instantiate()'` then `julia --project=docs docs/make.jl` (set `BUILD_VITEPRESS=false` to skip the VitePress/Node step when only checking Documenter/`@example` execution).
- Run benchmarks: `julia --project=benchmark -e 'using Pkg; Pkg.instantiate(); using BenchmarkTools; include("benchmark/benchmarks.jl"); show(stdout, MIME("text/plain"), run(SUITE))'` (or a subgroup like `SUITE["solver"]`), or compare revisions with `benchpkg IonosphereElectrodynamics --rev=main,dirty` (`AirspeedVelocity.jl`).
- Run the standalone polar plot example: `julia --project=docs examples/fleks_analytic_polar.jl`.
