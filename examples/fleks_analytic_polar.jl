# ==============================================================================
# FLEKS-Equivalent Standalone Polar Ionosphere Benchmark (6-Panel Comparison)
# ==============================================================================
#
# Reproduces the FLEKS analytic Region 1 + Region 2 Birkeland current benchmark
# with Moen–Brekke Solar EUV + Auroral conductance (`CPCP ≈ 128.2 kV`) in GSM
# coordinates (`sun_dir = (1, 0, 0)`, Earth-like dipole `m_z < 0`):
#   - Top:    12 MLT (Noon, +x)
#   - Bottom: 00 MLT (Midnight, -x)
#   - Right:  06 MLT (Dawn, -y, Φ > 0, downward R1 FAC)
#   - Left:   18 MLT (Dusk, +y, Φ < 0, upward R1 FAC)
#
# Usage:
#   julia --project=<env-with-CairoMakie-and-LingXiaoIonosphere> \
#       examples/fleks_analytic_polar.jl [output_path.png]

using CairoMakie
using LingXiaoIonosphere
using Printf
using StaticArrays

function run_fleks_analytic_polar(;
        outpath::AbstractString = isempty(ARGS) ? joinpath(@__DIR__, "fleks_analytic_polar.png") : ARGS[1],
    )
    r_planet = 6.371e6
    cond = AuroralConductance(
        MoenBrekkeEUVConductance(;
            f107 = 150.0,
            sigma_P_night = 3.0,
            sigma_H_night = 3.0,
            sun_dir = (1.0, 0.0, 0.0),
        );
        sigma_P_max = 15.0,
        hall_ratio = 1.5,
        fac_scale = 1.0e-6,
        mode = :tanh_sqrt,
    )
    cfg = IonosphereConfig(;
        n_theta_hem = 91,
        n_phi = 180,
        lat_boundary_deg = 20.0,
        low_lat_bc = :dirichlet,
        conductance = cond,
        use_dip_angle = true,
        sigma_parallel = 1000.0,
        sigma_P_floor = 0.1,
        neutralize_fac = false,
        n_smooth_fac = 1,
        tol = 1.0e-8,
        max_iter = 1000,
    )
    sol = IonosphereSolver(
        cfg;
        r_iono = 1.017 * r_planet,
        dipole_moment = (0.0, 0.0, -3.0e-5 * r_planet^3),
    )

    th_r1 = deg2rad(18.0)
    th_r2 = deg2rad(24.0)
    sig = deg2rad(2.5)
    jr = zeros(sol.n_theta, sol.n_phi)
    for i in 1:(sol.n_theta)
        th = sol.theta[i]
        th_hem = th <= 0.5 * pi ? th : (pi - th)
        r1 = exp(-0.5 * ((th_hem - th_r1) / sig)^2)
        r2 = exp(-0.5 * ((th_hem - th_r2) / sig)^2)
        for j in 1:(sol.n_phi)
            jr[i, j] = (1.2e-6 * r1 - 0.6e-6 * r2) * sin(sol.phi[j])
        end
    end

    ok = solve_ionosphere!(sol, jr)
    @assert ok "Ionosphere solver did not converge"
    cpcp = cross_polar_cap_potential(sol)
    @printf("CPCP_N = %.2f kV, CPCP_S = %.2f kV\n", cpcp.north * 1.0e-3, cpcp.south * 1.0e-3)

    # Polar dial helper (MLAT >= 50 deg, radial distance from pole up to 40 deg)
    th_dial_max = deg2rad(40.0)
    np = sol.n_phi
    phis = vcat([(j - 1) * sol.dphi for j in 1:np], [2 * pi])

    idx_N = findall(t -> t <= th_dial_max + 1.0e-12, sol.theta[1:(sol.n_theta_hem - 1)])
    idx_S = sort(
        findall(i -> (pi - sol.theta[i]) <= th_dial_max + 1.0e-12, (sol.n_theta_hem + 1):(sol.n_theta)) .+
            sol.n_theta_hem;
        by = i -> pi - sol.theta[i],
    )

    function extract_dial_grid(inds, is_north::Bool)
        nr = length(inds)
        X = zeros(nr, np + 1)
        Y = zeros(nr, np + 1)
        Jr_uA = zeros(nr, np + 1)
        Phi_kV = zeros(nr, np + 1)
        SigP = zeros(nr, np + 1)
        SigH = zeros(nr, np + 1)
        for j in 1:(np + 1)
            js = j <= np ? j : 1
            ph = phis[j]
            # GSM: phi = 0 is +x (Noon, Top +Y), phi = pi/2 is +y (Dusk, Left -X), phi = 3pi/2 is -y (Dawn, Right +X)
            sp_dial = -sin(ph)
            cp_dial = cos(ph)
            for (k, i) in enumerate(inds)
                a = is_north ? sol.theta[i] : (pi - sol.theta[i])
                rr = a / th_dial_max
                X[k, j] = rr * sp_dial
                Y[k, j] = rr * cp_dial
                Jr_uA[k, j] = sol.jr_outward[i, js] * 1.0e6
                Phi_kV[k, j] = sol.phi_solved[i, js] * 1.0e-3
                SigP[k, j] = sol.sigma_P[i, js]
                SigH[k, j] = sol.sigma_H[i, js]
            end
        end
        return X, Y, Jr_uA, Phi_kV, SigP, SigH
    end

    XN, YN, JrN, PhiN, SigPN, _ = extract_dial_grid(idx_N, true)
    XS, YS, JrS, PhiS, _, SigHS = extract_dial_grid(idx_S, false)

    view_f = ionosphere_field_view(sol)
    r_eval = sol.frame.r_iono
    function exb_arrows(is_north::Bool)
        qx = Float64[]; qy = Float64[]; qu = Float64[]; qv = Float64[]
        for th_deg in (8.0, 15.0, 22.0, 29.0, 36.0)
            rr = deg2rad(th_deg) / th_dial_max
            th = is_north ? deg2rad(th_deg) : (pi - deg2rad(th_deg))
            for k in 0:11
                ph = k * (2 * pi / 12)
                x_sim = SVector{3, Float64}(
                    r_eval * sin(th) * cos(ph),
                    r_eval * sin(th) * sin(ph),
                    r_eval * cos(th),
                )
                b0 = dipole_B0(sol.frame, x_sim)
                u = eval_ionosphere_u_exb(view_f, x_sim, b0; u_max = 1.0e7)
                push!(qx, -rr * sin(ph))
                push!(qy, rr * cos(ph))
                push!(qu, -u[2])
                push!(qv, u[1])
            end
        end
        u_ref = max(maximum(hypot.(qu, qv)), 1.0e-12)
        qu .*= 0.13 / u_ref
        qv .*= 0.13 / u_ref
        return qx, qy, qu, qv
    end

    function add_guides!(ax)
        th_ring = range(0, 2 * pi; length = 240)
        for mlat in (80, 70, 60, 50)
            frac = (90 - mlat) / 40.0
            lines!(
                ax, frac .* cos.(th_ring), frac .* sin.(th_ring);
                color = mlat == 50 ? :black : (:gray35, 0.55),
                linestyle = mlat == 50 ? :solid : :dot,
                linewidth = mlat == 50 ? 1.4 : 0.9,
            )
            if mlat in (80, 70, 60, 50)
                ang_lbl = deg2rad(-40.0)
                text!(
                    ax, frac * cos(ang_lbl) + 0.02, frac * sin(ang_lbl) - 0.02;
                    text = "$(mlat)°", fontsize = 11, color = :gray25,
                )
            end
        end
        lines!(ax, [-1.0, 1.0], [0.0, 0.0]; color = (:gray45, 0.45), linestyle = :dot, linewidth = 0.9)
        lines!(ax, [0.0, 0.0], [-1.0, 1.0]; color = (:gray45, 0.45), linestyle = :dot, linewidth = 0.9)
        text!(ax, 0.0, 1.05; text = "12 MLT\n(Noon)", align = (:center, :bottom), fontsize = 12)
        text!(ax, 0.0, -1.05; text = "00 MLT\n(Mid)", align = (:center, :top), fontsize = 12)
        text!(ax, 1.05, 0.0; text = "06 MLT\n(Dawn)", align = (:left, :center), fontsize = 12)
        text!(ax, -1.05, 0.0; text = "18 MLT\n(Dusk)", align = (:right, :center), fontsize = 12)
        xlims!(ax, -1.45, 1.45)
        ylims!(ax, -1.30, 1.30)
        hidedecorations!(ax)
        return hidespines!(ax)
    end

    fig = Figure(; size = (1650, 1020), fontsize = 14)
    Label(
        fig[0, 1:6],
        "LingXiaoIonosphere.jl: Analytic R1+R2 FACs + Auroral/EUV Conductance (FLEKS Benchmark)";
        fontsize = 18, font = :bold,
    )

    j_lim = max(maximum(abs, JrN), maximum(abs, JrS), 1.0)
    p_lim = max(maximum(abs, PhiN), maximum(abs, PhiS), 1.0)
    sp_hi = maximum(SigPN)
    sh_hi = maximum(SigHS)

    # Row 1: North
    ax1 = Axis(fig[1, 1]; title = "North: Outward FAC J_r (+up / -down)", aspect = DataAspect())
    hm1 = surface!(ax1, XN, YN, zeros(size(XN)); color = JrN, colormap = Reverse(:RdBu), colorrange = (-j_lim, j_lim), shading = NoShading)
    add_guides!(ax1)
    Colorbar(fig[1, 2], hm1; label = L"$J_r$ [$\mu$A/m$^2$]")

    ax2 = Axis(fig[1, 3]; title = @sprintf("North: Potential Φ & E × B (CPCP=%.1f kV)", cpcp.north * 1.0e-3), aspect = DataAspect())
    hm2 = surface!(ax2, XN, YN, zeros(size(XN)); color = PhiN, colormap = Reverse(:RdBu), colorrange = (-p_lim, p_lim), shading = NoShading)
    qxN, qyN, quN, qvN = exb_arrows(true)
    arrows!(ax2, qxN, qyN, quN, qvN; color = (:black, 0.8), arrowsize = 8, linewidth = 1.2)
    add_guides!(ax2)
    Colorbar(fig[1, 4], hm2; label = L"$\Phi$ [kV]")

    ax3 = Axis(fig[1, 5]; title = "North: Pedersen Conductance Σ_P", aspect = DataAspect())
    hm3 = surface!(ax3, XN, YN, zeros(size(XN)); color = SigPN, colormap = :viridis, colorrange = (0.0, sp_hi), shading = NoShading)
    add_guides!(ax3)
    Colorbar(fig[1, 6], hm3; label = L"$\Sigma_P$ [S]")

    # Row 2: South
    ax4 = Axis(fig[2, 1]; title = "South: Outward FAC J_r (+up / -down)", aspect = DataAspect())
    hm4 = surface!(ax4, XS, YS, zeros(size(XS)); color = JrS, colormap = Reverse(:RdBu), colorrange = (-j_lim, j_lim), shading = NoShading)
    add_guides!(ax4)
    Colorbar(fig[2, 2], hm4; label = L"$J_r$ [$\mu$A/m$^2$]")

    ax5 = Axis(fig[2, 3]; title = @sprintf("South: Potential Φ & E × B (CPCP=%.1f kV)", cpcp.south * 1.0e-3), aspect = DataAspect())
    hm5 = surface!(ax5, XS, YS, zeros(size(XS)); color = PhiS, colormap = Reverse(:RdBu), colorrange = (-p_lim, p_lim), shading = NoShading)
    qxS, qyS, quS, qvS = exb_arrows(false)
    arrows!(ax5, qxS, qyS, quS, qvS; color = (:black, 0.8), arrowsize = 8, linewidth = 1.2)
    add_guides!(ax5)
    Colorbar(fig[2, 4], hm5; label = L"$\Phi$ [kV]")

    ax6 = Axis(fig[2, 5]; title = "South: Hall Conductance Σ_H", aspect = DataAspect())
    hm6 = surface!(ax6, XS, YS, zeros(size(XS)); color = SigHS, colormap = :plasma, colorrange = (0.0, sh_hi), shading = NoShading)
    add_guides!(ax6)
    Colorbar(fig[2, 6], hm6; label = L"$\Sigma_H$ [S]")

    mkpath(dirname(outpath))
    save(outpath, fig; px_per_unit = 2)
    println("Saved ", outpath)
    return outpath
end

run_fleks_analytic_polar()
