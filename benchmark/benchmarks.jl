using BenchmarkTools
using IonosphereElectrodynamics
using StaticArrays

const SUITE = BenchmarkGroup()

let
    # 1. Conductance & mobility models
    m_unif = UniformConductance(5.0, 10.0)
    m_euv = SolarEUVConductance(;
        sigma_P_day = 8.0,
        sigma_H_day = 12.0,
        sigma_P_night = 2.0,
        sigma_H_night = 1.0,
        sun_dir = (1.0, 0.0, 0.0),
    )
    m_mb = MoenBrekkeEUVConductance(;
        f107 = 150.0,
        sigma_P_night = 3.0,
        sigma_H_night = 3.0,
        sun_dir = (1.0, 0.0, 0.0),
    )
    m_aur = AuroralConductance(m_mb; sigma_P_max = 15.0, hall_ratio = 1.5, fac_scale = 1.0e-6)

    th = 0.35
    ph = 1.2
    sin_th = sin(th)
    cos_th = cos(th)
    r_hat = SVector{3, Float64}(sin_th * cos(ph), sin_th * sin(ph), cos_th)
    b_hat = SVector{3, Float64}(0.2, -0.3, sqrt(1.0 - 0.13))

    SUITE["conductance"] = BenchmarkGroup(["conductance"])
    SUITE["conductance"]["uniform"] = @benchmarkable evaluate_conductance(
        $m_unif, $th, $ph, $r_hat, 0.0,
    )
    SUITE["conductance"]["solar_euv"] = @benchmarkable evaluate_conductance(
        $m_euv, $th, $ph, $r_hat, 0.0,
    )
    SUITE["conductance"]["moen_brekke_euv"] = @benchmarkable evaluate_conductance(
        $m_mb, $th, $ph, $r_hat, 0.0,
    )
    SUITE["conductance"]["auroral_moen_brekke"] = @benchmarkable evaluate_conductance(
        $m_aur, $th, $ph, $r_hat, 0.0; jr_outward = 1.2e-6,
    )
    SUITE["conductance"]["electron_mobility_tensor"] = @benchmarkable electron_mobility_tensor(
        5.0, 10.0, $b_hat; sigma_0 = 1000.0,
    )

    # 2. Dipole frame & field-line mapping
    r_iono = 1.017
    r_fac = 2.5
    frame = build_dipole_frame((0.0, 0.0, -3.0e-5); r_iono = r_iono)
    x_sim = SVector{3, Float64}(1.2, 0.8, 2.0)
    x_smg = frame.rot * (x_sim - frame.center)

    SUITE["dipole"] = BenchmarkGroup(["dipole"])
    SUITE["dipole"]["dipole_B0"] = @benchmarkable dipole_B0($frame, $x_sim)
    SUITE["dipole"]["map_dipole_smg"] = @benchmarkable map_dipole_smg($x_smg, $r_iono)
    SUITE["dipole"]["map_dipole_smg_with_jacobian"] = @benchmarkable map_dipole_smg_with_jacobian(
        $x_smg, $r_iono,
    )
    SUITE["dipole"]["dipole_conductance_tensor"] = @benchmarkable dipole_conductance_tensor(
        5.0, 10.0, $cos_th, $sin_th, -1.0,
    )
    SUITE["dipole"]["dipole_fac_sample_point"] = @benchmarkable dipole_fac_sample_point(
        $frame, $th, $ph, $r_fac,
    )

    # 3. 3D field & ExB evaluation via IonosphereFieldView
    cfg_33 = IonosphereConfig(;
        n_theta_hem = 33,
        n_phi = 64,
        conductance = m_unif,
        tol = 1.0e-8,
        max_iter = 500,
    )
    sol_view = IonosphereSolver(cfg_33; r_iono = r_iono, dipole_moment = (0.0, 0.0, -3.0e-5))
    jr_33 = [
        1.0e-6 * sin(sol_view.theta[i])^2 * sin(sol_view.phi[j])
            for i in 1:(sol_view.n_theta), j in 1:(sol_view.n_phi)
    ]
    solve_ionosphere!(sol_view, jr_33)
    view_f = ionosphere_field_view(sol_view)
    b0 = dipole_B0(frame, x_sim)
    x_pole = SVector{3, Float64}(0.0, 0.0, 2.5)
    b0_pole = dipole_B0(frame, x_pole)

    pts_3d = [
        let
            thi = 0.25 + 0.001 * mod(i, 400)
            phi = 0.00628 * i
            SVector{3, Float64}(
                2.5 * sin(thi) * cos(phi),
                2.5 * sin(thi) * sin(phi),
                2.5 * cos(thi),
            )
        end for i in 1:1000
    ]

    SUITE["mapping"] = BenchmarkGroup(["mapping"])
    SUITE["mapping"]["eval_ionosphere_E"] = @benchmarkable eval_ionosphere_E(
        $view_f, $x_sim; B0 = $b0,
    )
    SUITE["mapping"]["eval_ionosphere_E_pole"] = @benchmarkable eval_ionosphere_E(
        $view_f, $x_pole; B0 = $b0_pole,
    )
    SUITE["mapping"]["eval_ionosphere_u_exb"] = @benchmarkable eval_ionosphere_u_exb(
        $view_f, $x_sim, $b0,
    )
    SUITE["mapping"]["batch_u_exb_1000"] = @benchmarkable begin
        s = zero(SVector{3, Float64})
        @inbounds for x in $pts_3d
            b = dipole_B0($frame, x)
            s += eval_ionosphere_u_exb($view_f, x, b)
        end
        s[1] + s[2] + s[3]
    end

    # 4. 2D spherical ILU(0)-BiCGSTAB solvers (warm & cold starts)
    sol_warm = IonosphereSolver(cfg_33; r_iono = r_iono, dipole_moment = (0.0, 0.0, -3.0e-5))
    solve_ionosphere!(sol_warm, jr_33)
    sol_cold = IonosphereSolver(cfg_33; r_iono = r_iono, dipole_moment = (0.0, 0.0, -3.0e-5))

    cfg_33_aur = IonosphereConfig(;
        n_theta_hem = 33,
        n_phi = 64,
        conductance = m_aur,
        tol = 1.0e-8,
        max_iter = 500,
    )
    sol_aur = IonosphereSolver(cfg_33_aur; r_iono = r_iono, dipole_moment = (0.0, 0.0, -3.0e-5))

    r_planet = 6.371e6
    cfg_fleks = IonosphereConfig(;
        n_theta_hem = 91,
        n_phi = 180,
        lat_boundary_deg = 20.0,
        low_lat_bc = :dirichlet,
        conductance = m_aur,
        use_dip_angle = true,
        sigma_parallel = 1000.0,
        sigma_P_floor = 0.1,
        neutralize_fac = false,
        n_smooth_fac = 1,
        tol = 1.0e-8,
        max_iter = 1000,
    )
    sol_fleks = IonosphereSolver(
        cfg_fleks;
        r_iono = 1.017 * r_planet,
        dipole_moment = (0.0, 0.0, -3.0e-5 * r_planet^3),
    )
    th_r1 = deg2rad(18.0)
    th_r2 = deg2rad(24.0)
    sig = deg2rad(2.5)
    jr_91 = zeros(sol_fleks.n_theta, sol_fleks.n_phi)
    for i in 1:(sol_fleks.n_theta)
        th_i = sol_fleks.theta[i]
        th_hem = th_i <= 0.5 * pi ? th_i : (pi - th_i)
        r1 = exp(-0.5 * ((th_hem - th_r1) / sig)^2)
        r2 = exp(-0.5 * ((th_hem - th_r2) / sig)^2)
        for j in 1:(sol_fleks.n_phi)
            jr_91[i, j] = (1.2e-6 * r1 - 0.6e-6 * r2) * sin(sol_fleks.phi[j])
        end
    end

    SUITE["solver"] = BenchmarkGroup(["solver"])
    SUITE["solver"]["solve_33x64_uniform_warm"] = @benchmarkable solve_ionosphere!(
        $sol_warm, $jr_33,
    )
    SUITE["solver"]["solve_33x64_uniform_cold"] = @benchmarkable solve_ionosphere!(
        $sol_cold, $jr_33,
    ) setup = reset_ionosphere!($sol_cold) evals = 1
    SUITE["solver"]["solve_33x64_auroral_cold"] = @benchmarkable solve_ionosphere!(
        $sol_aur, $jr_33,
    ) setup = reset_ionosphere!($sol_aur) evals = 1
    SUITE["solver"]["solve_91x180_fleks_cold"] = @benchmarkable solve_ionosphere!(
        $sol_fleks, $jr_91,
    ) setup = reset_ionosphere!($sol_fleks) evals = 1
end
