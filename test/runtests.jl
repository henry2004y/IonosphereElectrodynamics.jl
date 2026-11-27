using LinearAlgebra: dot, norm
using Ionosphere
using LingXiaoTest: run_hygiene_checks, verify_api_coverage
using StaticArrays: SMatrix, SVector
using Test

@testset "Ionosphere.jl" begin
    @testset "Package Hygiene & API Docstrings" begin
        run_hygiene_checks(Ionosphere)
        cov = verify_api_coverage(Ionosphere, pkgdir(Ionosphere))
        @test isempty(cov.missing_docstrings)
    end

    @testset "Validation & Conductance Models" begin
        @test_throws ArgumentError UniformConductance(0.0, 0.0)
        @test_throws ArgumentError UniformConductance(-1.0, 0.0)
        @test_throws ArgumentError UniformConductance(1.0, Inf)
        @test_throws ArgumentError SolarEUVConductance(; sigma_P_night = 0.0)
        @test_throws ArgumentError SolarEUVConductance(; sun_dir = (0.0, 0.0, 0.0))
        @test_throws ArgumentError MoenBrekkeEUVConductance(; f107 = 0.5)
        @test_throws ArgumentError AuroralConductance(UniformConductance(1.0); mode = :invalid)

        @test_throws ArgumentError IonosphereConfig(; n_theta_hem = 4)
        @test_throws ArgumentError IonosphereConfig(; n_phi = 7)
        @test_throws ArgumentError IonosphereConfig(; n_phi = 9)
        @test_throws ArgumentError IonosphereConfig(; low_lat_bc = :invalid)
        @test_throws ArgumentError IonosphereConfig(; particle_bc = :invalid)
        @test_throws ArgumentError IonosphereConfig(; relax_alpha = 0.0)
        @test_throws ArgumentError IonosphereConfig(; couple_every = 0)
        @test_throws ArgumentError IonosphereConfig(; u_max = 0.0)
        @test_throws ArgumentError IonosphereConfig(; r_iono = 2.0, r_ib = 1.0)
        @test_throws ArgumentError IonosphereConfig(; r_fac = 0.8, r_ib = 1.0)

        mb = MoenBrekkeEUVConductance(; f107 = 150.0, sigma_P_night = 3.0, sigma_H_night = 3.0, sun_dir = (1.0, 0.0, 0.0))
        sp_day, sh_day = evaluate_conductance(mb, 0.5 * pi, 0.0, SVector(1.0, 0.0, 0.0), 0.0)
        sp_ngt, sh_ngt = evaluate_conductance(mb, 0.5 * pi, pi, SVector(-1.0, 0.0, 0.0), 0.0)
        @test sp_day > sp_ngt == 3.0
        @test sh_day > sh_ngt == 3.0

        aur = AuroralConductance(mb; sigma_P_max = 15.0, hall_ratio = 1.5, fac_scale = 1.0e-6)
        sp_up, sh_up = evaluate_conductance(aur, 0.5 * pi, pi, SVector(-1.0, 0.0, 0.0), 0.0; jr_outward = 1.0e-6)
        sp_dn, sh_dn = evaluate_conductance(aur, 0.5 * pi, pi, SVector(-1.0, 0.0, 0.0), 0.0; jr_outward = -1.0e-6)
        @test sp_up > sp_dn == 3.0
        @test sh_up > sh_dn == 3.0
    end

    @testset "Tilted Dipole Frame, 2x3 Jacobian & E ⋅ B0 ≡ 0 (Including Pole)" begin
        th_tilt = deg2rad(23.0)
        ph_tilt = deg2rad(35.0)
        b_eq = -3.0e-5
        mom = SVector(-b_eq * sin(th_tilt) * cos(ph_tilt), -b_eq * sin(th_tilt) * sin(ph_tilt), b_eq * cos(th_tilt))
        ctr = SVector(0.5, -0.3, 0.2)

        cfg = IonosphereConfig(; n_theta_hem = 91, n_phi = 180)
        sol = IonosphereSolver(cfg; r_iono = 1.017, dipole_moment = mom, center = ctr)
        @test sol.frame.dipole_sign == -1.0
        @test sol.frame.rot[3, 3] > 0.0

        # Seed smooth potential on both hemispheres
        for j in 1:sol.n_phi, i in 1:sol.n_theta
            sol.phi_pot[i, j] = 5.0e4 * sin(sol.theta[i])^2 * sin(sol.phi[j])
        end
        Ionosphere._compute_gradients!(sol)

        view = ionosphere_field_view(sol)
        r_body = 3.0
        eps_fd = 1.0e-6
        max_ortho = 0.0
        max_jac_err = 0.0

        for ilat in (-70, -50, 50, 70), ilon in 0:45:315
            th = deg2rad(90 - ilat)
            ph = deg2rad(ilon)
            x_sim = ctr + SVector(r_body * sin(th) * cos(ph), r_body * sin(th) * sin(ph), r_body * cos(th))
            b0 = dipole_B0(sol.frame, x_sim)
            e = eval_ionosphere_E(view, x_sim)
            if norm(e) > 1.0e-12
                max_ortho = max(max_ortho, abs(dot(e, b0)) / (norm(e) * norm(b0)))
            end

            x_smg = sol.frame.rot * (x_sim - ctr)
            i_hem, _, _, _, jac = map_dipole_smg_with_jacobian(x_smg, sol.frame.r_iono)
            if i_hem != 0
                for d in 1:3
                    dv = SVector{3, Float64}(d == 1 ? eps_fd : 0.0, d == 2 ? eps_fd : 0.0, d == 3 ? eps_fd : 0.0)
                    _, _, thp, php = map_dipole_smg(x_smg + dv, sol.frame.r_iono)
                    _, _, thm, phm = map_dipole_smg(x_smg - dv, sol.frame.r_iono)
                    dth_num = (thp - thm) / (2 * eps_fd)
                    dph_diff = atan(sin(php - phm), cos(php - phm))
                    dph_num = dph_diff / (2 * eps_fd)
                    max_jac_err = max(max_jac_err, abs(jac[1, d] - dth_num), abs(jac[2, d] - dph_num))
                end
            end
        end
        @test max_ortho < 1.0e-13
        @test max_jac_err < 1.0e-8

        # Exact magnetic pole regularity: nonzero convection E_IB and E ⋅ B0 == 0 right at rho = 0!
        for j in 1:sol.n_phi, i in 1:sol.n_theta
            sol.phi_pot[i, j] = 1.0e4 * sin(sol.theta[i]) * sin(sol.phi[j])
        end
        Ionosphere._compute_gradients!(sol)
        x_pole_N = ctr + r_body * (transpose(sol.frame.rot) * SVector(0.0, 0.0, 1.0))
        e_pole = eval_ionosphere_E(sol, x_pole_N)
        b_pole = dipole_B0(sol.frame, x_pole_N)
        @test norm(e_pole) > 1.0
        @test abs(dot(e_pole, b_pole)) / (norm(e_pole) * norm(b_pole)) < 1.0e-14
    end

    @testset "2nd-Order MMS Convergence (Pedersen, Hall+Dip, SolarEUV+Cowling, N & S)" begin
        r_iono = 1.017
        lat_bnd = 30.0
        th_max = deg2rad(90.0 - lat_bnd)
        phi0 = 4.0e4

        function mms_phi(th, ph)
            a = th <= 0.5 * pi ? th : pi - th
            a >= th_max && return 0.0
            return phi0 * sin(a)^2 * sin(pi * a / th_max) * (cos(ph) + 0.5 * sin(2 * ph))
        end

        function mms_jr(th, ph, cond, dip_sign, s0_par, use_dip)
            eps_d = 1.0e-5
            function flux(th_q, ph_q)
                r_hat = SVector(sin(th_q) * cos(ph_q), sin(th_q) * sin(ph_q), cos(th_q))
                sp, sh = evaluate_conductance(cond, th_q, ph_q, r_hat, 0.0)
                s_tt, s_tp, s_pp = dipole_conductance_tensor(
                    sp, sh, cos(th_q), sin(th_q), dip_sign;
                    sigma_parallel = s0_par, use_dip_angle = use_dip,
                )
                dp_dth = (mms_phi(th_q + eps_d, ph_q) - mms_phi(th_q - eps_d, ph_q)) / (2 * eps_d)
                dp_dph = (mms_phi(th_q, ph_q + eps_d) - mms_phi(th_q, ph_q - eps_d)) / (2 * eps_d)
                f_th = sin(th_q) * s_tt * dp_dth + s_tp * dp_dph
                f_ph = -s_tp * dp_dth + (s_pp / sin(th_q)) * dp_dph
                return (f_th, f_ph)
            end
            ft_p, _ = flux(th + eps_d, ph)
            ft_m, _ = flux(th - eps_d, ph)
            _, fp_p = flux(th, ph + eps_d)
            _, fp_m = flux(th, ph - eps_d)
            return ((ft_p - ft_m) + (fp_p - fp_m)) / (2 * eps_d * r_iono^2 * sin(th))
        end

        for (label, cond, dip_sign) in (
                ("Uniform Pedersen", UniformConductance(5.0, 0.0), -1.0),
                ("Uniform Ped+Hall+Dip", UniformConductance(5.0, 10.0), -1.0),
                ("SolarEUV+Cowling (reversed polarity)", SolarEUVConductance(; sigma_P_day = 8.0, sigma_H_day = 12.0, sigma_P_night = 2.0, sigma_H_night = 1.0, sun_dir = (1.0, 0.0, 0.0)), +1.0),
            )
            errs_N = Float64[]
            errs_S = Float64[]
            for nth_hem in (24, 46, 91)
                np = 2 * (nth_hem - 1)
                cfg = IonosphereConfig(
                    ;
                    n_theta_hem = nth_hem, n_phi = np,
                    lat_boundary_deg = lat_bnd, low_lat_bc = :dirichlet,
                    conductance = cond, use_dip_angle = true,
                    sigma_parallel = 1000.0, neutralize_fac = false,
                    tol = 1.0e-11, max_iter = 2000,
                )
                sol = IonosphereSolver(cfg; r_iono = r_iono, dipole_moment = (0.0, 0.0, dip_sign * 3.0e-5))
                jr = zeros(sol.n_theta, sol.n_phi)
                for j in 1:sol.n_phi
                    ph = sol.phi[j]
                    for i in (sol.i_min_N):(sol.i_max_N)
                        jr[i, j] = mms_jr(sol.theta[i], ph, cond, dip_sign, 1000.0, true)
                    end
                    for i in (sol.i_min_S):(sol.i_max_S)
                        jr[i, j] = mms_jr(sol.theta[i], ph, cond, dip_sign, 1000.0, true)
                    end
                end
                @test solve_ionosphere!(sol, jr)
                eN = 0.0; rN = 0.0
                for j in 1:sol.n_phi, i in (sol.i_min_N):(sol.i_max_N)
                    ex = mms_phi(sol.theta[i], sol.phi[j])
                    eN += (sol.phi_pot[i, j] - ex)^2
                    rN += ex^2
                end
                eS = 0.0; rS = 0.0
                for j in 1:sol.n_phi, i in (sol.i_min_S):(sol.i_max_S)
                    ex = mms_phi(sol.theta[i], sol.phi[j])
                    eS += (sol.phi_pot[i, j] - ex)^2
                    rS += ex^2
                end
                push!(errs_N, sqrt(eN / rN))
                push!(errs_S, sqrt(eS / rS))
            end
            @test log2(errs_N[1] / errs_N[2]) > 1.9
            @test log2(errs_N[2] / errs_N[3]) > 1.9
            @test log2(errs_S[1] / errs_S[2]) > 1.9
            @test log2(errs_S[2] / errs_S[3]) > 1.9
            @test errs_N[3] < 1.0e-3
            @test errs_S[3] < 1.0e-3
        end
    end

    @testset "Exact FLEKS Analytic R1+R2 + Moen-Brekke EUV + Auroral Benchmark (128.2 kV)" begin
        r_planet_si = 6.371e6
        cond = AuroralConductance(
            MoenBrekkeEUVConductance(; f107 = 150.0, sigma_P_night = 3.0, sigma_H_night = 3.0, sun_dir = (1.0, 0.0, 0.0));
            sigma_P_max = 15.0, hall_ratio = 1.5, fac_scale = 1.0e-6, mode = :tanh_sqrt,
        )
        cfg = IonosphereConfig(
            ;
            n_theta_hem = 91, n_phi = 180,
            lat_boundary_deg = 20.0, low_lat_bc = :dirichlet,
            conductance = cond, use_dip_angle = true,
            sigma_parallel = 1000.0, sigma_P_floor = 0.1,
            neutralize_fac = false, n_smooth_fac = 1,
            tol = 1.0e-8, max_iter = 1000,
        )
        sol = IonosphereSolver(cfg; r_iono = 1.017 * r_planet_si, dipole_moment = (0.0, 0.0, -3.0e-5 * r_planet_si^3))

        th_r1 = deg2rad(18.0); th_r2 = deg2rad(24.0); sig = deg2rad(2.5)
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

        @test solve_ionosphere!(sol, jr)
        cpcp = cross_polar_cap_potential(sol)
        cpcp_n_kv = cpcp.north * 1.0e-3
        cpcp_s_kv = cpcp.south * 1.0e-3
        @test isapprox(cpcp_n_kv, 128.2; rtol = 0.01)
        @test isapprox(cpcp_s_kv, cpcp_n_kv; rtol = 1.0e-6)

        # Verify GSM Dungey convection signs: Dawn (3pi/2) > 0, Dusk (pi/2) < 0, antisunward ExB (-x) across poles
        i_r1_N = argmin([abs(sol.theta[i] - th_r1) for i in 1:(sol.n_theta_hem)])
        i_r1_S = argmin([abs(sol.theta[i] - (pi - th_r1)) for i in (sol.n_theta_hem):(sol.n_theta)]) + sol.n_theta_hem - 1
        j_dusk = 1 + sol.n_phi ÷ 4
        j_dawn = 1 + 3 * (sol.n_phi ÷ 4)
        @test sol.phi_pot[i_r1_N, j_dawn] > 5.0e4
        @test sol.phi_pot[i_r1_N, j_dusk] < -5.0e4
        @test sol.phi_pot[i_r1_S, j_dawn] > 5.0e4
        @test sol.phi_pot[i_r1_S, j_dusk] < -5.0e4

        # Check Noon (j=1) potential shifts negative from Hall rotation in BOTH hemispheres
        @test sol.phi_pot[i_r1_N, 1] < -1.0e3
        @test sol.phi_pot[i_r1_S, 1] < -1.0e3

        # Antisunward ExB drift at both North and South poles (sun at +x -> antisunward is -x)
        x_pN = SVector(0.0, 0.2 * r_planet_si, 3.0 * r_planet_si)
        x_pS = SVector(0.0, 0.2 * r_planet_si, -3.0 * r_planet_si)
        u_N = eval_ionosphere_u_exb(sol, x_pN, dipole_B0(sol.frame, x_pN); u_max = 1.0e9)
        u_S = eval_ionosphere_u_exb(sol, x_pS, dipole_B0(sol.frame, x_pS); u_max = 1.0e9)
        @test u_N[1] < 0.0
        @test u_S[1] < 0.0

        # Reversed dipole polarity flips ExB direction across poles when FAC is unchanged,
        # and Hall rotation reverses azimuthal sense
        sol_rev = IonosphereSolver(cfg; r_iono = 1.017 * r_planet_si, dipole_moment = (0.0, 0.0, +3.0e-5 * r_planet_si^3))
        @test sol_rev.frame.dipole_sign == 1.0
        @test solve_ionosphere!(sol_rev, jr)
        @test sol_rev.phi_pot[i_r1_N, 1] > 1.0e3
        @test sol_rev.phi_pot[i_r1_S, 1] > 1.0e3
    end

    @testset "Neumann Low-Lat BC, Analytic 3D Corotation (No Boundary Spike) & 0 B Alloc" begin
        for dip_sign in (-1.0, +1.0)
            omega = SVector(0.1, -0.05, 0.25)
            cfg = IonosphereConfig(
                ;
                n_theta_hem = 33, n_phi = 64,
                lat_boundary_deg = 20.0, low_lat_bc = :neumann,
                conductance = UniformConductance(5.0, 2.0),
                omega_corot = omega, u_max = 100.0,
                tol = 1.0e-11, max_iter = 1000,
            )
            sol = IonosphereSolver(cfg; r_iono = 0.8, dipole_moment = (0.0, 0.0, dip_sign * 6.0))
            @test solve_ionosphere!(sol) # Jr == 0 -> pure corotation

            # Verify exact rigid-body Omega x r ALL THE WAY to just inside lat_boundary_deg (70 deg colat!)
            r_ib = 1.6
            max_cor_err = 0.0
            for th_deg in (10.0, 25.0, 45.0, 60.0, 68.0, 112.0, 120.0, 155.0), ph_deg in (15.0, 110.0, 230.0)
                th = deg2rad(th_deg); ph = deg2rad(ph_deg)
                x_sim = SVector(r_ib * sin(th) * cos(ph), r_ib * sin(th) * sin(ph), r_ib * cos(th))
                b0 = dipole_B0(sol.frame, x_sim)
                u_exb = eval_ionosphere_u_exb(sol, x_sim, b0)
                u_exact = SVector(
                    omega[2] * x_sim[3] - omega[3] * x_sim[2],
                    omega[3] * x_sim[1] - omega[1] * x_sim[3],
                    omega[1] * x_sim[2] - omega[2] * x_sim[1],
                )
                max_cor_err = max(max_cor_err, norm(u_exb - u_exact))
                e_cor = eval_ionosphere_E(sol, x_sim; B0 = b0)
                @test abs(dot(e_cor, b0)) / max(norm(e_cor) * norm(b0), 1.0e-30) < 1.0e-14
            end
            @test max_cor_err < 1.0e-13
        end

        # Steady-state 0 B allocation checks
        cfg_a = IonosphereConfig(; n_theta_hem = 25, n_phi = 48, conductance = AuroralConductance(SolarEUVConductance()))
        sol_a = IonosphereSolver(cfg_a; r_iono = 1.0, dipole_moment = (0.0, 0.0, -5.0))
        jr_a = [sin(sol_a.theta[i]) * sin(sol_a.phi[j]) for i in 1:sol_a.n_theta, j in 1:sol_a.n_phi]
        solve_ionosphere!(sol_a, jr_a)
        @test (@allocated solve_ionosphere!(sol_a, jr_a)) == 0

        v_a = ionosphere_field_view(sol_a)
        x_a = SVector(0.5, 0.4, 1.5)
        b_a = dipole_B0(sol_a.frame, x_a)
        eval_ionosphere_E(v_a, x_a; B0 = b_a)
        eval_ionosphere_u_exb(v_a, x_a, b_a)
        @test (@allocated eval_ionosphere_E(v_a, x_a; B0 = b_a)) == 0
        @test (@allocated eval_ionosphere_u_exb(v_a, x_a, b_a)) == 0
    end

    @testset "End-to-End 3D B1 -> J_parallel(r_fac) -> Phi -> (E, u_ExB) Mapping" begin
        r_iono = 0.6
        r_ib = 1.2
        r_fac = 2.0
        dip_mom = (0.0, 0.0, -5.0)
        L0 = 4.5
        dL = 1.8
        A0 = 0.45

        function b1_analytic(x_vec)
            x, y, z = x_vec[1], x_vec[2], x_vec[3]
            rho = hypot(x, y)
            r = hypot(rho, z)
            if r <= 0.35 * r_ib || rho <= 1.0e-8
                return SVector(0.0, 0.0, -0.2)
            end
            sin_th = rho / r
            L = r / (sin_th * sin_th)
            w_r = 0.5 * (1.0 + tanh((r - 0.8 * r_ib) / (0.2 * r_ib)))
            sin_phi = y / rho
            cos_phi = x / rho
            cos_th = z / r
            g_val = -A0 * cos_th * exp(-((L - L0) / dL)^2) * sin_phi * w_r
            b_phi = g_val / rho
            return SVector(-b_phi * sin_phi, b_phi * cos_phi, -0.2)
        end

        function curl_b1(x_vec)
            h = 1.0e-6
            ex = SVector(h, 0.0, 0.0)
            ey = SVector(0.0, h, 0.0)
            ez = SVector(0.0, 0.0, h)
            bx_p = b1_analytic(x_vec + ex); bx_m = b1_analytic(x_vec - ex)
            by_p = b1_analytic(x_vec + ey); by_m = b1_analytic(x_vec - ey)
            bz_p = b1_analytic(x_vec + ez); bz_m = b1_analytic(x_vec - ez)
            dBz_dy = (by_p[3] - by_m[3]) / (2 * h)
            dBy_dz = (bz_p[2] - bz_m[2]) / (2 * h)
            dBx_dz = (bz_p[1] - bz_m[1]) / (2 * h)
            dBz_dx = (bx_p[3] - bx_m[3]) / (2 * h)
            dBy_dx = (bx_p[2] - bx_m[2]) / (2 * h)
            dBx_dy = (by_p[1] - by_m[1]) / (2 * h)
            return SVector(dBz_dy - dBy_dz, dBx_dz - dBz_dx, dBy_dx - dBx_dy)
        end

        cfg5 = IonosphereConfig(;
            r_iono = r_iono, r_ib = r_ib, r_fac = r_fac,
            dipole_moment = dip_mom, n_theta_hem = 46, n_phi = 72,
            conductance = UniformConductance(5.0, 0.0),
            u_max = 100.0, tol = 1.0e-11, max_iter = 1000,
        )
        cfg25 = IonosphereConfig(;
            r_iono = r_iono, r_ib = r_ib, r_fac = r_fac,
            dipole_moment = dip_mom, n_theta_hem = 46, n_phi = 72,
            conductance = UniformConductance(25.0, 0.0),
            u_max = 100.0, tol = 1.0e-11, max_iter = 1000,
        )
        cfg_euv = IonosphereConfig(;
            r_iono = r_iono, r_ib = r_ib, r_fac = r_fac,
            dipole_moment = dip_mom, n_theta_hem = 46, n_phi = 72,
            conductance = SolarEUVConductance(;
                sigma_P_day = 8.0, sigma_H_day = 12.0,
                sigma_P_night = 2.0, sigma_H_night = 1.0,
                sun_dir = (-1.0, 0.0, 0.0),
            ),
            u_max = 100.0, tol = 1.0e-11, max_iter = 1000,
        )

        sol5 = IonosphereSolver(cfg5)
        sol25 = IonosphereSolver(cfg25)
        sol_euv = IonosphereSolver(cfg_euv)

        jr_3d = zeros(sol5.n_theta, sol5.n_phi)
        for j in 1:sol5.n_phi, i in 1:sol5.n_theta
            in_active = (sol5.i_min_N <= i <= sol5.i_max_N) || (sol5.i_min_S <= i <= sol5.i_max_S)
            in_active || continue
            valid, x_fac, _, _, b_iono_mag, br_over_b_iono = dipole_fac_sample_point(
                sol5.frame, sol5.theta[i], sol5.phi[j], r_fac,
            )
            valid || continue
            b_tot = dipole_B0(sol5.frame, x_fac) + b1_analytic(x_fac)
            b_tot_mag = norm(b_tot)
            j_vec = curl_b1(x_fac)
            j_par = dot(j_vec, b_tot) / b_tot_mag
            jr_3d[i, j] = j_par * (b_iono_mag / b_tot_mag) * br_over_b_iono
        end

        @test solve_ionosphere!(sol5, jr_3d)
        @test solve_ionosphere!(sol25, jr_3d)
        @test solve_ionosphere!(sol_euv, jr_3d)

        j_dawn = 1 + sol5.n_phi ÷ 4         # phi = pi/2 (+y Dawn)
        j_dusk = 1 + 3 * (sol5.n_phi ÷ 4)   # phi = 3pi/2 (-y Dusk)
        i_r1_N = argmin([abs(sol5.theta[i] - deg2rad(16.0)) for i in 1:(sol5.n_theta_hem)])
        i_r2_N = argmin([abs(sol5.theta[i] - deg2rad(28.0)) for i in 1:(sol5.n_theta_hem)])

        # Clear R1 (poleward: -down at Dawn, +up at Dusk) & R2 (equatorward: +up at Dawn, -down at Dusk)
        @test sol5.jr_outward[i_r1_N, j_dawn] < -1.0
        @test sol5.jr_outward[i_r1_N, j_dusk] > 1.0
        @test sol5.jr_outward[i_r2_N, j_dawn] > 0.2
        @test sol5.jr_outward[i_r2_N, j_dusk] < -0.2

        # Exact North/South FAC & potential symmetry
        jr_S_mirrored = reverse(sol5.jr_outward[(sol5.n_theta_hem):end, :]; dims = 1)
        @test isapprox(sol5.jr_outward[1:(sol5.n_theta_hem), :], jr_S_mirrored; rtol = 1.0e-8)
        @test sol5.phi_solved[i_r1_N, j_dawn] > 0.0
        @test sol5.phi_solved[i_r1_N, j_dusk] < 0.0

        # 1 / Sigma_P CPCP scaling & Hall phase shift
        cp5 = cross_polar_cap_potential(sol5)
        cp25 = cross_polar_cap_potential(sol25)
        @test cp5.north > 0.0
        @test isapprox(cp5.south, cp5.north; rtol = 1.0e-6)
        @test isapprox(cp5.north / cp25.north, 5.0; rtol = 1.0e-6)
        @test maximum(abs, sol_euv.phi_solved .- sol5.phi_solved) > 1.0e-4

        # Map back to 3D inner boundary r_ib: antisunward (+x) ExB at poles and E . B0 == 0
        for sol_test in (sol5, sol_euv)
            x_pole_N = SVector(0.0, 0.1 * r_ib, 0.99 * r_ib)
            x_pole_S = SVector(0.0, 0.1 * r_ib, -0.99 * r_ib)
            u_pN = eval_ionosphere_u_exb(sol_test, x_pole_N, dipole_B0(sol_test.frame, x_pole_N))
            u_pS = eval_ionosphere_u_exb(sol_test, x_pole_S, dipole_B0(sol_test.frame, x_pole_S))
            @test u_pN[1] > 0.0
            @test u_pS[1] > 0.0

            max_ortho_3d = 0.0
            for th_deg in (12.0, 22.0, 35.0, 145.0, 158.0, 168.0), ph_deg in (20.0, 90.0, 210.0, 300.0)
                th = deg2rad(th_deg); ph = deg2rad(ph_deg)
                x_ib = SVector(r_ib * sin(th) * cos(ph), r_ib * sin(th) * sin(ph), r_ib * cos(th))
                b0 = dipole_B0(sol_test.frame, x_ib)
                e_ib = eval_ionosphere_E(sol_test, x_ib; B0 = b0)
                if norm(e_ib) > 1.0e-12
                    max_ortho_3d = max(max_ortho_3d, abs(dot(e_ib, b0)) / (norm(e_ib) * norm(b0)))
                end
            end
            @test max_ortho_3d < 1.0e-13
        end
    end

    @testset "Electron Mobility Tensor" begin
        b_z = SVector(0.0, 0.0, 1.0)
        M = electron_mobility_tensor(2.0, 3.0, b_z; sigma_0 = 5.0)
        @test M ≈ SMatrix{3, 3, Float64}(
            2.0, 3.0, 0.0,
            -3.0, 2.0, 0.0,
            0.0, 0.0, 5.0,
        )
        @test M * b_z ≈ 5.0 * b_z
        @test (@allocated electron_mobility_tensor(2.0, 3.0, b_z; sigma_0 = 5.0)) == 0
    end
end
