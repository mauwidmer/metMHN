import unittest

import jax.numpy as jnp
import numpy as np

import metmhn.regularized_optimization as reg_opt


class TestCompositePenaltyGradient(unittest.TestCase):
    def test_composite_penalty_gradient_matches_finite_difference(self):
        n = 4
        n_loc = 3
        rng = np.random.default_rng(seed=42)

        log_theta = jnp.array(rng.normal(size=(n, n)))
        theta_loc_GM = jnp.array(rng.normal(size=(n, n_loc)))
        theta_loc_MG = jnp.array(rng.normal(size=(n, n_loc)))
        omega_m = jnp.array(rng.normal(size=(n,)))
        omega_p = jnp.array(rng.normal(size=(n,)))
        c = jnp.array(rng.uniform(0, 1, size=(n_loc,)))

        theta_ext = reg_opt.create_theta_extended(
            log_theta=log_theta,
            omega_m=omega_m,
            omega_p=omega_p,
            theta_loc_GM=theta_loc_GM,
            theta_loc_MG=theta_loc_MG,
        )

        lam = 0.7
        penalty = float(reg_opt.composite_penalty(theta_ext, lam, c))
        grad_analytical = reg_opt.composite_penalty_(theta_ext, lam, c)

        h = 1e-6
        grad_numerical = np.zeros(log_theta.size)
        for idx in range(log_theta.size):
            i = idx // n
            j = idx % n
            log_theta_h = log_theta.at[i, j].add(h)
            theta_ext_h = reg_opt.create_theta_extended(
                log_theta=log_theta_h,
                omega_m=omega_m,
                omega_p=omega_p,
                theta_loc_GM=theta_loc_GM,
                theta_loc_MG=theta_loc_MG,
            )
            penalty_h = float(reg_opt.composite_penalty(theta_ext_h, lam, c))
            grad_numerical[idx] = (penalty_h - penalty) / h

        np.testing.assert_allclose(
            grad_analytical,
            grad_numerical,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_reparametrization_scales_last_row_and_column_for_all_locations(self):
        n = 4
        n_loc = 3
        rng = np.random.default_rng(seed=24)

        log_theta = jnp.array(rng.normal(size=(n, n)))
        theta_loc_GM = jnp.array(rng.normal(size=(n, n_loc)))
        theta_loc_MG = jnp.array(rng.normal(size=(n, n_loc)))
        omega_m = jnp.array(rng.normal(size=(n,)))
        omega_p = jnp.array(rng.normal(size=(n,)))

        theta_ext = reg_opt.create_theta_extended(
            log_theta=log_theta,
            omega_m=omega_m,
            omega_p=omega_p,
            theta_loc_GM=theta_loc_GM,
            theta_loc_MG=theta_loc_MG,
        )

        theta_rep = reg_opt.reparametrization(theta_ext)

        self.assertEqual(theta_rep.shape, (n_loc, n, n))
        for loc_idx in range(n_loc):
            np.testing.assert_allclose(theta_rep[loc_idx, :-1, :-1], log_theta[:-1, :-1])
            np.testing.assert_allclose(
                theta_rep[loc_idx, -1, :-1],
                log_theta[-1, :-1] * theta_loc_GM[:-1, loc_idx],
            )
            np.testing.assert_allclose(
                theta_rep[loc_idx, :-1, -1],
                log_theta[:-1, -1] * theta_loc_MG[:-1, loc_idx],
            )
            np.testing.assert_allclose(theta_rep[loc_idx, -1, -1], log_theta[-1, -1])

    def test_reparametrization_grad_matches_finite_difference(self):
        n = 4
        n_loc = 3
        rng = np.random.default_rng(seed=100)

        log_theta = jnp.array(rng.normal(size=(n, n)))
        theta_loc_GM = jnp.array(rng.normal(size=(n, n_loc)))
        theta_loc_MG = jnp.array(rng.normal(size=(n, n_loc)))
        omega_m = jnp.array(rng.normal(size=(n,)))
        omega_p = jnp.array(rng.normal(size=(n,)))

        theta_ext = reg_opt.create_theta_extended(
            log_theta=log_theta,
            omega_m=omega_m,
            omega_p=omega_p,
            theta_loc_GM=theta_loc_GM,
            theta_loc_MG=theta_loc_MG,
        )

        grad_rep = jnp.array(rng.normal(size=(n_loc, n, n)))
        grad_theta = reg_opt.reparametrization_grad(theta_ext, grad_rep)

        base_value = jnp.sum(reg_opt.reparametrization(theta_ext) * grad_rep)
        h = 1e-6
        grad_numerical = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                log_theta_h = log_theta.at[i, j].add(h)
                theta_ext_h = reg_opt.create_theta_extended(
                    log_theta=log_theta_h,
                    omega_m=omega_m,
                    omega_p=omega_p,
                    theta_loc_GM=theta_loc_GM,
                    theta_loc_MG=theta_loc_MG,
                )
                value_h = jnp.sum(reg_opt.reparametrization(theta_ext_h) * grad_rep)
                grad_numerical[i, j] = float((value_h - base_value) / h)

        np.testing.assert_allclose(grad_theta, grad_numerical, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
