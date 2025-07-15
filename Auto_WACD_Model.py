import numpy as np
from scipy.optimize import minimize


def compute_psi_wacd(params, x, p, q):
    omega = params[0] # omega coeff
    alphas = params[1:1+p] # alphas coeff
    betas = params[1+p:1+p+q] # beta coeff
    T = len(x)
    psi = np.zeros(T)
    psi[0] = np.mean(x) # Init psi
    for t in range(1, T):
        val_alpha = sum(alphas[i] * x[t - i - 1] for i in range(p) if t - i - 1 >= 0)
        val_beta = sum(betas[j] * psi[t - j - 1] for j in range(q) if t - j - 1 >= 0)
        psi[t] = max(omega + val_alpha + val_beta, 1e-6)

    return psi


def neg_log_likelihood_wacd(params, x, p, q):
    omega = params[0]
    alphas = params[1:1+p]
    betas = params[1+p:1+p+q]
    k = params[-1]  # Weibull shape parameter (last param)

    # Constraints
    if omega <= 0 or k <= 0 or np.any(np.array(alphas) < 0) or np.any(np.array(betas) < 0) or (np.sum(alphas) + np.sum(betas) >= 1):
        return 1e10

    psi = compute_psi_wacd(params[:-1], x, p, q)
    if np.any(psi <= 0) or np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        return 1e10
    z = x / psi # Error calculation
    if np.any(z <= 0) or np.any(np.isnan(z)):
        return 1e10

    # Calculation of log likelihood for Weilbull
    ll = np.sum(np.log(k) - np.log(psi) + (k - 1) * np.log(z) - z ** k)
    return -ll # Return negative log likelihood


def select_best_wacd_model(x, p_max=3, q_max=3):
    best_aic = np.inf
    best_model = None
    best_params = None
    best_ll = None
    T = len(x)

    for p in range(1, p_max+1):
        for q in range(1, q_max+1):
            init_params = [np.mean(x) * 0.1] + [0.1] * p + [0.8] * q + [1.5]  # Last element is k (shape)
            bounds = [(1e-6, None)] + [(0, 0.999)]*(p+q) + [(1e-3, None)]  # Shape > 0

            res = minimize(neg_log_likelihood_wacd, init_params, args=(x, p, q), method="L-BFGS-B", bounds=bounds)

            if res.success:
                k_params = len(init_params)
                ll = -res.fun
                aic = 2 * k_params - 2 * ll

                if aic < best_aic:
                    best_aic = aic
                    best_model = (p, q)
                    best_params = res.x
                    best_ll = ll

    print(f"Best WACD model: p={best_model[0]}, q={best_model[1]}, with AIC={best_aic:.2f} and log-likelihood={best_ll:.2f}")
    print("Parameters:", best_params)
    return best_model, best_params