import numpy as np
from scipy.optimize import minimize


# We compute psi with parameters
def compute_psi_acd(params, x, p, q):
    omega = params[0] # omega coeff
    alphas = params[1:1 + p] # alpha coeffs
    betas = params[1 + p:1 + p + q] # beta coeffs
    T = len(x)
    psi = np.zeros(T) # Initialize psi to an array of 0
    psi[0] = np.mean(x)  # First estimate of conditional expectation
    for t in range(1, T):
        val_alpha = sum(alphas[i] * x[t - i - 1] for i in range(p) if t - i - 1 >= 0)
        val_beta = sum(betas[j] * psi[t - j - 1] for j in range(q) if t - j - 1 >= 0)
        psi[t] = max(omega + val_alpha + val_beta, 1e-6)

    return psi # Return psi values for all t-i>=0


# We try to minimize the negative log likelihood
def neg_log_likelihood_acd(params, x, p, q):
    omega = params[0]
    alphas = params[1:1 + p]
    betas = params[1 + p:1 + p + q]

    # Constraints: positive and sum of alpha and beta <=1 otherwise returns a large value (1e10) to force the optimizer to avoid this region.
    if omega <= 0 or np.any(np.array(alphas) < 0) or np.any(np.array(betas) < 0) or (np.sum(alphas) + np.sum(betas) >=1):
        return 1e10

    psi = compute_psi_acd(params, x, p, q)
    if np.any(psi <= 0) or np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        return 1e10
    eps = x / psi # Error calculation

    # If one error is <=0 or NaN, the model isn't valid
    if np.any(eps <= 0) or np.any(np.isnan(eps)):
        return 1e10

    # Calculation of log likelihood
    ll = -np.sum(np.log(psi) + eps)
    return -ll # Return negative log likelihood


def select_best_acd_model(x, p_max=3, q_max=3):
    best_aic = np.inf
    best_model = None
    best_params = None
    best_ll = -np.inf
    T = len(x)

    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            # Init params: omega small, moderate alphas, conservative betas
            init_params = [np.mean(x) * 0.1] + [0.1] * p + [0.8] * q
            # Parameters bounds (omega>0 and p,q >=0 & <1)
            bounds = [(1e-6, None)] + [(0, 0.999)] * p + [(0,0.999)] * q

            # Use minimize function to minimize the negative likelihood function by varying parameters
            res = minimize(neg_log_likelihood_acd, init_params, args=(x, p, q), method='L-BFGS-B', bounds=bounds)

            # If minimize is a success
            if res.success:
                k = len(init_params)
                ll = -res.fun
                aic = 2 * k - 2 * ll
                # or bic = np.log(n)*k - 2*ll

                if aic < best_aic:
                    best_aic = aic
                    best_model = (p, q)
                    best_params = res.x
                    best_ll = ll

    print(f"Best ACD model: p={best_model[0]}, q={best_model[1]}, with AIC={best_aic:.2f} and log-likelihood={best_ll:.2f}")
    print("Parameters:", best_params)
    return best_model, best_params

