from metmhn.jx import likelihood as ssr
import metmhn.jx.one_event as one
import logging 
import jax.numpy as jnp
from jax import vmap
import numpy as np
import scipy.optimize as opt
from typing import Callable
from dataclasses import dataclass


def L1(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return jnp.sum(jnp.sqrt(theta_**2 + eps))


def L1_(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return theta_.flatten() / jnp.sqrt(theta_.flatten()**2 + eps)

def L2(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Computes the L2 norm
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return jnp.sqrt(jnp.sum(theta_**2) + eps)

def L2_(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L2 norm
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return (theta_ / (jnp.sqrt(jnp.sum(theta_**2) + eps))).flatten()

def sym_penal(log_theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    n = log_theta.shape[0]
    theta_ = log_theta.at[jnp.diag_indices(n)].set(0.)
    penal = jnp.sum(jnp.sqrt(theta_**2 + theta_.T**2 - theta_ * theta_.T + eps))
    return 0.5*(penal - n*jnp.sqrt(eps))


def sym_penal_(log_theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    n = log_theta.shape[0]
    theta_ = log_theta.at[jnp.diag_indices(n)].set(0.)
    penal_denom = 2*jnp.sqrt(theta_**2 + theta_.T**2 - theta_ * theta_.T + eps)
    penal_num = 2*theta_ - theta_.T
    return (penal_num/penal_denom).flatten()


def symmetric_penal(params: np.array, n_total: int, eps=1e-05) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    penal = np.array(sym_penal(log_theta) + L1(log_d_p) + L1(log_d_m))
    penal_ = np.concatenate((sym_penal_(log_theta), L1_(log_d_p), L1_(log_d_m)))
    return penal, penal_

def symmetric_penal2(params: np.array, n_total: int, eps=1e-05):
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    
    # remove last row + column
    log_theta_sub = log_theta[:-1, :-1]

    penal = np.array(sym_penal(log_theta_sub) + L1(log_d_p) + L1(log_d_m))

    # compute gradient
    grad_theta_sub = sym_penal_(log_theta_sub)
    grad_theta_full = jnp.zeros_like(log_theta).flatten()
    idx = jnp.arange((n_total - 1)**2)
    grad_theta_full = grad_theta_full.at[idx].set(grad_theta_sub)

    penal_ = np.concatenate((grad_theta_full, L1_(log_d_p), L1_(log_d_m)))

    return penal, penal_

def symmetric_penal3(params: np.array, n_total: int, eps=1e-05):
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    
    # remove last row + column
    log_theta_sub = log_theta[:-1, :-1]

    penal = np.array(sym_penal(log_theta_sub) + L2(log_d_p) + L2(log_d_m))

    # compute gradient
    grad_theta_sub = sym_penal_(log_theta_sub)
    grad_theta_full = jnp.zeros_like(log_theta).flatten()
    idx = jnp.arange((n_total - 1)**2)
    grad_theta_full = grad_theta_full.at[idx].set(grad_theta_sub)

    penal_ = np.concatenate((grad_theta_full, L2_(log_d_p), L2_(log_d_m)))

    return penal, penal_

def symmetric_penal4(params: np.array, n_total: int, eps=1e-05):
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    
   # ---- split matrix ----
    log_theta_core = log_theta[:-1, :-1]
    log_theta_row  = log_theta[-1, :-1]   # last row
    log_theta_col  = log_theta[:-1, -1]   # last column

    # ---- penalties ----
    penal_core = sym_penal(log_theta_core)
    penal_row  = L1(log_theta_row)        # or custom function
    penal_col  = L1(log_theta_col)

    penal = np.array(penal_core + penal_row + penal_col + L1(log_d_p) + L1(log_d_m))

    # ---- gradients ----
    grad_theta = jnp.zeros_like(log_theta)

    # core gradient
    grad_core = sym_penal_(log_theta_core).reshape(log_theta_core.shape)
    grad_theta = grad_theta.at[:-1, :-1].set(grad_core)

    # row gradient (asymmetric)
    grad_row = L1_(log_theta_row).reshape(-1)
    grad_theta = grad_theta.at[-1, :-1].set(grad_row)

    # column gradient (asymmetric)
    grad_col = L1_(log_theta_col).reshape(-1)
    grad_theta = grad_theta.at[:-1, -1].set(grad_col)

    # flatten theta gradient
    grad_theta_flat = grad_theta.flatten()

    penal_ = np.concatenate((grad_theta_flat, L1_(log_d_p), L1_(log_d_m)))

    return penal, penal_

@dataclass
class Theta_extended:
    """
    this class stores and manages the data used to reparametrize the theta matrix.

    Attributes
    ----------
    theta_g : np.ndarray
        An n x n matrix - Genomic interactions
    theta_met_br : jnp.ndarray
        An m vector - base rates of seeding to the different tissues
    theta_gm : jnp.ndarray
        n vecotr - general shared metastasis seeding programm independent of locaitons
    theta_mg : jnp.ndarray
        n vector - general effects of metastasis on seeding independent of locaitons
    theta_loc_gm : jnp.ndarray
        n x m matrix - effects of the mutations in the PT on the seeding rates to the different MTs
    theta_loc_mg : jnp.ndarray
        n x m matrix - effects of the locations of the MTs on the accumulation rates of the MTs
    omega_g : jnp.ndarray
        n vector - observation effects in primary tumours.
    omega_m : jnp.ndarray
        n x m matrix - observation effects in metastasis locations .

    """

    theta_g: jnp.ndarray
    theta_met_br: jnp.ndarray
    theta_gm: jnp.ndarray
    theta_mg: jnp.ndarray
    theta_loc_gm: jnp.ndarray
    theta_loc_mg: jnp.ndarray
    omega_g: jnp.ndarray
    omega_m: jnp.ndarray


@dataclass
class Theta_extended_grad:
    """
    Dataclass to hold gradients with respect to the raw parameters in Theta_extended.
    Each field corresponds to the gradient for one parameter, computed per datapoint.
    
    Attributes
    ----------
    grad_theta_g : jnp.ndarray
        Shape (n, n), gradient wrt theta_g (genomic interactions).
    grad_theta_met_br : jnp.ndarray
        Shape (), scalar gradient wrt theta_met_br (base rate of seeding).
    grad_theta_gm : jnp.ndarray
        Shape (n,), gradient wrt theta_gm, non-zero only at location i.
    grad_theta_mg : jnp.ndarray
        Shape (n,), gradient wrt theta_mg, non-zero only at location i.
    grad_theta_loc_gm : jnp.ndarray
        Shape (n, m), gradient wrt theta_loc_gm, non-zero only at row i.
    grad_theta_loc_mg : jnp.ndarray
        Shape (n, m), gradient wrt theta_loc_mg, non-zero only at row i.
    grad_omega_g : jnp.ndarray
        Shape (n,), gradient wrt omega_g (observation effects in primary tumors).
    grad_omega_m : jnp.ndarray
        Shape (n, m), gradient wrt omega_m (observation effects in metastases).
    """
    grad_theta_g: jnp.ndarray
    grad_theta_met_br: jnp.ndarray
    grad_theta_gm: jnp.ndarray
    grad_theta_mg: jnp.ndarray
    grad_theta_loc_gm: jnp.ndarray
    grad_theta_loc_mg: jnp.ndarray
    grad_omega_g: jnp.ndarray
    grad_omega_m: jnp.ndarray


def create_Theta_extended(
        theta_g: jnp.ndarray,
        theta_met_br: jnp.ndarray,
        theta_gm: jnp.ndarray,
        theta_mg: jnp.ndarray,
        theta_loc_gm: jnp.ndarray,
        theta_loc_mg: jnp.ndarray,
        omega_m: jnp.ndarray,
        omega_g: jnp.ndarray
        ) -> Theta_extended:

    # --- Validation ---
    n = theta_g.shape[0]
    m = theta_loc_gm.shape[1]

    if theta_g.ndim != 2 or theta_g.shape[0] != theta_g.shape[1]:
        raise ValueError("theta_g must be a square matrix of shape (n, n)")

    for vec, name in zip(
        [theta_gm, theta_mg, omega_g],
        ["theta_gm", "theta_mg", "omega_g"]
    ):
        if vec.ndim != 1 or vec.shape[0] != n:
            raise ValueError(f"{name} must be a vector of size {n}")

    if theta_met_br.ndim != 1:
        raise ValueError("theta_met_br must be a vector")
    if theta_met_br.shape[0] != m:
        raise ValueError(f"theta_met_br must have size {m}")

    if omega_m.ndim != 2:
        raise ValueError("omega_m must be a matrix")
    if omega_m.shape != (n, m):
        raise ValueError(f"omega_m must have shape ({n}, {m})")

    if theta_loc_gm.ndim != 2 or theta_loc_gm.shape[0] != n:
        raise ValueError(f"theta_loc_gm must be a matrix with {n} rows")

    if theta_loc_mg.ndim != 2 or theta_loc_mg.shape[0] != n:
        raise ValueError(f"theta_loc_mg must be a matrix with {n} rows")

    if theta_loc_gm.shape[1] != theta_loc_mg.shape[1]:
        raise ValueError("theta_loc_gm and theta_loc_mg must have the same number of columns")

    return Theta_extended(
        theta_g=theta_g,
        theta_met_br=theta_met_br,
        theta_gm=theta_gm,
        theta_mg=theta_mg,
        theta_loc_gm=theta_loc_gm,
        theta_loc_mg=theta_loc_mg,
        omega_g=omega_g,
        omega_m=omega_m,
    )


def create_Theta_extended_from_flat_params(params: jnp.ndarray, n_total: int, m: int = 1) -> Theta_extended:
    """
    Create a Theta_extended object from a flat parameter vector for the current single-location case.

    This is a backward-compatible helper used by score_and_grad_reg when the
    optimization parameter vector still contains only log_theta, log_d_p, and log_d_m.
    """
    n_mut = n_total - 1

    idx = 0

    theta_g = params[idx:idx+n_mut**2].reshape((n_mut, n_mut))
    idx += n_mut**2

    theta_met_br = params[idx:idx+1]
    idx += 1

    theta_gm = params[idx:idx+n_mut]
    idx += n_mut

    theta_mg = params[idx:idx+n_mut]
    idx += n_mut

    omega_g = params[idx:idx+n_total]
    idx += n_total

    omega_m = params[idx:idx+n_total*m].reshape((n_total, m))
    idx += n_total * m

    theta_loc_gm = jnp.ones((n_mut, m))
    theta_loc_mg = jnp.ones((n_mut, m))

    return Theta_extended(
        theta_g=theta_g,
        theta_met_br=theta_met_br,
        theta_gm=theta_gm,
        theta_mg=theta_mg,
        theta_loc_gm=theta_loc_gm,
        theta_loc_mg=theta_loc_mg,
        omega_g=omega_g,
        omega_m=omega_m,
    )


def reparametrization(theta_extended: Theta_extended) -> jnp.ndarray:
    """
    Reparameterize the full theta matrix for every metastasis location.

    The genomic effect block (upper-left (n-1)x(n-1)) remains unchanged.
    The last row and last column are scaled location-wise by the
    corresponding columns of theta_loc_gm and theta_loc_mg.

    Args:
        theta_extended (Theta_extended): Extended theta data.

    Returns:
        jnp.ndarray: Reparameterized theta matrices with shape (m, n, n),
                    where m is the number of metastasis locations.
    """
    n_mut = theta_extended.theta_g.shape[0]
    m = theta_extended.theta_loc_gm.shape[1]

    theta_rep = jnp.zeros((m, n_mut+1, n_mut+1))

    # genomic block
    theta_rep = theta_rep.at[:, :-1, :-1].set(theta_extended.theta_g)

    # metastasis row
    scaled_row = (
        theta_extended.theta_loc_gm.T * theta_extended.theta_gm[None, :]
    )

    theta_rep = theta_rep.at[:, -1, :-1].set(scaled_row)
    
    # metastasis column
    scaled_col = (
        theta_extended.theta_loc_mg.T * theta_extended.theta_mg[None, :]
    )

    theta_rep = theta_rep.at[:, :-1, -1].set(scaled_col)

    # bottom-right
    theta_rep = theta_rep.at[:, -1, -1].set(theta_extended.theta_met_br)

    return theta_rep

def reparametrization_(theta_extended: Theta_extended, grad_rep: jnp.ndarray) -> jnp.ndarray:
    """
    Map gradients on the reparameterized matrices back to the original theta.

    Args:
        theta_extended (Theta_extended): Extended theta data.
        grad_rep (jnp.ndarray): Gradient tensor on the reparameterized matrices
                               with shape (m, n, n).

    Returns:
        jnp.ndarray: Gradient with respect to the original theta matrix.
    """
    n_mut = theta_extended.theta_g.shape[0]
    n_total = n_mut + 1
    m = theta_extended.theta_loc_gm.shape[1]

    if grad_rep.shape != (m, n_total, n_total):
        raise ValueError(f"grad_rep must have shape {(m, n_total, n_total)}, got {grad_rep.shape}")

    grad_theta = jnp.zeros((n_total, n_total))

    # top-left block is copied unchanged for every location
    grad_theta = grad_theta.at[:-1, :-1].set(jnp.sum(grad_rep[:, :-1, :-1], axis=0))

    # last row entries depend on the original last row scaled by theta_loc_gm
    grad_theta = grad_theta.at[-1, :-1].set(jnp.sum(grad_rep[:, -1, :-1] * theta_extended.theta_loc_gm.T, axis=0))

    # last column entries depend on the original last column scaled by theta_loc_mg
    grad_theta = grad_theta.at[:-1, -1].set(jnp.sum(grad_rep[:, :-1, -1] * theta_extended.theta_loc_mg.T, axis=0))

    # bottom-right element is shared across every location matrix
    grad_theta = grad_theta.at[-1, -1].set(jnp.sum(grad_rep[:, -1, -1]))

    return grad_theta

def reparametrization_to_raw_params(
    theta_extended: Theta_extended,
    grad_rep: jnp.ndarray,
    met_loc: int,
    grad_d_p: jnp.ndarray,
    grad_d_m: jnp.ndarray
) -> Theta_extended_grad:
    """
    Map a datapoint gradient on a location-specific reparameterized theta matrix
    back to the raw Theta_extended parameters.

    Args:
        theta_extended (Theta_extended): Extended theta data.
        grad_rep (jnp.ndarray): Gradient on the reparameterized matrix for one location, shape (n+1, n+1).
        met_loc (int): Metastasis location index for this datapoint.
        grad_d_p (jnp.ndarray): Gradient wrt observation effects in primary tumor, shape (n+1,).
        grad_d_m (jnp.ndarray): Gradient wrt observation effects in metastasis, shape (n+1,).

    Returns:
        Theta_extended_grad: Gradients for all raw parameters.
    """
    n_mut = theta_extended.theta_g.shape[0]
    n_total = n_mut + 1
    m = theta_extended.theta_loc_gm.shape[1]

    grad_theta_g = grad_rep[:-1, :-1]
    grad_theta_met_br = grad_rep[-1, -1]

    grad_last_row = grad_rep[-1, :-1]
    grad_last_col = grad_rep[:-1, -1]

    grad_theta_gm = (theta_extended.theta_loc_gm[:, met_loc] * grad_last_row)
    grad_theta_mg = (theta_extended.theta_loc_mg[:, met_loc] * grad_last_col)

    grad_theta_loc_gm = jnp.zeros((n_mut, m))
    grad_theta_loc_gm = grad_theta_loc_gm.at[:, met_loc].set(theta_extended.theta_gm * grad_last_row)

    grad_theta_loc_mg = jnp.zeros((n_mut, m))
    grad_theta_loc_mg = grad_theta_loc_mg.at[:, met_loc].set(theta_extended.theta_mg * grad_last_col)

    grad_omega_g = grad_d_p
    grad_omega_m = jnp.zeros((n_total, m))
    grad_omega_m = grad_omega_m.at[:, met_loc].set(grad_d_m)

    assert grad_d_m.shape == (n_total,)
    assert grad_omega_m.shape == (n_total, m)

    assert grad_theta_g.shape == theta_extended.theta_g.shape
    assert grad_theta_gm.shape == theta_extended.theta_gm.shape
    assert grad_theta_mg.shape == theta_extended.theta_mg.shape

    assert grad_theta_loc_gm.shape == theta_extended.theta_loc_gm.shape
    assert grad_theta_loc_mg.shape == theta_extended.theta_loc_mg.shape

    assert theta_extended.omega_m.shape == (n_total, m)

    return Theta_extended_grad(
        grad_theta_g=grad_theta_g,
        grad_theta_met_br=grad_theta_met_br,
        grad_theta_gm=grad_theta_gm,
        grad_theta_mg=grad_theta_mg,
        grad_theta_loc_gm=grad_theta_loc_gm,
        grad_theta_loc_mg=grad_theta_loc_mg,
        grad_omega_g=grad_omega_g,
        grad_omega_m=grad_omega_m,
    )

def composite_penalty(theta_extended: Theta_extended, lam: float, c: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the composite penalty for the model parameters theta.

    Args:
        theta_extended (Theta_extended): An instance of the Theta_extended class containing the theta and additional data.
        lam (float): Lambda tuning parameter for the composite penalty.
        c (jnp.ndarray): A vector of coefficients for the composite penalty.

    Returns:
        jnp.ndarray: penalty term for the composite theta matrix.
    """
    n = theta_extended.theta_g.shape[0]

    # --- composite penalty ---
    theta = theta_extended.theta_g

    composite_penalty_value = 0

    for i in range(theta_extended.theta_loc_gm.shape[1]):
        composite_penalty_value += (
            lam * c[i] * L1(theta_extended.theta_gm) +
            lam * c[i] * L1(theta_extended.theta_mg) +
            (1 - c[i]) * L1(theta_extended.theta_loc_gm[:, i]) +
            (1 - c[i]) * L1(theta_extended.theta_loc_mg[:, i])
        )
        # Note: the first two terms penalize the last row and column of the theta matrix.
        # The last two terms penalize the location parameters, which are proxies for the seeding effects.

    composite_penalty_value += (
        lam * (L1(theta) + L1(theta_extended.omega_m) + L1(theta_extended.omega_g)) +
        (1 - lam) * (
            L1(theta_extended.theta_loc_gm @ theta_extended.theta_loc_gm.T) +
            L1(theta_extended.theta_loc_mg @ theta_extended.theta_loc_mg.T)
        )
    )
    # Note: the second part penalizes the base genomic block and observation effects,
    # while the Gram-style term controls the size of the location parameters.

    return composite_penalty_value

def composite_penalty_(theta_extended: Theta_extended, lam: float, c: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the gradient of the composite penalty with respect to the theta matrix.

    Args:
        theta_extended (Theta_extended): An instance of the Theta_extended class containing the theta and additional data for reparametrization.
        lam (float): Lambda parameter for reparametrization.
        c (jnp.ndarray): A vector of coefficients for the composite penalty.

    Returns:
        jnp.ndarray: Gradient of the penalty term with respect to the theta matrix.
    """
    n = theta_extended.theta_g.shape[0]

    # --- composite penalty gradient ---
    theta = theta_extended.theta_g

    # Gradient from lam * L1(theta) in top-left block
    grad_theta_main = lam * L1_(theta).reshape(theta.shape)

    # Gradient from loop: lam * sum(c[i]) * L1_(theta_g[-1, :-1]) in last row
    grad_last_row = lam * jnp.sum(c) * L1_(theta_extended.theta_g[-1, :-1])
    # Gradient from loop: lam * sum(c[i]) * L1_(theta_g[:-1, -1]) in last column
    grad_last_col = lam * jnp.sum(c) * L1_(theta_extended.theta_g[:-1, -1])

    # Initialize full gradient matrix
    grad_full = jnp.zeros_like(theta_extended.theta_g)
    grad_full = grad_full.at[:-1, :-1].set(grad_theta_main)
    grad_full = grad_full.at[-1, :-1].set(grad_last_row)
    grad_full = grad_full.at[:-1, -1].set(grad_last_col)

    return grad_full

def score(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, dat: jnp.ndarray, 
          perc_met: float)-> jnp.ndarray:
    """Calculates the log. likelihood of the dataset dat

    Args:
        log_theta (jnp.ndarray): (n+1)x(n+1)-dimensional Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the PT on the rate of its observation event
        log_d_m (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the MT on the rate of its observation event
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset.
    
    Returns:
        jnp.ndarray: Log. likelihood
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    score, score_pt = 0., 0.
    for i in range(dat.shape[0]):
        if dat[i,-1] == 0:
            # Never metastasizing primary tumors
            state_obs = dat[i, 0:2*n_total-1:2]
            n_prim = int(state_obs.sum())
            if n_prim == 0:
                score_pt += ssr._lp_prim_obs_az(log_theta)
            else:
                score_pt += ssr._lp_prim_obs(log_theta, log_d_p, state_obs, n_prim)
        else:
            if dat[i,-1] == 1:
            # Metastasized primary tumors without sequenced metastasis
                state_obs = dat[i, 0:2*n_total-1:2]
                n_prim = int(state_obs.sum())      
                score += ssr._lp_prim_obs(log_theta, log_d_p, state_obs, n_prim)
            elif dat[i, -1] == 2:
                # Metastates without sequenced primary tumor
                state_obs = dat[i, 0:2*n_total-1]
                state_met = jnp.append(state_obs[1:2*n_total-1:2], 1)
                n_met = int(state_met.sum())
                score += ssr._lp_met_obs(log_theta, log_d_p, log_d_m, state_met, n_met)
            elif dat[i, -1] == 3:
                # Paired primary tumor and metastasis observation
                state_obs = dat[i, 0:2*n_mut+1]
                n_prim = int(state_obs[::2].sum())
                n_met = int(state_obs[1::2].sum() + 1)
                order = dat[i,-2]
                if order == 0:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_0(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_0(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)
                elif order == 1:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_1(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_1(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)
                else:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_2(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_2(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)

    n_em = jnp.sum(dat[:,-3])
    n_nm = dat.shape[0] - n_em
    # Weight MTs relative to PTs to achieve the prespecified ratio perc_met 
    if n_em*n_nm != 0:
        w = perc_met * n_nm/((1-perc_met)*n_em)
    else:
        w = 1
    n_full = w*n_em + n_nm
    score = (w*score + score_pt)/n_full
    return score


def score_reg(params: np.ndarray, dat: jnp.ndarray, perc_met: float, penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], 
              w_penal: float) -> np.ndarray:
    """Calculates the negative log. likelihood and its gradient of the dataset dat with regularization penal

    Args:
        params (np.ndarray): (n+1)*(n+2)-dimensional vecor of parameters, the first (n+1)**2 entries correspond to log. Theta, 
            the next (n+1) to log_d_p and the last (n+1)-entries to log_d_m
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndaray]]): Penalization function, should take a parametervector params and total number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        w_penal (float): weight of the penalization

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative penalized log. likelihood, grad wrt. to all model parameters
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    # Transfer parameters to the device
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    sc = score(log_theta, log_d_p, log_d_m, dat, perc_met)
    pen, _ = penal(params, n_total)
    return np.array(-sc + w_penal*pen)


def score_and_grad(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, dat: jnp.ndarray, 
                   perc_met: float, theta_extended: Theta_extended, fixed_grad: bool = False)->tuple[jnp.ndarray, list[Theta_extended_grad]]:
    """Calculates the log. likelihood and its gradient of the dataset dat

    Args:
        log_d_p (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the PT on the rate of its observation event
        log_d_m (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the MT on the rate of its observation event
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset.
        theta_extended (Theta_extended): The Theta_extended object containing the raw parameters for shape information.
        fixed_grad (bool): Whether to fix the gradient to 0 for the d_p and d_m.

    Returns:
        tuple[np.array, list[Theta_extended_grad]]: Log. likelihood, list of Theta_extended_grad objects (one per datapoint) with gradients wrt raw parameters
    """
    theta_rep = reparametrization(theta_extended)[0]

    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1

    score, score_pt = 0., 0.
    d_th, d_th_pt = jnp.zeros((n_total, n_total)), jnp.zeros((n_total, n_total))
    d_d_p, d_d_p_pt = jnp.zeros(n_total), jnp.zeros(n_total) 
    d_d_m = jnp.zeros(n_total)

    grad_rep_list = []
    grad_dp_list = []
    grad_dm_list = []
    is_met_list = []

    # Never metastasizing primary tumors
    dat_po = dat[dat[:,-1]==0,:]
    n_active = jnp.unique(dat_po[:,:-2:2].sum(axis=1))
    for i in n_active:
        tmp = dat_po[dat_po[:,:-2:2].sum(axis=1)==i, :-2:2]
        if i == 0:
            n_az = tmp.shape[0]
            lik, th_, dp_ = ssr._grad_prim_obs_az(theta_rep)
            for j in range(n_az):
                grad_rep_list.append(th_)
                grad_dp_list.append(dp_)
                grad_dm_list.append(jnp.zeros_like(log_d_m))
                is_met_list.append(False)
            score_pt += n_az * lik
            d_th_pt += n_az * th_
            d_d_p_pt += n_az * dp_
        else:
            lik, th_, dp_ = vmap(ssr._grad_prim_obs, (None, None, 0, None), out_axes=(0))(theta_rep, log_d_p, tmp, int(i))
            for j in range(len(lik)):
                grad_rep_list.append(th_[j])
                grad_dp_list.append(dp_[j])
                grad_dm_list.append(jnp.zeros_like(log_d_m))
                is_met_list.append(False)
            score_pt += lik.sum()
            d_th_pt += th_.sum(axis=0)
            d_d_p_pt += dp_.sum(axis=0)

    # Metastasized primary tumors
    dat_pm = dat[dat[:,-1]==1,:]
    n_active = jnp.unique(dat_pm[:,:-2:2].sum(axis=1))
    for i in n_active:
        tmp = dat_pm[dat_pm[:,:-2:2].sum(axis=1)==i, :-2:2]
        lik, th_, dp_ = vmap(ssr._grad_prim_obs, (None, None, 0, None), out_axes=(0))(theta_rep, log_d_p, tmp, int(i))
        for j in range(len(lik)):
            grad_rep_list.append(th_[j])
            grad_dp_list.append(dp_[j])
            grad_dm_list.append(jnp.zeros_like(log_d_m))
            is_met_list.append(True)
        score += lik.sum()
        d_th += th_.sum(axis=0)
        d_d_p += dp_.sum(axis=0)
    
    # Metastases
    dat_m = dat[dat[:,-1]==2,:]
    n_active = jnp.unique(dat_m[:,1:-2:2].sum(axis=1)) + 1
    for i in n_active:
        tmp = dat_m[dat_m[:,1:-2:2].sum(axis=1)+1==i, 1:-2:2]
        tmp = jnp.hstack((tmp, jnp.ones(tmp.shape[0], dtype=jnp.int8).reshape(-1,1)))
        lik, th_, dp_, dm_ = vmap(ssr._grad_met_obs, (None, None, None, 0, None), out_axes=(0))(theta_rep, log_d_p, log_d_m, tmp, int(i))
        for j in range(len(lik)):
            grad_rep_list.append(th_[j])
            grad_dp_list.append(dp_[j])
            grad_dm_list.append(dm_[j])
            is_met_list.append(True)
        score += lik.sum()
        d_th += th_.sum(axis=0)
        d_d_p += dp_.sum(axis=0)
        d_d_m += dm_.sum(axis=0)
    
    # Paired primary tumors and metastases
    dat_c = dat[dat[:,-1]==3,:]
    for i in range(dat_c.shape[0]):
        state_obs = dat_c[i, 0:2*n_mut+1]
        n_prim = int(state_obs[::2].sum())
        n_met = int(state_obs[1::2].sum() + 1)
        order = dat_c[i,-2]
        if order == 0:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_0(theta_rep, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_0(theta_rep, log_d_p, log_d_m, state_obs,
                                                        n_prim, n_met)
        elif order == 1:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_1(theta_rep, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_1(theta_rep, log_d_p, log_d_m, state_obs,
                                                        n_prim, n_met)
        else:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_2(theta_rep, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_2(theta_rep, log_d_p, log_d_m, state_obs,
                                                      n_prim, n_met)
        grad_rep_list.append(th_)
        grad_dp_list.append(d_p_)
        grad_dm_list.append(d_m_)
        is_met_list.append(True)
        score += s
        d_th += th_
        d_d_p += d_p_
        d_d_m += d_m_

    n_em = jnp.sum(dat[:,-3])
    n_nm = dat.shape[0] - n_em

    if n_em*n_nm != 0:
        w = perc_met * n_nm/((1-perc_met)*n_em)
    else:
        w = 1

    n_full = w*n_em + n_nm
    score = (w*score + score_pt)/n_full

    d_th = (w*d_th + d_th_pt)/n_full
    d_d_p = (w*d_d_p + d_d_p_pt)/n_full
    d_d_m = w*d_d_m/n_full

    if fixed_grad:
        d_d_p = d_d_p.at[-1].set(0.0)
        d_d_m = d_d_m.at[-1].set(0.0)

    grad_list = []
    # No explicit per-datapoint location labels are available in dat.
    # The location-specific effects are still supported in Theta_extended via the m vectors.
    met_loc = 0
    for rep, dp, dm, is_met in zip(grad_rep_list, grad_dp_list, grad_dm_list, is_met_list):
        raw_grad = reparametrization_to_raw_params(theta_extended, rep, met_loc, dp, dm)
        factor = w / n_full if is_met else 1 / n_full
        raw_grad = Theta_extended_grad(
            grad_theta_g=raw_grad.grad_theta_g * factor,
            grad_theta_met_br=raw_grad.grad_theta_met_br * factor,
            grad_theta_gm=raw_grad.grad_theta_gm * factor,
            grad_theta_mg=raw_grad.grad_theta_mg * factor,
            grad_theta_loc_gm=raw_grad.grad_theta_loc_gm * factor,
            grad_theta_loc_mg=raw_grad.grad_theta_loc_mg * factor,
            grad_omega_g=raw_grad.grad_omega_g * factor,
            grad_omega_m=raw_grad.grad_omega_m * factor,
        )
        if fixed_grad:
            raw_grad = Theta_extended_grad(
                grad_theta_g=raw_grad.grad_theta_g,
                grad_theta_met_br=raw_grad.grad_theta_met_br,
                grad_theta_gm=raw_grad.grad_theta_gm,
                grad_theta_mg=raw_grad.grad_theta_mg,
                grad_theta_loc_gm=raw_grad.grad_theta_loc_gm,
                grad_theta_loc_mg=raw_grad.grad_theta_loc_mg,
                grad_omega_g=raw_grad.grad_omega_g.at[-1].set(0.0),
                grad_omega_m=raw_grad.grad_omega_m.at[:, -1].set(0.0),
            )
        grad_list.append(raw_grad)

    return score, grad_list


def score_and_grad_reg(params: np.ndarray, dat: jnp.ndarray, perc_met: float, penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], 
                       w_penal: float, fixed_grad: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the negative log. likelihood and its gradient of the dataset dat with regularization penal

    Args:
        params (np.ndarray): (n+1)*(n+2)-dimensional vecor of parameters, the first (n+1)**2 entries correspond to log. Theta, 
            the next (n+1) to log_d_p and the last (n+1)-entries to log_d_m
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]): Penalty function, should take parametervector params and totoal number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        w_penal (float): weight of the penalization
        fixed_grad (bool): Whether to fix the gradient to 0 for the d_p and d_m.

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative penalized log. likelihood, grad wrt. to all model parameters (raw parameters)
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    # Transfer parameters to the device
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    theta_extended = create_Theta_extended_from_flat_params(params, n_total)
    score, grad_list = score_and_grad(log_d_p, log_d_m, dat, perc_met, theta_extended, fixed_grad)
    # Sum the gradients
    grad_theta_g_total = sum(g.grad_theta_g for g in grad_list)
    grad_theta_met_br_total = sum(g.grad_theta_met_br for g in grad_list)
    grad_theta_gm_total = sum(g.grad_theta_gm for g in grad_list)
    grad_theta_mg_total = sum(g.grad_theta_mg for g in grad_list)
    grad_theta_loc_gm_total = sum(g.grad_theta_loc_gm for g in grad_list)
    grad_theta_loc_mg_total = sum(g.grad_theta_loc_mg for g in grad_list)
    grad_omega_g_total = sum(g.grad_omega_g for g in grad_list)
    grad_omega_m_total = sum(g.grad_omega_m for g in grad_list)
    grad_vec = jnp.concatenate([
        grad_theta_g_total.flatten(),
        jnp.atleast_1d(grad_theta_met_br_total),
        grad_theta_gm_total.flatten(),
        grad_theta_mg_total.flatten(),
        grad_omega_g_total,
        grad_omega_m_total[:, 0],
])
    pen, pen_ = penal(params, n_total)
    return np.array(-score + w_penal*pen), -grad_vec + w_penal*pen_ 


def learn_mhn(th_init: jnp.ndarray, dp_init: jnp.ndarray, dm_init: jnp.ndarray, dat: jnp.ndarray, perc_met: float, 
              penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], w_penal: float, fixed_grad: bool = False, opt_iter: int=1e05, opt_ftol: float=1e-04, 
              opt_v: bool=True) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Infer a metMHN from data

    Args:
        th_init (jnp.ndarray): Initial estimate for the log-theta matrix. Matrix of dimension (n_muts+1) x (n_muts+1)
        dp_init (jnp.ndarray): Initial estimate for the effects of muts on PT-observation. Vector of size n_muts+1
        dm_init (jnp.ndarray): Inital estimate for the effects of muts on MT-observation. Vector of size n_muts+1 
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float):  Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]): Penalty function, should take parametervector params and totoal number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        penal (float): Weight of the penalty
        fixed_grad (bool): Whether to fix the gradient to 0 for the d_p and d_m.
        opt_iter (int): Maximal number of iterations for optimizer. Defaults to 1e05
        opt_ftol (float): Tolerance for optimizer. Defaults to 1e-04
        opt_v (bool):  Print out optimizer progress. Defaults to TRUE
        

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Estimated log. theta, log. d_p, log. d_m
    """
    n_total = th_init.shape[0]
    theta_g = th_init[:-1, :-1]
    theta_met_br = jnp.array([th_init[-1, -1]])
    theta_gm = th_init[-1, :-1]
    theta_mg = th_init[:-1, -1]

    start_params = np.concatenate([
        theta_g.flatten(),
        theta_met_br,
        theta_gm,
        theta_mg,
        dp_init,
        dm_init
    ])
    x = opt.minimize(fun=score_and_grad_reg, jac=True, x0=start_params, method="L-BFGS-B",  
                     args=(dat, perc_met, penal, w_penal, fixed_grad), 
                     options={"maxiter":opt_iter, "disp": opt_v, "ftol": opt_ftol})
    theta_extended = create_Theta_extended_from_flat_params(x.x, n_total)
    theta = reparametrization(theta_extended)[0]
    d_p = theta_extended.omega_g
    d_m = theta_extended.omega_m[:, 0]

    return theta, d_p, d_m