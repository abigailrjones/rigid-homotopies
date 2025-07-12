import numpy as np
from scipy.special import binom, gammainc
from scipy.optimize import root_scalar

# a package that performs automatic differentiation
import jax
import jax.numpy as jnp

# use double types instead of floats
jax.config.update('jax_enable_x64', True)


def compute_initial_system(F):
    N = len(F)

    # FIXME I would like the seed to change
    key = jax.random.key(2)
    keys = jax.random.split(key,N+4)

    zeta, H = sample_linear_intersection(N,keys[:N+2])

    U = []
    for idx in range(N):
        y_i = sample_Xi(F[idx], N, keys[N+2:])
        U_i = build_start_sys_unitary(F[idx], N, y_i, zeta, H[idx])
        assert np.allclose(U_i @ y_i, zeta)
        U.append(U_i)

    return U, zeta


def sample_linear_intersection(N, keys):
    H = []
    aux = jnp.array([])
    for idx in range(N):
        lambda_i = jax.random.normal(keys[idx],(1,N+1),dtype=complex)
        _,sigma,V_i = jnp.linalg.svd(lambda_i)
        assert (len(sigma) == 1)
        if idx == 0: aux = V_i[0]
        else: aux = jnp.vstack((aux,V_i[0]))
        H.append(V_i[1:])

    _,sigma,V_int = jnp.linalg.svd(aux)
    number_rows = len(V_int)
    r = len(sigma)
    random_coeffs = jax.random.normal(keys[N],number_rows-r,dtype=complex).T
    zeta = random_coeffs @ V_int[r:]

    return zeta / np.linalg.norm(zeta), H


def sample_Xi(f, N, keys):
    # FIXME this process can fail (like in the quadratic case that I originally
    # had, where the intersection never crossed the x-axis), also I had to
    # restrict to reals which might give me problems down the line (restricting
    # to reals is actually probably why the process can fail...)
    p = jax.random.normal(keys[0],(N+1))#,dtype=complex)
    q = jax.random.normal(keys[1],(N+1))#,dtype=complex)
    Y = jax.random.normal(keys[2],(1))#,dtype=complex)

    root_res = root_scalar(lambda XX : f(*(p*XX + q*Y)),x0=0.,method='newton')
    if root_res.converged: y = p*root_res.root + q*Y
    else: raise RuntimeError("Failed to sample a y_i, try again with a different seed.")

    return y / np.linalg.norm(y)


def build_start_sys_unitary(f, N, y, zeta, h):
    # compute basis for TyX
    # grad_f_y = jax.grad(f,[i for i in range(jnp.shape(y)[0])])(*y)
    grad_f_y = jnp.reshape(jax.jacrev(lambda X : f(*X))(y), (1,N+1))
    # print(grad_f_y)
    _,sigma,Vh = jnp.linalg.svd(grad_f_y)
    assert (len(sigma)==1)
    TyX = Vh[1:]

    # write y in terms of basis for TyX
    alpha = TyX @ y
    alpha = jnp.reshape(alpha, (1,N))
    alpha = alpha /jnp.linalg.norm(alpha)
    print(alpha @ TyX - y)

    # write zeta in terms of basis for h (computed previously)
    # FIXME FIXME FIXME there's an issue here
    beta = h @ zeta
    beta = jnp.reshape(beta, (1,N))
    beta = beta / jnp.linalg.norm(beta)
    print(beta @ h - zeta)

    # compute Gamma such that Gamma alpha = beta
    U,_,Vh = jnp.linalg.svd(beta.T @ alpha.conj())
    Gamma = U @ Vh
    assert jnp.allclose(Gamma @ alpha.T, beta.T)

    # compute unitary matrix that maps y to zeta (among other things)
    # (note that the transposes in the following line are NOT conjugate
    # transposes; we just want to think of the rows as columns)
    U,_,Vh = jnp.linalg.svd(h.T @ Gamma @ TyX.conj())
    """
    print(U.T.conj() @ U)
    print(Vh.T.conj() @ Vh)
    print()
    """

    return U @ Vh


# from Daniel's answer on
# https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
# FIXME Jax might have a function that does this automatically (see jax.random.ball)
def sample(center,radius,n_per_sphere):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p


def gamma_prob(F,Z,jac_norm,D,eps):
    # See Algorithm 2, part II.
    # F is a system of N polynomials, where each polynomial has N+1 variables
    # z is a vector with dimension (1,N+1).
    # jac_norm is the row-wise L2 norm of the Jacobian matrix of F evaluated at z
    # D is an upper bound of the degree of f, should be a factor of 2?
    N = np.shape(jac_norm)[0]
    H = lambda X : F(Z + X)
    s = int(np.ceil(1 + np.log2(D/eps)))
    roots_unity = np.exp([2*np.pi*1j/(D+1)**j for j in range(D+1)])
    H_sum = np.zeros((D-1,N))
    # sample unit ball in C^(N+1) uniformly
    W = sample(np.zeros(N+1),1,s)

    for j in range(1,s):
        wj = W[j]
        Hj = np.asarray([H(zeta*wj) for zeta in roots_unity])
        assert np.shape(Hj)==(D+1,N)

        H_sum += np.abs(np.vander(1/roots_unity,N=D+1,increasing=True)[:,2:].T @ Hj)**2

    fac = [(32*N*k)**k * binom(N+1+k,k) / s for k in range(2,D+1)]
    res = (fac * np.array(H_sum * jac_norm).T)**[1/(2*k-2) for k in range(2,D+1)]
    assert np.shape(res)==(N,D-1)

    return np.max(res,axis=1)


def cond_num(jac):
    # returns condition number, which is inverse of the least singular value of L
    # uses (15) and process in proof of Corollary 33 in part I.
    L = np.diag(1/np.linalg.norm(jac,axis=1)) @ jac
    return 1/np.min(np.linalg.svdvals(L))


def bounded_blackbox_NC(F,path,init_zero,K_max,eps,D):
    N = len(F)
    eta = eps/N/K_max
    g_sum = 0
    t = 1

    W_t = np.array(path(t))
    # a jnp.array of the ``shifted" polynomial system
    F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])
    # use jax (automatic differentiation) to build jacobian
    jac = jax.jacfwd(F_t, holomorphic=True)

    zero = init_zero
    for k in range(1,K_max+1):
        if k % 100 == 0: print(f"Iteration {k} at t={t}")
        jac_norm = jnp.linalg.norm(jac(zero),axis=1)**2
        g_sum = jnp.linalg.norm(gamma_prob(F_t,zero,jac_norm,D,eta))
        fac = 1e0
        kappa = cond_num(jac(zero))
        t -= 1/(240*kappa**2*g_sum/fac)
        """
        if k == 1:
            if (240*kappa**2*g_sum/fac) > K_max:
                raise RuntimeError(f"K_max ({K_max}) is probably too small given the initial step size ({1/(240*6*N**2*g_sum/fac)})")
        """
        if t <= 0:
            F_t = lambda X : jnp.array([F[idx](*X) for idx in range(N)])
            jac = jax.jacfwd(F_t, holomorphic=True)
            final_zero, _ = proj_newton(zero, F_t, jac)
            return final_zero/np.linalg.norm(final_zero), k
        else:
            W_t = np.array(path(t))
            F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])
            jac = jax.jacfwd(F_t, holomorphic=True)
            # compute next_zero using projective Newton's method, with previous zero
            # as the initial guess
            zero, _ = proj_newton(zero, F_t, jac)
            zero = zero/np.linalg.norm(zero)
            # print(t, zero[0])

    raise RuntimeError(f"Bounded blackbox NC failed to converge in fewer than K_max ({K_max}) iterations.")


    """
    # D >= maximum degree of the system, and should be a power of 2 if we want to use FFT at some point
    eps = 0.1 # FIXME ?
    D = int(2**np.ceil(np.log2(max_deg)))
    """
