import numpy as np
from scipy.linalg import logm, expm
from scipy.special import binom, gammainc

# a package that performs automatic differentiation
from jax import jacfwd
import jax.numpy as jnp

import sys


def build_unitary(moved_zero, fixed_zero):
    # this process uses the SVD and stems from the solution to the
    # balanced Procrustes problem (see Wikipedia page for more details)

    # FIXME we need to handle case when moved_zero == fixed_zero separately,
    # which isn't very neat
    if np.allclose(moved_zero, fixed_zero):
        return np.eye(len(fixed_zero))

    M = fixed_zero[:, np.newaxis] @ moved_zero[np.newaxis]
    U, S, Vt = np.linalg.svd(M)
    return U @ Vt


def build_random_unitary(n):
    U, S, Vt = np.linalg.svd(np.random.rand(n,n))
    return U @ Vt


def build_path(A, N):
    # (to build this path, we are following section 3.4 in RH part 1)
    log_A = [logm(A[idx]) for idx in range(N)]

    # TODO this choice of T might be giving us the needed Lipschitz continuity?
    T = np.sqrt( (1/np.sqrt(2)) * sum([np.linalg.norm(A[idx])**2 for idx in range(N)]))
    W_t = lambda t : [expm((t/T) * A_j) for A_j in A]

    # TODO assuming T=1, which may mess up something down the line? not sure
    # W_t = lambda t : [expm((t) * A_j) for A_j in A]
    return W_t


# from Daniel's answer on
# https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
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


def proj_newton(guess, F, jac, tol=1e-10, max_iter=5000):
    # guess is a guess for where the zero is located
    # F is the polynomial system
    # jac is the Jacobian of F (should be N x (N + 1), since F has N
    # polynomials with (N + 1) variables)
    # the returned approx. zero will be within tol of the true zero
    # (if the computation succeeds, anyway)

    # N is the number of polynomials in the system F
    N = len(F(guess))

    DF_mat = lambda X : np.vstack([jac(X.T.flatten()), X.T])
    assert np.shape(DF_mat(guess)) == (N+1, N+1)

    F_mat = lambda X : np.append(np.array([func for func in F(X)]), 0)
    assert np.shape(F_mat(guess)) == (N+1,)

    next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
    err = np.linalg.norm(next_guess - guess)
    guess = next_guess
    count = 1
    give_up = 1e10

    while err > tol and err < give_up and count < max_iter:
        next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
        count += 1
        err = np.linalg.norm(next_guess - guess)
        guess = next_guess

    if count > max_iter:
        raise RuntimeError('Too many iterations.')
    if err > give_up:
        raise ValueError('Going to infinity and beyond.')

    return [guess, count]


def bounded_blackbox_NC(F,path,init_zero,K_max,eps,D):
    N = len(F)
    eta = eps/N/K_max
    g_sum = 0
    t = 1

    W_t = np.array(path(t))
    # a jnp.array of the ``shifted" polynomial system
    F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])
    # use jax (automatic differentiation) to build jacobian
    jac = jacfwd(F_t, holomorphic=True)

    zero = init_zero
    for k in range(1,K_max+1):
        if k % 100 == 0: print(f"Iteration {k} at t={t}")
        jac_norm = jnp.linalg.norm(jac(zero),axis=1)**2
        g_sum = jnp.linalg.norm(gamma_prob(F_t,zero,jac_norm,D,eta))
        fac = 1e5
        kappa = cond_num(jac(zero))
        t -= 1/(240*kappa**2*g_sum/fac)
        if k == 1:
            if (240*kappa**2*g_sum/fac) > K_max:
                raise RuntimeError(f"K_max ({K_max}) is probably too small given the initial step size ({1/(240*6*N**2*g_sum/fac)})")
        if t <= 0:
            F_t = lambda X : jnp.array([F[idx](*X) for idx in range(N)])
            jac = jacfwd(F_t, holomorphic=True)
            final_zero, _ = proj_newton(zero, F_t, jac)
            return final_zero, k
        else:
            W_t = np.array(path(t))
            F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])
            jac = jacfwd(F_t, holomorphic=True)
            # compute next_zero using projective Newton's method, with previous zero
            # as the initial guess
            zero, _ = proj_newton(zero, F_t, jac)

    raise RuntimeError(f"Bounded blackbox NC failed to converge in fewer than K_max ({K_max}) iterations.")


# FIXME handing in max degree?
def main(F, zeros, max_deg, K_max=100):
    # F is the system of polynomials we are solving
    # TODO zeros is a list of initial zeros of each poly in F (not common
    # zeros, just plain old zeros). eventually, we want the program to build
    # these initial zeros itself


    """ INITIAL CHECKS """

    arg_counts = [func.__code__.co_argcount for func in F]

    # check that each func in F has the same number of variables
    assert np.allclose(arg_counts, arg_counts[0])

    # verify that this number of variables is exactly one more
    # than the number of polynomials in the system
    N = len(F)
    num_vars = arg_counts[0]
    assert num_vars == (N + 1)


    """ BUILD START SYSTEM """

    # TODO compute initial zeros, store in list `zeros' (currently an input)

    # scaling initial zeros to have magnitude one
    zeros = [zero / np.linalg.norm(zero) for zero in zeros]

    # check that all scaled zeros are still zeros of the original
    # polynomial system (this is also an implicit check that the original
    # system is indeed homogeneous, as required)
    assert np.allclose([F[idx](*zeros[idx]) for idx in range(N)], 0)

    # now we use build_unitary to construct a unitary matrix that maps
    # each initial zero to the first in the list (i.e., zeros[0] will
    # be (almost) the common zero of our start system)
    A = np.array([build_unitary(zeros[idx], zeros[0]) for idx in range(0,N)])
    # print(A[1])
    assert np.allclose(A[0], np.eye(num_vars)) # FIXME
    assert np.allclose([A[idx] @ zeros[idx] for idx in range(N)], zeros[0])

    # to make the start system truly generic, we hit the matrix system A
    # with an arbitrary unitary matrix
    V = build_random_unitary(num_vars)
    # print(V)
    A = [V @ A[idx] for idx in range(N)]

    # check that the new polynomial system (A \cdot F) has common zero
    # V @ zeros[0]
    np.allclose([F[idx](*A[idx].T.conj() @ V @ zeros[0]) for idx in range(N)], 0)


    """ BUILD PATH """

    # now we need to build a path in \mathcal{U} that maps array
    # of num_vars by num_vars identity matrices to A

    # build path, which is a function of time that returns a 1xN vector
    # of (unitary) matrices
    path = build_path(A, N)

    # let's check that this path does what we expect

    # at t = 0, path(0) should equal a 1xN vector of num_vars by num_vars
    # identity matrices
    np.allclose(path(0), [np.eye(num_vars)]*N)

    # at t = 1, path(1) should equal the 1xN vector of matrices, A
    np.allclose(path(1), A)


    """ TRACK ZERO """

    # next we track the common zero V @ zeros[0] along this path using a
    # projective Newton's method

    # D >= maximum degree of the system, and should be a power of 2 if we want to use FFT at some point
    init_zero = np.reshape((V @ zeros[0]).astype(complex), (num_vars,))
    eps = 0.1 # FIXME ?
    D = int(2**np.ceil(np.log2(max_deg)))
    final_zero, num_iter = bounded_blackbox_NC(F,path,init_zero,K_max,eps,D)

    # if last coordinate of final_zero is far from zero, normalize so
    # that it is one
    if not np.isclose(final_zero[-1], 0):
        final_zero = final_zero / final_zero[-1]

    assert np.allclose([F[idx](*final_zero) for idx in range(N)], 0)
    # print('Zero of original system: ', final_zero)

    # TODO post-processing?
    return final_zero, num_iter


if __name__ == '__main__':
    """
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z: x**2 + y**2 - 4*z**2

    # zeros of test polynomials
    # p = np.array([1j,np.sqrt(2),1])
    # q = np.array([np.sqrt(2),np.sqrt(2),1])
    #p = np.array([1,1,np.sqrt(2)])
    # q = np.array([np.sqrt(2),np.sqrt(2),1])
    s,t = np.random.rand(2)
    p = np.array([s,t,(s**2+t**2)**(1/2)])
    s,t = np.random.rand(2)
    q = np.array([s,t,0.5*(s**2+t**2)**(1/2)])

    main([f,g],[p,q])
    """

    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 - 2*x*z + y**2
    g = lambda x, y, z : x**2 + y**2 - z**2

    # zeros of test polynomials
    p = np.array([1,1,1])
    q = np.array([1,0,1])

    final_zero, num_iter = main([g, f], [q, p], max_deg=2, K_max=1000)
    print(f"Final zero: {final_zero}")
    print(f"Converged in {num_iter} iterations")
