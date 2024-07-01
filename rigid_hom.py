import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# a package that performs automatic differentiation
# FIXME version 0.4.17 of jax/jaxlib may be new enough to not have that
# obnoxious GPU/TPU warning, but for now, we'll just deal with it
import jax
import jax.numpy as jnp

# need matrix exponentials and matrix logarithms (which are concepts
# that, according to Wikipedia, lead to Lie theory; intriguing)
from scipy.linalg import logm, expm


def build_unitary(moved_zero, fixed_zero):
    # this process uses the SVD and stems from the solution to the
    # balanced Procrustes problem (see Wikipedia page for more details)

    # need to handle case when moved_zero == fixed_zero separately FIXME
    if np.allclose(moved_zero, fixed_zero):
        return np.eye(len(fixed_zero))

    M = fixed_zero[:, np.newaxis] @ moved_zero[np.newaxis]
    U, S, Vt = np.linalg.svd(M)
    return U @ Vt


def get_random_unitary(n):
    U, S, Vt = np.linalg.svd(np.random.rand(n,n))
    return U @ Vt


def get_path(A, T):
    W_t = lambda t : [expm((t/T) * A_j) for A_j in A]
    return W_t


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
        # FIXME relying on guess still having the right shape to input into DF_mat,
        # namely (num_vars,)
        # FIXME also, possibly a finite precision problem?
        next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
        count += 1
        # FIXME is this somewhere where finite precision could be an issue?
        err = np.linalg.norm(next_guess - guess)
        guess = next_guess

    if count >= max_iter:
        raise Exception('Too many iterations.')
    if err >= give_up:
        raise Exception('Going to infinity and beyond.')

    return [guess, count]


def main(F, zeros):
    # F is the system of polynomials we are solving
    # zeros is a list of initial zeros of each poly in F (not common zeros, just plain old zeros) FIXME


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

    # TODO compute initial zeros, store in list list `zeros' (currently an input)

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
    assert np.allclose(A[0], np.eye(num_vars)) # FIXME

    assert np.allclose([A[idx] @ zeros[idx] for idx in range(N)], zeros[0])

    # to make the start system truly generic, we hit the matrix system A
    # with an arbitrary unitary matrix
    V = get_random_unitary(num_vars)
    A = [V @ A[idx] for idx in range(N)]

    # check that the new polynomial system (A \cdot F) has common zero
    # V @ zeros[0]
    np.allclose([F[idx](*A[idx].T.conj() @ V @ zeros[0]) for idx in range(N)], 0)


    """ BUILD PATH """

    # now we need to build a path in \mathcal{U} that maps A \cdot F to array
    # of num_vars by num_vars identity matrices

    # (to do this, we are following section 3.4 in RH part 1)
    log_A = [logm(A[idx]) for idx in range(N)]

    # this choice of T might be giving us the needed Lipschitz continuity? TODO
    T = np.sqrt( (1/np.sqrt(2)) * sum([np.linalg.norm(A[idx])**2 for idx in range(N)]))

    # build path, which is a function of time that returns a 1xN vector
    # of (unitary) matrices
    path = get_path(log_A, T)

    # let's check that this path does what we expect

    # at t = 0, path(t) should equal a 1xN vector of num_vars by num_vars
    # identity matrices
    np.allclose(path(0), [np.eye(num_vars)]*N)

    # at t = T, path(t) should equal the 1xN vector of matrices, A
    np.allclose(path(T), A)


    """ TRACK ZERO """

    # next we track the common zero V @ zeros[0] along this path using a
    # projective Newton's method

    # FIXME we need to figure out how to set num_steps to guarantee convergence to
    # some precision
    num_steps = 50

    # need to cast to complex type for jax
    # TODO I'm surprised it isn't throwing a fit that it isn't a jnp array?
    # FIXME this reshape is obnoxious. Full stop.
    next_zero = np.reshape((V @ zeros[0]).astype(complex), (num_vars,))
    times = np.linspace(T, 0, num_steps)
    for t in times:
        # pick out the path matrix at time t
        W_t = np.array(path(t))
        # a jnp.array of the ``shifted" polynomial system (note that F_t must be
        # a jnp array to use automatic differentiation in jax)
        # note that the elements of W_t do NOT need to be jnp arrays because we
        # unpackage their elements in the definition of F_t, so that they are
        # simply numbers
        F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])

        # since F_t is just a system of shifted polynomials, each polynomial
        # is certainly holomorphic, which jax needs to know since we are possibly
        # inputting complex values in proj_newton
        jac = jax.jacfwd(F_t, holomorphic=True)

        next_zero, _ = proj_newton(next_zero, F_t, jac)

    # the final zero we find (at t = 0) is the common zero of our original system;
    # let's check this
    final_zero = next_zero
    assert np.allclose([F[idx](*final_zero) for idx in range(N)], 0)
    print('Zero of original system: ', final_zero)

    return final_zero


if __name__ == '__main__':
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z : x + y - z

    # FIXME
    # zeros of test polynomials
    p = np.array([1,1,np.sqrt(2)])
    q = np.array([1,1,2])

    main([g, f], [q, p])

