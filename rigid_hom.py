import numpy as np
from scipy.linalg import logm, expm

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
    # T = np.sqrt( (1/np.sqrt(2)) * sum([np.linalg.norm(A[idx])**2 for idx in range(N)]))
    # W_t = lambda t : [expm((t/T) * A_j) for A_j in A]

    # TODO assuming T=1, which may mess up something down the line? not sure
    W_t = lambda t : [expm((t) * A_j) for A_j in A]
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
        next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
        count += 1
        err = np.linalg.norm(next_guess - guess)
        guess = next_guess

    if count > max_iter:
        raise RuntimeError('Too many iterations.')
    if err > give_up:
        raise ValueError('Going to infinity and beyond.')

    return [guess, count]


def main(F, zeros):
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

    # TODO implement GammaProb, or some other timestepping solution
    t = 1
    num_steps = 100
    dt = (1/num_steps)
    max_reverse = 4
    next_zero = np.reshape((V @ zeros[0]).astype(complex), (num_vars,))
    while (t > 0 and dt > (1/num_steps)*0.5**max_reverse):
        t -= dt
        # pick out the path matrix at time t
        W_t = np.array(path(t))
         #print(W_t)

        # a jnp.array of the ``shifted" polynomial system
        F_t = lambda X : jnp.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])

        # use jax (automatic differentiation) to build jacobian
        jac = jacfwd(F_t, holomorphic=True)

        # compute next_zero using projective Newton's method, with previous zero
        # as the initial guess
        try:
            next_zero, _ = proj_newton(next_zero, F_t, jac)

        except:
            # maybe we went too far when we stepped back, so let's reverse to the last time
            t += dt
            # and then decrease time step
            dt *= 0.5

        else:
            # if last coordinate of next_zero is far from zero, normalize so
            # that it is one
            if not np.isclose(next_zero[-1], 0):
                next_zero = next_zero / next_zero[-1]

            np.allclose([F[idx](*W_t[idx].T.conj() @ next_zero) for idx in range(N)], 0)
            # print(f"F_t(next_zero) = {F_t(next_zero)}")
            # print(f"F_t(next_zero) = {[F[idx](*W_t[idx].T.conj() @ next_zero) for idx in range(N)]}")

        finally:
            # FIXME plotting experiment
            # print(*next_zero)
            pass

    if dt > (1/num_steps)*0.5**max_reverse:
        assert np.isclose(t,0.)
    else:
        raise RuntimeError("Projective Newton failed to converge.")


    # the final zero we find (at t = 0) is the common zero of our original system;
    # let's check this
    final_zero = next_zero
    assert np.allclose([F[idx](*final_zero) for idx in range(N)], 0)
    # print('Zero of original system: ', final_zero)

    # TODO run Newton's method on this final zero and verify quadratic
    # convergence as a confidence boost
    F_final = lambda X : jnp.array([F[idx](*X) for idx in range(N)])
    jac = jacfwd(F_final, holomorphic=True)
    for i in range(5):
        print(*final_zero)
        final_zero, _ = proj_newton(final_zero, F_final, jac)

    return final_zero


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

    main([g, f], [q, p])
