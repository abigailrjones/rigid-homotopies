import mpmath as mp
import utils


def build_unitary(moved_zero, fixed_zero):
    # FIXME handle identity case more smoothly
    if utils.allclose(moved_zero, fixed_zero):
        return mp.eye(len(fixed_zero))

    M = fixed_zero @ moved_zero.T
    U, S, Vh = mp.svd(M)
    return U @ Vh


def build_random_unitary(n):
    U, S, Vh = mp.svd(mp.randmatrix(n) + 1j*mp.randmatrix(n))
    return U @ Vh


def build_path(A, num_funcs):
    # to build this path, we are following section 3.4 in RH part 1 (see p.512)
    log_A = [mp.logm(A[idx]) for idx in range(num_funcs)]

    # TODO this choice of T might be giving us the needed Lipschitz continuity?
    # T = mp.sqrt(sum([mp.norm(A[idx])**2 for idx in range(num_funcs)]) / 2.)
    # W_t = lambda t : [mp.expm((t/T) * A[idx]) for idx in range(num_funcs)]

    W_t = lambda t : [mp.expm(t * log_A[idx]) for idx in range(num_funcs)]
    return W_t


def newton(guess, F_t, num_funcs, num_vars, projective=True, max_iter=1000):
    jac = utils.build_jacobian(F_t, num_funcs, num_vars, projective)

    if projective:
        F_t.append(lambda *args: mp.mpf('0.'))
        compute_inverse = mp.inverse
    else:
        compute_inverse = utils.pseudoinverse

    err = mp.mpf('1.')
    give_up = 1e10
    num_iter = 0
    while not mp.almosteq(err, mp.mpf('0.')) and err < give_up and num_iter < max_iter:
        next_guess = guess - (compute_inverse(jac(guess)) @ utils.eval_sys(F_t, guess))
        err = mp.norm(next_guess - guess)
        guess = next_guess
        num_iter += 1

    if num_iter > max_iter:
        raise RuntimeError('Too many iterations.')
    if err > give_up:
        raise RuntimeError('Going to infinity and beyond.')

    # FIXME
    # print(f"Number of Newton iterations: {num_iter}")
    return guess, num_iter


def track_path(F, path, init_zero, num_funcs, num_vars, max_iter, projective=True):
    t = mp.mpf('1.')
    dt = mp.mpf('.1')
    step_sizes = list()

    W_t = path(t)
    F_t = utils.compute_shifted_system(F, W_t, num_funcs)

    utils.print_input(t, dt, projective)

    zero = init_zero
    for itr in range(1,max_iter+1):
        if mp.almosteq(t, mp.mpf('0.')):
            # FIXME
            num_refine = 1
            for _ in range(num_refine):
                zero, _ = newton(zero, F.copy(), num_funcs, num_vars,
                                 projective=projective)

            assert utils.allclose(utils.eval_sys(F, zero),
                                  mp.zeros(num_funcs,1))
            # TODO how to scale to get a ``unique" representative? (also need
            # to use same choice in else statement below; recall example where
            # Newton was basically trying to return (0,0,0), but without
            # exactly equalling zero, and scaling it made it no longer a
            # solution), maybe make above assert for scaled version of zero
            # return utils.scale(zero), itr, utils.mean(step_sizes)
            return zero / zero[0], itr-1, utils.mean(step_sizes)

        else:
            t -= dt
            step_sizes.append(dt)
            if itr % (max_iter // 10) == 0:
                print(f"Iteration {itr} at t={mp.nstr(t)}")
                print(f"Previous time at t={mp.nstr(t+dt)}")

            W_t = path(t)
            F_t = utils.compute_shifted_system(F, W_t, num_funcs)
            try:
                zero, _ = newton(zero, F_t, num_funcs, num_vars,
                                 projective=projective)
            except RuntimeError:
                t += dt
                dt *= 0.5
            else:
                zero = utils.scale(zero)

    raise RuntimeError(f"Failed to converge in fewer than {max_iter} iterations.")


def main(F, zeros, max_iter=1000, projective=True):
    # TODO zeros is a list of initial zeros of each poly in F (not common
    # zeros, just plain old zeros). eventually, we want the program to build
    # these initial zeros itself

    # TODO what needs projective?

    # TODO add an option to use paper implementation or use passed in zeros


    """ INITIAL CHECKS """
    arg_counts = [func.__code__.co_argcount for func in F]
    num_funcs = len(F)

    # check that each func in F has the same number of variables
    assert utils.allclose(mp.matrix(arg_counts), arg_counts[0]*mp.ones(num_funcs,1))

    # verify that this number of variables is exactly one more
    # than the number of polynomials in the system
    # FIXME
    # num_vars = arg_counts[0]
    num_vars = len(zeros[0])
    assert (num_vars == num_funcs + 1)


    """ BUILD START SYSTEM """
    # scaling initial zeros to have magnitude one
    zeros = [mp.matrix(zero) / mp.norm(mp.matrix(zero)) for zero in zeros]

    # check that all scaled zeros are still zeros of the original
    # polynomial system (this is also an implicit check that the original
    # system is indeed homogeneous, as required)
    assert utils.allclose(mp.matrix([F[idx](*zeros[idx]) for idx in range(num_funcs)]),
                          mp.zeros(num_funcs,1))

    # now we use build_unitary to construct a unitary matrix that maps
    # each initial zero to the first in the list (i.e., zeros[0] will
    # be (almost) the common zero of our start system)
    A = [build_unitary(zeros[idx], zeros[0]) for idx in range(num_funcs)]
    assert utils.allclose(A[0], mp.eye(num_vars))
    for idx in range(num_funcs):
        assert utils.allclose(A[idx] @ zeros[idx], zeros[0])

    # to make the start system truly generic, we hit the matrix system A
    # with an arbitrary unitary matrix
    # TODO options to fix randomness?
    V = build_random_unitary(num_vars)
    # V = mp.eye(num_vars)
    A = [V @ A[idx] for idx in range(num_funcs)]

    # check that the new polynomial system (A \cdot F) has common zero
    # V @ zeros[0]
    for idx in range(num_funcs):
        assert mp.almosteq(F[idx](*A[idx].conjugate().T @ V @ zeros[0]),
                           mp.mpf('0.'))


    """ BUILD PATH """
    # now we need to build a path in \mathcal{U} that maps array
    # of num_vars by num_vars identity matrices to A
    path = build_path(A, num_funcs)

    # let's check that this path does what we expect
    # at t = 0, each matrix in W_t = patth(0) should equal a (num_vars x
    # num_vars) identity matrix
    W_t = path(mp.mpf('0.'))
    for idx in range(num_funcs):
        assert utils.allclose(W_t[idx], mp.eye(num_vars))

    # at t = 1, W_t = path(1) should equal A
    W_t = path(mp.mpf('1.'))
    for idx in range(num_funcs):
        assert utils.allclose(W_t[idx], A[idx])


    """ TRACK ZERO """
    # next we track the common zero V @ zeros[0] along this path using a
    # projective Newton's method
    init_zero = V @ zeros[0]
    final_zero, num_iter, avg_step_size = track_path(F, path, init_zero,
                                                     num_funcs, num_vars,
                                                     max_iter, projective)

    utils.print_output(F, final_zero, num_iter, avg_step_size)
    return final_zero, num_iter, avg_step_size


if __name__ == '__main__':
    mp.mp.pretty = True
    mp.mp.dps = 40

    # test (homogeneous) polynomials
    # f = lambda x, y, z: x**2 - 2*x*z + y**2
    f = lambda x, y, z: x**5 - 2*z*x**4 + y**5
    g = lambda x, y, z: x**2 + y**2 - z**2

    # sys.exit()
    # zeros of test polynomials
    p = [1,1,1]
    q = [1,0,1]

    final_zero, num_iter, avg_step_size = main([g,f], [q, p], max_iter=1000,
                                               projective=False)
