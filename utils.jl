using Zygote: jacobian

function newton!(guess, F_t, max_iter=1000)
    # next line is complex diffentiation (recall Cauchy-Riemann)
    jac_pinv = input -> pinv(jacobian(x -> real(F_t(x)), input)[1] |> conj)
    err = 1.0
    give_up = 1e10
    num_iter = 0
    while !isapprox(err, 0.0, atol=TOL) & (err < give_up) & (num_iter < max_iter)
        if size(guess) == ()
            # jac_pinv returns a 1x1 vector in this case, when we need a scalar
            # so that operations with guess are defined
            next_guess = guess - (jac_pinv(guess) * F_t(guess))[1]
        else
            next_guess = guess - (jac_pinv(guess) * F_t(guess))
        end
        err = norm(next_guess - guess)
        #=
        println("Iteration $num_iter: guess=$(round.(guess; digits=3)), next
                guess=$(round.(next_guess; digits=3))")
        =#
        guess = next_guess
        num_iter += 1
    end

    if num_iter > max_iter
        throw(ErrorException("Newton's method failed to converge within $max_iter iterations."))
    end

    if err > give_up
        throw(ErrorException("Newton's method is going to infinity."))
    end

    return guess, num_iter
end

# FIXME this isn't unique
function choose_unique_rep(x)
    # return x / norm(x)
    return x
end

# TODO
function print_input()
end

function print_output(target_system, final_root, use_heuristic, num_steps, avg_step_size, avg_newton_iters)
    println("")
    println("$(use_heuristic ? "Using a heuristic timestep..." : "Using a rigorous timestep...")")
    println("Converged in $num_steps step(s)")
    println("Average timestep: $avg_step_size")
    println("Average number of Newton iterations per step: $avg_newton_iters")
    println("")
    println("Final root: $(round.(final_root; digits=8))")
    println("System residuals: $(round.(target_system(final_root); digits=8))")
    return
end

function check_init_roots(F, init_roots, num_vars)
    for idx in 1:length(F)
        @assert isapprox(F[idx](init_roots[idx]), 0.0, atol=TOL)
        @assert (length(init_roots[idx]) == num_vars)
    end
end

# build_start_system will only work as expected if inputted polynomials are
# homogeneous (the process with unitary matrices fails otherwise); if this
# assert fails, it could mean that the inputted polynomials did not satisfy
# this requirement
function check_build_start_system(F, start_system, start_root, num_funcs)
    for idx in 1:num_funcs
        f = F[idx]
        U = start_system[idx]
        @assert isapprox(f(U' * start_root), 0.0, atol=TOL)
    end
    return
end

function check_build_path(path, start_system, num_vars)
    W_0 = path(0.0)
    W_1 = path(1.0)
    for idx in 1:length(W_0)
        @assert isapprox(W_0[idx]-I, zeros(num_vars,num_vars), atol=TOL)
        @assert isapprox(W_1[idx]-start_system[idx], zeros(num_vars,num_vars), atol=TOL)
    end
    return
end

function check_track_path(target_system, final_root)
    @assert isapprox(target_system(final_root), zeros(num_funcs,), atol=TOL)
    return
end
