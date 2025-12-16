using FFTW: fft
using Zygote: jacobian

CUT_OFF = 5.0

function check_inputs(num_funcs, num_vars)
    if (num_funcs != num_vars-1)
        throw(ErrorException("Number of functions must equal number of non-homogeneous variables."))
    end
end

# returns a vector with D+1 components, representing the 0:Dth degree
# components of the given polynomial f evaluated at the input
function compute_deg_components(f::Function,input,D::Integer)
    return fft([f(exp(2*pi*im*j/(D+1))*input) for j in 0:D]) / (D+1)
end

function newton!(guess, F_t; max_iter=1000, tol=eps()*100)
    # next line is complex diffentiation (recall Cauchy-Riemann)
    # jac_pinv = input -> pinv(jacobian(x -> real(F_t(x)), input)[1] |> conj)
    # split line 20 into differentiation and inversion so that profiler can
    # distinguish
    # jac_pinv = input -> pinv(jacobian(x -> real(F_t(x)), input)[1] |> conj)

    err = 1.0
    residual = 1.0
    give_up = 1e10
    num_iter = 0
    num_small = 0
    magic = 3

    # while (num_small < magic) & (!isapprox(residual, 0.0, atol=tol)) & (err < give_up) & (num_iter < max_iter)
    while (err < give_up) & (num_iter < max_iter)
        if (num_small > magic) & isapprox(residual, 0.0, atol=tol)
            # println(residual)
            return guess, num_iter
        else
            jac = jacobian(x -> real(F_t(x)), guess)
            if size(guess) == ()
                # jac_pinv returns a 1x1 vector in this case, when we need a scalar
                # so that operations with guess are defined
                next_guess = guess - (pinv(jac[1] |> conj) * F_t(guess))[1]
            else
                next_guess = guess - (pinv(jac[1] |> conj) * F_t(guess))
            end
            residual = norm(F_t(guess))
            err = norm(next_guess - guess)
            if isapprox(err, 0.0, atol=tol) num_small += 1
            else num_small = 0
            end
            #=
            println("Iteration $num_iter: guess=$(round.(guess; digits=3)), next
                    guess=$(round.(next_guess; digits=3))")
            =#
            guess = next_guess
            num_iter += 1
        end
    end

    if err >= give_up
        throw(ErrorException("Newton's method is going to infinity."))
    end

    if num_iter >= max_iter
        throw(ErrorException("Newton's method failed to converge within $max_iter iterations."))
    end
end

    #=
    @assert (num_small == magic)
    if size(F_t(guess)) == ()
        # xs = [0,1,2]
        # plot(xs,F_t)
        if (!isapprox(F_t(guess), 0.0, atol=tol))
            # println(F_t.(-10:0.1:10))
            open("broken_newton.txt", "a") do f
                write(f, "$(real.(F_t.(-10:0.1:10)))\n")
            end
            throw(ErrorException("Newton's method converged to a non-root."))
        end
    else
        if (!isapprox(F_t(guess), zeros(size(F_t(guess))), atol=tol))
            # println(F_t.(-10:0.1:10))
            open("broken_newton.txt", "a") do f
                write(f, "$(real.(F_t.(-10:0.1:10)))\n")
            end
            throw(ErrorException("Newton's method converged to a non-root."))
        end
    end

    if (!isapprox(F_t(guess), 0.0, atol=TOL))
        println("error in Newton: $err")
        println("evaluating function: $(F_t(guess))")
        println("num iterations: $num_iter")
        println("prev func eval: $prev_func")
        println("prev jacobian: $prev_jac")
        svd_res = svd(update_step)
        println("singular vals of previous update step: $(svd_res.S)")
        @assert false
    end
    println("prev func eval: $prev_func")
    println("prev jacobian: $prev_jac")
    svd_res = svd(update_step)
    println("singular vals of previous update step: $(svd_res.S)")
    =#

function scale_root(x)
    max_idx = argmax(abs.(x))
    if abs(x[max_idx]) >= CUT_OFF
        x = x ./ x[max_idx]
    end
    return x
end

# TODO
function print_input()
end

function print_output(success, target_system, final_root, use_heuristic,
                      num_steps, avg_step_size, avg_newton_iters)
    println("")
    println("$(use_heuristic ? "Using a heuristic timestep..." : "Using a rigorous timestep...")")
    if success
        println("Converged in $num_steps step(s)")
    else
        println("Failed to converge in $num_steps step(s)")
    end
    println("Average timestep: $avg_step_size")
    println("Average number of Newton iterations per step: $avg_newton_iters")
    println("")
    if success
        println("Final root: $(round.(final_root; digits=8))")
        # println("System residuals: $(round.(target_system(final_root); digits=20))")
        println("System residuals: $(target_system(final_root))")
    end
    return
end

function check_init_roots(F, init_roots, num_vars)
    for idx in 1:length(F)
        @assert isapprox(F[idx](init_roots[idx]), 0.0, atol=TOL)
        @assert (length(init_roots[idx]) == num_vars)
    end
end

function check_sampled_init_root(f, init_root)
    if (!isapprox(f(init_root), 0.0, atol=TOL))
        println(f(init_root))
    end
    @assert isapprox(f(init_root), 0.0, atol=TOL)
    if (!isapprox(f(init_root/norm(init_root)), 0.0, atol=TOL))
        println(norm(init_root))
        println(f(init_root/norm(init_root)))
    end
    @assert isapprox(f(init_root/norm(init_root)), 0.0, atol=TOL)
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
