using Zygote: jacobian
using LinearAlgebra

# TODO incorporate paper implementation for sampler
# include("start_system.jl")
include("choose_timestep.jl")

# all functions are passed in with a vector input; the maximum number of
# variables is stored in num_vars, but this is not necessarily the number of
# vars in each function
# max_degree is the highest degree term that appears in the entire system
# we assume that all functions passed in F are holomorphic (really I'm assuming
# polynomial, but I don't think the code currently requires anything beyond
# holomorphic)
# in order to set up the start system, we also require that all polynomials in
# the system be homogeneous
#
# FIXME do num_funcs and num_vars need to be passed in?
# TODO write a little about:
# - complex differentation
# - isapprox definition and choices for atol and rtol (using only atol since
#   we're near zero, but needing to set TOL to be slightly higher than machine
#   precision in order to make things work out nicely)
# - make sure to append ! to function names that edit inputs
# - add options for predictor-corrector
# - comparing heuristic and paper with same inputs

# FIXME making this a global variable is perhaps a bit lazy/hacky?
TOL = eps()*100

function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
              max_degree::Int, max_iter::Int, use_heuristic::Bool, start_system,
              start_root)
    check_build_start_system(F, start_system, start_root, num_funcs)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
               max_degree::Int, max_iter::Int, use_heuristic::Bool, init_roots)
    for idx in 1:num_funcs
        @assert isapprox(F[idx](init_roots[idx]), 0.0, atol=TOL)
    end
    start_system, start_root = build_start_system(init_roots, num_vars)
    check_build_start_system(F, start_system, start_root, num_funcs)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

# TODO will call paper sampler
function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
               max_degree::Int, max_iter::Int, use_heuristic::Bool)
    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function rigid_hom(F::Vector{Function}, num_funcs::Int, num_vars::Int,
              max_degree::Int, max_iter::Int, use_heuristic::Bool, start_system,
              start_root)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    final_root, num_steps, avg_step_size, avg_newton_iters =
    track_path(F, path, start_root, max_degree, max_iter, num_funcs, num_vars, use_heuristic)
    check_track_path(F, final_root)

    print_output(F, final_root, use_heuristic, num_steps, avg_step_size, avg_newton_iters)
    return final_root, num_steps, avg_step_size
end

function build_start_system(init_roots, num_vars)
    V = build_random_unitary(num_vars)
    start_root = V * (init_roots[1] / sum(abs.(init_roots[1]).^2))
    start_system = [V]
    for idx in 2:length(init_roots)
        root = init_roots[idx]
        scaled_root = root / sum(abs.(root).^2)
        push!(start_system, build_unitary(scaled_root, start_root, num_vars))
    end
    return start_system, start_root
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

function build_random_unitary(num_vars)
    U, S, V = svd(rand(num_vars,num_vars) + im*rand(num_vars,num_vars))
    return U * V'
end

function build_unitary(moved_root, fixed_root, num_vars)
    if isapprox(moved_root, fixed_root, atol=TOL)
        return Matrix(1.0*I,num_vars,num_vars)
    end
    U, S, V = svd(fixed_root * moved_root')
    return U * V'
end

function build_path(start_system)
    # we are building the path given in section 3.4 of RH1 (see p.512)
    # (FIXME note that we don't use T, which might be an issue?)
    return t -> [exp(t * log(mat)) for mat in start_system]
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

function track_path(F, path, start_root, max_degree, max_iter, num_funcs, num_vars, use_heuristic=true)
    t = 1.0
    # TODO passing in initial step for heuristic?
    dt = use_heuristic ? 1.0 : choose_timestep(F, path(t), start_root, max_degree,
                                               max_iter, num_funcs, num_vars)
    num_iter = 0.0
    step_sizes = []
    newton_iter = []

    root = start_root
    for iter in 1:max_iter
        if isapprox(t, 0.0, atol=TOL)
            final_root, _ = newton!(root, X -> [F[idx](X) for idx in 1:num_funcs])
            return choose_unique_rep(final_root), iter-1,
                   sum(step_sizes)/length(step_sizes),
                   sum(newton_iter)/length(newton_iter)
        else
            t -= dt
            if iter % (max_iter // 100) == 0
                println("Iteration $iter at time t=$t, dt=$dt")
                println("Previous time at t=$(t+dt)")
            end
            W_t = path(t)
            F_t = X -> [F[idx](W_t[idx]' * X) for idx in 1:num_funcs]
            try
                root, num_iter = newton!(root, F_t)
            catch e
                if use_heuristic
                    if isa(e, ErrorException)
                        println(e)
                        t += dt
                        dt *= 0.5
                    else
                        println("A different error occurred: $e.")
                        throw(e)
                    end
                else
                    println("Using the proven timestep, Newton's method failed with error $e.")
                    throw(e)
                end
            else
                push!(step_sizes, dt)
                push!(newton_iter, num_iter)
                root = choose_unique_rep(root)
                if !use_heuristic
                    dt = choose_timestep(F, path(t), start_root, max_degree,
                                         max_iter, num_funcs, num_vars)
                end
            end
        end
    end
    throw(ErrorException("Failed to converge in fewer than $max_iter iterations."))
end

function newton!(guess, F_t, max_iter=1000)
    jac_pinv = input -> pinv(jacobian(x -> real(F_t(x)), input)[1] |> conj)
    err = 1.0
    give_up = 1e10
    num_iter = 0
    while !isapprox(err, 0.0, atol=TOL) & (err < give_up) & (num_iter < max_iter)
        next_guess = guess - jac_pinv(guess) * F_t(guess)
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

function check_track_path(F, final_root)
    @assert isapprox([func(final_root) for func in F], zeros(num_funcs,), atol=TOL)
    #=
    if !isapprox([func(final_root) for func in F], zeros(num_funcs,), atol=TOL)
        println("$([func(final_root) for func in F])")
    end
    =#
    return
end

# TODO
function print_input()
end

function print_output(F, final_root, use_heuristic, num_steps, avg_step_size, avg_newton_iters)
    println("")
    println("$(use_heuristic ? "Using a heuristic timestep..." : "Using a rigorous timestep...")")
    println("Converged in $num_steps step(s)")
    println("Average timestep: $avg_step_size")
    println("Averge number of Newton iterations per step: $avg_newton_iters")
    println("")
    println("Final root: $(round.(final_root; digits=8))")
    println("System residuals: $([round.(func(final_root); digits=8) for func in F])")
    return
end
