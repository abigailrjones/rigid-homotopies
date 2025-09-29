# FIXME not needed?
# using LinearAlgebra

# TODO I should figure out how julia handles includes (since I am including
# utils in start_system)
include("utils.jl")
include("start_system.jl")
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

function solve(F::Vector, num_funcs::Int, num_vars::Int,
               max_degree::Int, max_iter::Int, use_heuristic::Bool, start_system,
               start_root, path)
    check_build_start_system(F, start_system, start_root, num_funcs)
    # TODO
    # check_build_path(path, start_system, num_vars)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
              max_degree::Int, max_iter::Int, use_heuristic::Bool, start_system,
              start_root)
    check_build_start_system(F, start_system, start_root, num_funcs)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
               max_degree::Int, max_iter::Int, use_heuristic::Bool, init_roots)
    check_init_roots(F, init_roots, num_vars)
    start_system, start_root = build_start_system(F, init_roots, num_vars)
    check_build_start_system(F, start_system, start_root, num_funcs)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function solve(F::Vector{Function}, num_funcs::Int, num_vars::Int,
               max_degree::Int, max_iter::Int, use_heuristic::Bool)
    start_system, start_root = build_start_system(F, num_vars)
    check_build_start_system(F, start_system, start_root, num_funcs)

    return rigid_hom(F, num_funcs, num_vars, max_degree, max_iter,
                     use_heuristic, start_system, start_root)
end

function rigid_hom(F::Vector, num_funcs::Int, num_vars::Int, max_degree::Int,
                   max_iter::Int, use_heuristic::Bool, start_system, start_root)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    final_root, num_steps, avg_step_size, avg_newton_iters =
    track_path(F, path, start_root, max_degree, max_iter, num_funcs, num_vars, use_heuristic)
    check_track_path(F, final_root)

    print_output(F, final_root, use_heuristic, num_steps, avg_step_size, avg_newton_iters)
    return final_root, num_steps, avg_step_size
end

function build_path(start_system)
    # we are building the path given in section 3.4 of RH1 (see p.512)
    # (FIXME note that we don't use T, which might be an issue?)
    return t -> [exp(t * log(mat)) for mat in start_system]
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
