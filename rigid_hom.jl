using Statistics
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

function solve(system::Vector, num_funcs::Int, num_vars::Int, max_degree::Int,
        max_iter::Int, start_system, start_root, path;
        use_heuristic::Bool=false, mid_print::Bool=false,
        initial_dt::Number=0.1)
    check_inputs(num_funcs, num_vars)
    if (mid_print) println("A start system, start root, and path were provided.") end
    check_build_start_system(system, start_system, start_root, num_funcs)
    # TODO check_build_path

    return rigid_hom(system, num_funcs, num_vars, max_degree, max_iter,
                     start_system, start_root, path, use_heuristic, mid_print,
                     initial_dt)
end

function solve(system::Vector, num_funcs::Int, num_vars::Int, max_degree::Int,
        max_iter::Int, start_system, start_root; use_heuristic::Bool=false,
        mid_print::Bool=false, initial_dt::Number=0.1)
    check_inputs(num_funcs, num_vars)
    if (mid_print) println("A start system and start root were provided. The \
                            default path will be used.") end
    check_build_start_system(system, start_system, start_root, num_funcs)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    return rigid_hom(system, num_funcs, num_vars, max_degree, max_iter,
                     start_system, start_root, path, use_heuristic, mid_print,
                     initial_dt)
end

function solve(system::Vector, num_funcs::Int, num_vars::Int, max_degree::Int,
        max_iter::Int, init_roots; use_heuristic::Bool=false,
        mid_print::Bool=false, initial_dt::Number=0.1)
    check_inputs(num_funcs, num_vars)
    if (mid_print) println("A list of initial zeros was provided. A default \
                       start system will be computed from these initial zeros, \
                   and the default path will be used.") end
    check_init_roots(system, init_roots, num_vars)
    start_system, start_root = build_start_system(system, init_roots, num_vars)
    check_build_start_system(system, start_system, start_root, num_funcs)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    return rigid_hom(system, num_funcs, num_vars, max_degree, max_iter,
                     start_system, start_root, path, use_heuristic, mid_print,
                     initial_dt)
end

function solve(system::Vector, num_funcs::Int, num_vars::Int, degrees::Vector{Int},
        max_iter::Int; use_heuristic::Bool=false, mid_print::Bool=false,
        initial_dt::Number=0.1)
    check_inputs(num_funcs, num_vars)
    max_degree = maximum(degrees)
    if (mid_print) println("The default random start system and start root will \
                       be used, as well as the default path.") end
    start_system, start_root = build_start_system(system, degrees, num_vars)
    check_build_start_system(system, start_system, start_root, num_funcs)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    return rigid_hom(system, num_funcs, num_vars, max_degree, max_iter,
                     start_system, start_root, path, use_heuristic, mid_print,
                     initial_dt)
end

function rigid_hom(system::Vector, num_funcs::Int, num_vars::Int, max_degree::Int,
        max_iter::Int, start_system, start_root, path, use_heuristic::Bool,
        mid_print::Bool, initial_dt,)
    success, final_root, target_system, num_steps, avg_step_size,
    avg_newton_iters = track_path(system, path, start_root, max_degree, max_iter,
                                  num_funcs, num_vars, use_heuristic,
                                  mid_print, initial_dt)
    if success
        try
            check_track_path(target_system, final_root)
        catch
            # FIXME
            final_root, num_steps, avg_step_size = fill(NaN+NaN*im,num_vars),
                                                   NaN, NaN
        end
    else
        # TODO handle different errors
        println("Track path was unsuccessful.")
        @assert false
    end

    if (mid_print) print_output(success, target_system, final_root,
                            use_heuristic, num_steps, avg_step_size,
                        avg_newton_iters) end
    return final_root, num_steps, avg_step_size
end

function build_path(start_system)
    # we are building the path given in section 3.4 of RH1 (see p.512)
    # (FIXME note that we don't use T, which might be an issue?)
    return t -> [exp(t * log(mat)) for mat in start_system]
end

function track_path(system, path, start_root, max_degree, max_iter, num_funcs,
        num_vars, use_heuristic, mid_print, initial_dt)
    t = 1.0
    shifted_system = shift(system, path(t))
    dt = use_heuristic ? initial_dt : choose_timestep(shifted_system, start_root,
                                                      max_degree, max_iter,
                                                      num_funcs, num_vars)
    num_iter = 0.0
    step_sizes = []
    newton_iter = []

    root = complex(start_root)
    for iter in 1:max_iter
        if isapprox(t, 0.0, atol=TOL) || (t < 0.0)
            target_system = shift(system, path(0.0))
            newton!(root, target_system)

            println("Median: $(median(step_sizes)), minimum: $(minimum(step_sizes)), maximum: $(maximum(step_sizes))")
            #=
            println("Number of iterations: $(iter-1)")
            return true, NaN, [], iter-1, sum(step_sizes)/length(step_sizes),NaN
                   #sum(newton_iter)/length(newton_iter)
            =#
            return true, scale_root(root), target_system, iter-1,
                   sum(step_sizes)/length(step_sizes),
                   sum(newton_iter)/length(newton_iter)
        else
            t -= dt
            shifted_system = shift(system, path(t))
            try
                num_iter = newton!(root, shifted_system)
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
                    if (mid_print) println("Using the proven timestep, Newton's \
                                           method failed with error $e after \
                                           $iter iterations.") end
                    # throw(e)
                    break
                end
            else
                push!(step_sizes, dt)
                push!(newton_iter, num_iter)
                root = scale_root(root)
                if !use_heuristic
                    dt = choose_timestep(shifted_system, root, max_degree,
                                         max_iter, num_funcs, num_vars)
                end
            end
            #=
            push!(step_sizes, dt)
            dt = choose_timestep(system, path(t), start_root, max_degree, max_iter,
                                 num_funcs, num_vars)
            =#
        end
    end

    return false, NaN, [], max_iter, sum(step_sizes)/length(step_sizes),
           sum(newton_iter)/length(newton_iter)
end
