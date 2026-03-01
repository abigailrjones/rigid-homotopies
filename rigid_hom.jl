using Statistics

# TODO I should figure out how julia handles includes (since I am including
# utils in start_system)
include("utils.jl")
include("start_system.jl")
include("choose_timestep.jl")

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
        mid_print::Bool, initial_dt)
    success, final_root, num_steps, avg_step_size, avg_newton_iters =
    track_path(system, path, start_root, max_degree, max_iter, num_funcs,
               num_vars, use_heuristic, mid_print, initial_dt)
    if success
        try
            check_track_path(system, path(0.0), final_root)
        catch
            final_root, num_steps, avg_step_size = fill(NaN+NaN*im,num_vars),
                                                   NaN, NaN
        end
    else
        # TODO handle different errors
        println("Track path was unsuccessful.")
    end

    if (mid_print) print_output(success, system, path(0.0), final_root,
                            use_heuristic, num_steps, avg_step_size,
                        avg_newton_iters) end
    if !success
        @assert false
    end
    return final_root, num_steps, avg_step_size
end

function build_path(start_system)
    # we are building the path given in section 3.4 of RH1 (see p.512)
    # taking conjugate transpose now because this is how the matrices act on
    # the input of the polynomial system
    return t -> [exp(t * log(mat))' for mat in start_system]
end

function track_path(system, path, start_root, max_degree, max_iter, num_funcs,
        num_vars, use_heuristic, mid_print, initial_dt)
    t = 1.0
    W_t = path(t)
    dt = use_heuristic ? initial_dt : choose_timestep(system, W_t, start_root,
                                                      max_degree, max_iter,
                                                      num_funcs, num_vars)
    num_newton_iter = 0
    prog_data = [1.0, 0.0, 0.0, 0.0]
    #         = [min dt, max dt, avg dt, avg newton iter]

    root = complex(start_root)
    for iter in 1:max_iter
        if isapprox(t, 0.0, atol=TOL) || (t < 0.0)
            # refine root
            newton!(root, system, path(0.0))
            println("Minimum timestep: $(prog_data[1]), maximum timestep: $(prog_data[2])")

            return true, scale_root(root), iter-1,
                   prog_data[3], prog_data[4]
        else
            t -= dt
            W_t = path(t)
            try
                num_newton_iter = newton!(root, system, W_t)
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
                # update min, max, and average timestep data
                if (dt < prog_data[1]) prog_data[1] = dt end
                if (dt > prog_data[2]) prog_data[2] = dt end
                prog_data[3] = prog_data[3] + (dt - prog_data[3])/iter
                # update average newton iteration data
                prog_data[4] = prog_data[4] + (num_newton_iter - prog_data[4])/iter
                root = scale_root(root)
                # compute timestep for next step
                if !use_heuristic
                    dt = choose_timestep(system, W_t, root, max_degree,
                                         max_iter, num_funcs, num_vars)
                end
            end
        end
    end

    return false, NaN, max_iter, prog_data[3], prog_data[4]
end
