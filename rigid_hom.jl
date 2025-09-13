
# FIXME I'm really not sure how Julia handles overlapping ``using" and
# ``include" in files
using Zygote: jacobian

# TODO incorporate paper implementation with heuristics
# include("start_system.jl")
# include("choose_timestep.jl")

# all functions that are passed in must take in the same number of variables
# max_degree is the highest degree term that appears in the entire system
# we assume that all functions passed in F are holomorphic (really I'm assuming
# polynomial, but I don't think the code currently requires anything beyond
# holomorphic)
# FIXME does anything require square? (Newton?)
# FIXME do num_funcs and num_vars need to be passed in?
# TODO write a little about:
# - complex differentation
# - isapprox definition and choices for atol and rtol
# - make sure to append ! to function names that edit inputs

function main(F::Vector{Function}, num_funcs::Int,num_vars::Int,
              max_degree::Int, init_roots=Nothing)
    start_system, start_root= build_start_system(init_roots, num_vars)
    check_build_start_system(F, start_system, start_root, num_funcs)

    path = build_path()
    check_path(path, start_system, num_vars)

    # TODO add options for Newton (predictor-corrector), timestepper
    final_root, num_iter, avg_step_size = track_path_heuristic(F, path,
                                                               start_root,
                                                               max_iter,
                                                               num_funcs,
                                                               num_vars)
    # FIXME
    # check_track_path(F, final_root)

    print_output(F, final_root, num_iter, avg_step_size)
    return final_root, num_iter, avg_step_size
end

# TODO add a method that just tracks a system along a given path (does this
# need its own wrapper?)

# TODO overload this function (one calls paper implementation, other calls heuristic)
# (I'm not sure which declaration would trigger if Nothing is passed)
function build_start_system(init_roots, num_vars)
    if init_roots == Nothing
        @assert false
    end
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
        @assert isapprox(f(U' * start_root), 0.0, atol=eps())
    end
    return
end

function build_random_unitary(num_vars)
    U, S, Vh = svd(rand(num_vars,num_vars) + im*rand(num_vars,num_vars))
    return U * Vh
end

function build_unitary(moved_root, fixed_root, num_vars)
    if isapprox(moved_root, fixed_root, atol=eps())
        return Matrix(1.0*I,num_vars,num_vars)
    end
    U, S, Vh = svd(fixed_root * moved_root')
    return U * Vh
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
        @assert isapprox(W_0[idx]-I, zeros(num_vars,num_vars), atol=eps())
        @assert isapprox(W_1[idx]-start_system[idx], zeros(num_vars,num_vars), atol=eps())
    end
    return
end

# TODO (is there a neater way of having both a paper and a heuristic implementation?)
function track_path_paper()
end

function track_path_heuristic(F, path, start_root, max_iter, num_funcs, num_vars)
    t = 1.0
    dt = 1.0
    step_sizes = []
    root = start_root
    for iter in 1:max_iter
        if isapprox(t, 0.0, atol=eps())
            final_root, _ = newton()
            return choose_unique_rep(final_root), iter-1, sum(step_sizes)/length(step_sizes)
        else
            t -= dt
            W_t = path(t)
            F_t = X -> [F[idx](W_t[idx]' * X) for idx in 1:num_funcs]
            try
                root, _ = newton()
            catch
                t += dt
                dt *= 0.5
            else
                push!(step_sizes, dt)
                # TODO make sure that this is the same root that we are working with throughout
                # (i.e., that we're not just forgetting it every time we exit a block)
                root = choose_unique_rep(root)
            end
        end
    end
    throw(ErrorException, "Failed to converge in fewer than $max_iter iterations.")
end

# TODO
function newton(guess, F_t)
    # TODO TODO TODO START HERE
    # TODO could we make this a function?
    jac = jacobian(X -> real(F_t(X)), guess)[1] |> conj
    # pinv in Julia is pseudoinverse
end

# TODO
function choose_unique_rep(x)
    return x
end

function check_track_path(F, final_root)
    @assert isapprox([func(final_root) for func in F], zeros(num_vars,), atol=eps())
    return
end

# TODO
function print_input()
end

function print_output(F, final_root, num_iter, avg_step_size)
    println("Converged in $num_iter iteration(s)")
    println("Average timestep: $avg_step_size")
    println("Final root: $final_root")
    println("System residuals: $([func(final_root) for func in F])")
    return
end
