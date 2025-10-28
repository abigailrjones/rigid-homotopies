#=

Compute the average (proven) timestep needed over the first N steps for a
set of increasing dimension determinantal problems.

=#

include("../rigid_hom.jl")
include("example_utils.jl")

TOL = eps()*1000

open("examples/data/data_timestep.txt", "w") do f
    # empty file of previous contents
end

for dim in 2:8
    num_vars = dim
    degrees = fill(dim, 1, dim-1)
    num_funcs = length(degrees)
    max_degree = maximum(degrees)
    max_iter = 100
    use_heuristic = false
    mid_print = false

    F = build_my_system(degrees, num_vars)

    final_root, num_steps, avg_step_size = solve(F, num_funcs, num_vars,
                                                 max_degree, max_iter;
                                                 use_heuristic=use_heuristic,
                                                 mid_print=mid_print)

    open("examples/data/data_timestep.txt", "a") do f
        write(f, "$dim $avg_step_size\n")
    end
end
