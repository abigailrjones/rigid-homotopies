include("../rigid_hom.jl")
include("example_utils.jl")
include("../utils.jl")

TOL = eps()*1000

open("examples/data/data_dist_zeros.txt", "w") do f
    # empty file of previous contents
end

num_vars = 3 # free
degrees = [3,3] # had issues when deg >= num_funcs for any degree
num_funcs = length(degrees)
max_degree = maximum(degrees)
max_iter = 100
use_heuristic = true
mid_print = false

#=
F = [X -> X[1]^3 + X[1]*X[2]*X[3] + X[1]*X[2]^2 + X[2]*X[3]^2,
     X -> X[1]^2*X[3] - X[1]*X[2]*X[3] + X[2]^3 + X[3]^3]
=#

F = build_my_system(degrees, num_vars)

# build start system and path here, so that randomness is fixed throughout the example
start_system, start_root = build_start_system(F, num_vars)
check_build_start_system(F, start_system, start_root, num_funcs)
path = build_path(start_system)
check_build_path(path, start_system, num_vars)

roots = Dict()
for idx in 1:10000
    final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter;
                          use_heuristic=use_heuristic, mid_print=mid_print)
    compare_zero!(roots, final_root, num_vars)
end
println(length(roots))

for (root, count) in roots
    open("examples/data/data_dist_zeros.txt", "a") do f
        write(f, "$root $count\n")
    end
end
