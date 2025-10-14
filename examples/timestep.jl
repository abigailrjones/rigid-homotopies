#=

Compute the average (proven) timestep needed over the first N steps for a
set of increasing dimension determinantal problems.

=#

# using DelimitedFiles: writedlm
include("../rigid_hom.jl")
include("det_poly.jl")

open("examples/data_timestep.txt", "w") do f
    # empty file of previous contents
end

for dim in 2:8
    num_vars = dim
    degrees = dim .+ zeros(Int64, 1, dim-1) # had issues when deg >= num_funcs for any degree
    num_funcs = length(degrees)
    max_degree = maximum(degrees)
    max_iter = 100
    use_heuristic = false
    mid_print = false

    F = build_det_poly_system(degrees, num_vars)

    final_root, num_steps, avg_step_size = solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic, mid_print)

    open("examples/data_timestep.txt", "a") do f
        write(f, "$dim $avg_step_size\n")
    end
end
