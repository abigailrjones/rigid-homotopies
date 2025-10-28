#=

Compute the average (proven) timestep needed over the first N steps for a
set of increasing dimension determinantal problems.

=#

using Plots
plotlyjs()

include("../rigid_hom.jl")
include("example_utils.jl")

TOL = eps()*1000

p = plot()

for idx in 1:10
    dims = 2:8
    data = []
    for dim in dims
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

        push!(data, avg_step_size)
    end

    # plot
    plot!(dims, data, yscale=:log10, c=idx, label="")
    plot!(dims, data, st=:scatter, yscale=:log10, c=idx,
          label="",xlabel="N",ylabel="timestep",title="(N-1) degree N \
          polynomials in N variables")
end

display(p)
