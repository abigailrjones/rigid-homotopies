using Zygote: jacobian

include("../rigid_hom.jl")
include("example_utils.jl")
include("../utils.jl")

TOL = eps()*1000

open("examples/data/data_waring.txt", "w") do f
    # empty file of previous contents
end

num_vars = 2
# degrees = [idx for idx in 2:3]
degrees = [3]
rank = 3
num_funcs = length(degrees)
max_iter = 1_000_000_000
use_heuristic = false
mid_print = true

F = build_waring_system(rank, degrees, num_vars)

roots = Dict()
for idx in 1:1
    local final_root
    try
        final_root, _ = solve(F, num_funcs, num_vars, degrees, max_iter;
                              use_heuristic=use_heuristic, mid_print=mid_print)
    catch e
        if isa(e, ErrorException) || isa(e, LoadError)
            println(e)
        else
            println(e)
            # throw(e)
        end
    else
        compare_zero!(roots, final_root, num_vars)
    end
end
println(length(roots))

#=
FF = X -> [F[idx](X) for idx in 1:num_funcs]
jac = input -> jacobian(x -> real(FF(x)), input)[1] |> conj
for (root, count) in roots
    open("examples/data/data_waring.txt", "a") do f
        write(f, "$root $(cond(jac(root))) $count\n")
    end
end
=#
