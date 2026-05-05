using Zygote: jacobian
import HomotopyContinuation

include("../rigid_hom.jl")
include("example_utils.jl")

TOL = eps()*10000

open("examples/data/data_big.txt", "w") do f
    # empty file of previous contents
end

num_vars = 3
degrees = [20, 2]
num_funcs = length(degrees)
max_degree = maximum(degrees)
max_iter = 1
use_heuristic = true
mid_print = false

# F = build_my_system(degrees, num_vars)

F = [X -> X[1]^20 + X[2]*X[3]^19,
# F = [X -> X[1]^100 + X[2]*X[3]^99,
     # X -> X[1]*X[3]*X[2]^98 + X[3]^100]
     X -> X[1]*X[2] + X[3]^2]
#=
=#

roots = Dict()
for idx in 1:1
    final_root, num_steps, avg_step_size = solve(F, num_funcs, num_vars,
                                                 max_degree, max_iter;
                                                 use_heuristic=use_heuristic,
                                                 mid_print=mid_print)
    println("$idx --- num steps: $num_steps, avg step size: $avg_step_size")
    compare_zero!(roots, final_root, num_vars)
end

println(length(roots))

for (root, count) in roots
    open("examples/data/data_big.txt", "a") do f
        write(f, "$root $count\n")
    end
end

#=
FF = X -> [F[idx](X) for idx in 1:num_funcs]
jac = input -> jacobian(x -> real(FF(x)), input)[1] |> conj
for (root, count) in roots
    open("examples/data/data_big.txt", "a") do f
        write(f, "$root $(cond(jac(root))) $count\n")
    end
end

HomotopyContinuation.ModelKit.@var x[1:num_vars]
FF = [F[idx](x) for idx in 1:num_funcs]
res = HomotopyContinuation.solve(FF)
solutions = [(path.solution ./ path.solution[end], path.accuracy) for path in res]
count = []
for (sol,acc) in solutions
    if isapprox(acc,0.0,atol=TOL)
        push!(count, 0)
    end
end
println("HC.jl: $(length(count))")

for (z,count) in roots
    println("Zero: $z, count: $count")
end
=#
