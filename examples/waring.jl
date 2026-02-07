# using Zygote: jacobian
using ProfileView

include("../rigid_hom.jl")
include("../utils.jl")

TOL = eps()*1000

# open("examples/data/data_waring.txt", "w") do f
    # empty file of previous contents
# end

a = 2
b = 2
base = 2

num_vars = 3
degrees = [3,4]
num_funcs = length(degrees)
max_iter = 100_000#_000
use_heuristic = true
mid_print = true

for i in a:b
    println(i)
    rank = base^i
    F = build_waring_system(num_vars, degrees, rank*ones(Int,length(degrees)))
    local num_steps
    local avg_step_size

    @profview solve(F, num_funcs, num_vars, degrees, max_iter;
                    use_heuristic=use_heuristic, mid_print=mid_print,
                    initial_dt=0.001)
    #=
    try
        _, num_steps, avg_step_size = solve(F, num_funcs, num_vars, degrees,
                                            max_iter;
                                            use_heuristic=use_heuristic,
                                            mid_print=mid_print)
    catch e
        if isa(e, ErrorException) || isa(e, LoadError)
            println(e)
        else
            println(e)
            # throw(e)
        end
    else
        open("examples/data/data_waring.txt", "a") do f
            write(f, "$i $num_steps $avg_step_size\n")
        end
        # compare_zero!(roots, final_root, num_vars)
    end
    =#
end
# println(length(roots))

#=
FF = X -> [F[idx](X) for idx in 1:num_funcs]
jac = input -> jacobian(x -> real(FF(x)), input)[1] |> conj
for (root, count) in roots
    open("examples/data/data_waring.txt", "a") do f
        write(f, "$root $(cond(jac(root))) $count\n")
    end
end
=#
