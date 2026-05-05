include("../rigid_hom.jl")
include("../utils.jl")

function write_system(F, file)
    open(file, "w") do f
        # empty file of previous contents
        for func in F
            write(f, "$(func.num_vars) $(func.deg) $(func.rank) $(func.M)\n")
        end
    end
end

# @assert false

TOL = eps()*1000

# a = 2
# a = 5
# b = 8
base = 2

num_vars = 3
degrees = [4,4]
num_funcs = length(degrees)
max_iter = 1_000_000_000
use_heuristic = false
mid_print = true

@assert length(ARGS)==1

# for i in a:b
rank = base^parse(Int,ARGS[1])

# create files
#=
open("examples/data/complexity/start_system/data_start_system_$rank.txt", "w") do f
    # empty file of previous contents
end

open("examples/data/complexity/data_complexity_$rank.txt", "w") do f
    # empty file of previous contents
end
=#

# run iterations
for iter in 1:100
    F = build_waring_system(num_vars, degrees, rank*ones(Int,length(degrees)))
    write_system(F, "examples/data/complexity/start_system/data_start_system_$rank.txt")

    local num_steps
    local avg_step_size
    try
        _, num_steps, avg_step_size = solve(F, num_funcs, num_vars, degrees,
                                            max_iter;
                                            use_heuristic=use_heuristic,
                                            mid_print=mid_print)
    catch e
        println("Error on run $iter where i=$i (so rank is $rank).")
        throw(e)
    else
        open("examples/data/complexity/data_complexity_$rank.txt", "a") do f
            write(f, "$rank $iter $num_steps $avg_step_size\n")
        end
        # compare_zero!(roots, final_root, num_vars)
    end
end
# end
