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

TOL = eps()*1000

open("examples/data/data_other_complexity.txt", "w") do f
    # empty file of previous contents
end

a = 2
b = 8
base = 2

num_vars = 6
degrees = [5,5,5,5,5]
num_funcs = length(degrees)
max_iter = 1_000_000_000
use_heuristic = false
mid_print = true

for i in a:b
    # println(i)
    rank = base^i
    for iter in 1:100
        F = build_waring_system(num_vars, degrees, rank*ones(Int,length(degrees)))
        write_system(F, "examples/data/data_other_start_system.txt")

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
            open("examples/data/data_other_complexity.txt", "a") do f
                write(f, "$rank $iter $num_steps $avg_step_size\n")
            end
            # compare_zero!(roots, final_root, num_vars)
        end
    end
end
