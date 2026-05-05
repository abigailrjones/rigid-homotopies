include("../../rigid_hom.jl")
include("../../utils.jl")

function write_system(F, file)
    open(file, "w") do f
        # empty file of previous contents
        for func in F
            write(f, "$(func.num_vars) $(func.deg) $(func.rank) $(func.M)\n")
        end
    end
end

@assert false

use_heuristic = false
mid_print = false

@assert length(ARGS) == 3

num_vars = parse(Int,ARGS[1])
num_funcs = num_vars - 1
deg = parse(Int,ARGS[2])
degrees = ones(Int,num_funcs)*deg
rank = parse(Int,ARGS[3])

if num_vars==2
    max_iter = 1_000_000
else
    max_iter = 10_000_000
end

# run iterations
iter = 0
num_iter = 100
while iter < num_iter
    F = build_waring_system(num_vars, degrees, rank*ones(Int,num_funcs))
    write_system(F, "data/start_system/data_tracking_start_system_$(num_vars)_$(deg)_$(rank).txt")

    local final_root, num_steps, min_step_size, max_step_size, avg_step_size,
    avg_duration, min_gammaprob, max_gammaprob, avg_gammaprob, min_condnum,
    max_condnum, avg_condnum
    try
        final_root, num_steps, min_step_size, max_step_size, avg_step_size,
        avg_duration, min_gammaprob, max_gammaprob, avg_gammaprob, min_condnum,
        max_condnum, avg_condnum = solve(F, num_funcs, num_vars, degrees,
                                         max_iter; use_heuristic=use_heuristic,
                                         mid_print=mid_print)
    catch e
        println("Error on run $(iter+1) (rank is $rank).")
        println(e)
        println("")
    else
        global iter += 1
        open("data/data_tracking_$(num_vars)_$(deg)_$(rank).txt", "a") do f
            write(f, "$num_steps $min_step_size $max_step_size $avg_step_size $min_gammaprob $max_gammaprob $avg_gammaprob $min_condnum $max_condnum $avg_condnum\n")
        end
    end
end
