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

num_vars = 2
degrees = [3]
num_funcs = length(degrees)
max_iter = 1_000_000
use_heuristic = false
mid_print = true

@assert length(ARGS)==1
rank = parse(Int,ARGS[1])

# run iterations
iter = 0
while iter < 100
    F = build_waring_system(num_vars, degrees, rank*ones(Int,length(degrees)))
    write_system(F, "examples/data/complexity_april/start_system/data_univ_start_system_$rank.txt")

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
    else
        global iter += 1
        open("examples/data/complexity_april/data_univ_complexity_$rank.txt", "a") do f
            write(f, "$num_steps\n")
        end
    end
end
