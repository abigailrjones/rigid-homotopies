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

num_vars = 2
degrees = [3]
num_funcs = length(degrees)
max_iter = 1_000_000
mid_print = true

rank = 4

if false
    for val in [0,1,2,3,4,5]
        open("data_heuristic_$(rank)_$(val).txt", "w") do f
            # empty file
        end
    end
end

# run iterations
for iter in 1:100
    F = build_waring_system(num_vars, degrees, rank*ones(Int,length(degrees)))
    write_system(F, "data_heuristic_start_system_$rank.txt")
    start_system, start_root = build_start_system(F, degrees, num_vars)

    local result
    local num_steps
    local avg_step_size
    local rigorous_result
    try # rigorous
        rigorous_result, num_steps, min_step_size, max_step_size,
        avg_step_size, avg_duration, min_gammaprob, max_gammaprob,
        avg_gammaprob, min_condnum, max_condnum, avg_condnum = solve(F,
                                                                     num_funcs,
                                                                     num_vars,
                                                                     degrees,
                                                                     max_iter,
                                                                     start_system,
                                                                     start_root;
                                                                     use_heuristic=false,
                                                                     mid_print=mid_print)
    catch e
        println("Error on run $iter using rigorous timestep.")
        throw(e)
    else
        open("data_heuristic_$(rank)_0.txt", "a") do f
            write(f, "$iter $avg_step_size $num_steps 0.0 $rigorous_result\n")
        end
    end
    for val in [1,2,3,4,5]
        try # heuristic, dt=10^(-val)
            result, num_steps, min_step_size, max_step_size, avg_step_size,
            avg_duration, min_gammaprob, max_gammaprob, avg_gammaprob,
            min_condnum, max_condnum, avg_condnum = solve(F, num_funcs,
                                                          num_vars,
                                                          degrees,
                                                          max_iter,
                                                          start_system,
                                                          start_root;
                                                          use_heuristic=true,
                                                          mid_print=mid_print,
                                                          initial_dt=10.0^(-1*val))
        catch e
            println("Error on run $iter using heuristic timestep.")
            throw(e)
        else
            diff = sqrt(sum((rigorous_result - result).^2))
            open("data_heuristic_$(rank)_$(val).txt", "a") do f
                write(f, "$iter $avg_step_size $num_steps $diff $result\n")
            end
        end
    end
end
