using DelimitedFiles
using Plots
using Statistics

function dt_statistics(data, val)
    initial_dt = 10.0^(-1*val)
    num_rows = size(data)[1]
    avg_dt = mean(data[:,2])
    avg_num_steps = mean(data[:,3])
    avg_diff = mean(data[:,4])
    return num_rows, avg_dt, avg_num_steps, avg_diff
end

function print_statistics(initial_dt, num_samples, avg_dt, avg_num_steps, avg_diff)
    if initial_dt != 1.0
        println("There are $num_samples entries with initial step size $initial_dt.")
    else
        println("There are $num_samples entries using the rigorous step size.")
    end
    println("Average timestep: $avg_dt")
    println("Average number of steps: $avg_num_steps")
    println("Average residual: $avg_diff\n")
end

function print_success(data, val)
    success = 0
    total = 0
    for diff in data[:,4]
        total += 1
        if diff < 10^(-14)
            success += 1
        end
    end
    println("For $val: of $total iterations, $success were successful.\n\n")
end


rank = 4

# initial_dt = [rigorous, 10^(-1), 10^(-2), 10^(-3), 10^(-4), 10^(-5)]
dt_exps = [0, 1, 2, 3, 4, 5]
for val in dt_exps
    data = readdlm("data_heuristic_$(rank)_$(val).txt")
    # iter avg_step_size num_steps diff result

    NUM_SAMPLES, AVG_DT, AVG_NUM_STEPS, AVG_DIFF = dt_statistics(data, val)
    print_statistics(10.0^(-1*val), NUM_SAMPLES, AVG_DT, AVG_NUM_STEPS, AVG_DIFF)
    print_success(data, val)
end
