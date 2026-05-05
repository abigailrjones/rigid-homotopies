using DelimitedFiles
using Statistics
using Printf

function rank_statistics(rank, data)
    num_rows = size(data)[1]
    avg_iterations = mean(data[:,1])
    median_iterations = median(data[:,1])
    min_iterations = minimum(data[:,1])
    max_iterations = maximum(data[:,1])
    min_dt = minimum(data[:,2])
    max_dt = maximum(data[:,3])
    avg_dt = mean(data[:,4])
    median_dt = median(data[:,4])
    min_gammaprob = minimum(data[:,5])
    max_gammaprob = maximum(data[:,6])
    avg_gammaprob = mean(data[:,7])
    median_gammaprob = median(data[:,7])
    min_condnum = minimum(data[:,8])
    max_condnum = maximum(data[:,9])
    avg_condnum = mean(data[:,10])
    median_condnum = median(data[:,10])
    return num_rows, avg_iterations, median_iterations, min_iterations,
    max_iterations, min_dt, max_dt, avg_dt, median_dt, min_gammaprob,
    max_gammaprob, avg_gammaprob, median_gammaprob, min_condnum, max_condnum,
    avg_condnum, median_condnum
end

function print_statistics(num_vars, deg, rank, num_rows, avg_iterations,
        median_iterations, min_iterations, max_iterations, min_dt, max_dt,
        avg_dt, median_dt, min_gammaprob, max_gammaprob, avg_gammaprob,
        median_gammaprob, min_condnum, max_condnum, avg_condnum,
        median_condnum)
    println("There are $num_rows entries with $(num_vars) variables, degree $deg, rank $rank.")
    println("")
    println("mean # of iterations: $(round(avg_iterations,sigdigits=3))")
    println("median # of iterations: $(round(median_iterations,sigdigits=3))")
    # println("min # of iterations: $min_iterations")
    # println("max # of iterations: $max_iterations")
    println("")
    println("mean step size: $(round(avg_dt,sigdigits=3))")
    println("median step size: $(round(median_dt,sigdigits=3))")
    # println("min step size: $(round(min_dt,sigdigits=3))")
    # println("max step size: $(round(max_dt,sigdigits=3))")
    println("")
    println("mean gammaprob: $(round(avg_gammaprob,digits=2))")
    println("median gammaprob: $(round(median_gammaprob,digits=2))")
    # println("min gammaprob: $(round(min_gammaprob,digits=2))")
    # println("max gammaprob: $(round(max_gammaprob,digits=2))")
    println("")
    println("mean condition number: $(round(avg_condnum,digits=2))")
    println("median condition number: $(round(median_condnum,digits=2))")
    # println("min condition number: $(round(min_condnum,digits=2))")
    # println("max condition number: $(round(max_condnum,digits=2))")
    println("")
end

function val_to_latex(val, prec)
    s = @sprintf("%.*e", prec, val)
    local b, e
    flag = true
    try
        b, e = collect(match(r"(?<b>\d+.\d+)e\+0(?<e>\d+)",s))
    catch
        flag = false
        b, e = collect(match(r"(?<b>\d+.\d+)e\-0(?<e>\d+)",s))
    end
    return flag ? "\$$b \\times 10^{$e}\$" : "\$$b \\times 10^{-$e}\$"
end

function write_statistics(num_vars, deg, rank, num_rows, avg_iterations,
        median_iterations, min_iterations, max_iterations, min_dt, max_dt,
        avg_dt, median_dt, min_gammaprob, max_gammaprob, avg_gammaprob,
        median_gammaprob, min_condnum, max_condnum, avg_condnum,
        median_condnum, filename)
    open("data/$filename.txt", "a") do f
        write(f, "\$($(num_vars-1), $deg, $rank)\$ & $(val_to_latex(avg_iterations,2)) & $(val_to_latex(median_iterations,2)) & $(val_to_latex(avg_dt,2)) & $(val_to_latex(median_dt,2)) & \$$(round(avg_gammaprob,digits=2))\$ & \$$(round(median_gammaprob,digits=2))\$ & \$$(round(avg_condnum,digits=2))\$ & \$$(round(median_condnum,digits=2))\$ \\\\ \n")
    end
end

@assert length(ARGS) == 3

num_vars = parse(Int,ARGS[1])
num_funcs = num_vars - 1
deg = parse(Int,ARGS[2])
degrees = ones(Int,num_funcs)*deg
rank = parse(Int,ARGS[3])
filename = "data_table1"

data = readdlm("data/data_tracking_$(num_vars)_$(deg)_$(rank).txt")
num_rows, avg_iterations, median_iterations, min_iterations, max_iterations,
min_dt, max_dt, avg_dt, median_dt, min_gammaprob, max_gammaprob, avg_gammaprob,
median_gammaprob, min_condnum, max_condnum, avg_condnum, median_condnum =
rank_statistics(rank, data)
println("$num_rows")
write_statistics(num_vars, deg, rank, num_rows, avg_iterations,
                 median_iterations, min_iterations, max_iterations, min_dt,
                 max_dt, avg_dt, median_dt, min_gammaprob, max_gammaprob,
                 avg_gammaprob, median_gammaprob, min_condnum, max_condnum,
                 avg_condnum, median_condnum, filename)
