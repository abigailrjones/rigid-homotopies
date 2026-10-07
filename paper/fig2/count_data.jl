using DelimitedFiles

@assert length(ARGS) == 3

num_vars = parse(Int,ARGS[1])
num_funcs = num_vars - 1
deg = parse(Int,ARGS[2])
degrees = ones(Int,num_funcs)*deg
rank = parse(Int,ARGS[3])

data = readdlm("data/data_tracking_$(num_vars)_$(deg)_$(rank).txt")
println("$(num_vars) $(deg) $(rank) : $(size(data)[1])")
