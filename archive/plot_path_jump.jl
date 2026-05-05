
using DelimitedFiles: readdlm
using Plots
plotlyjs()

data = readdlm("examples/data/data_path_jump.txt")
num_runs = length(data)

println("Path jumping is happening roughly $((num_runs-sum(data))/num_runs)% of the time.")

#=
plot(data[1:end,1], data[1:end,2], st=:scatter, c=1,
     label="",xlabel="dt",ylabel="final root",title="Three degree 4 \
     polynomials in four variables")

num_rows,_ = size(data)

for idx in 2:(num_rows-1)
    println(data[idx,2]/data[idx-1,2])
end
=#
