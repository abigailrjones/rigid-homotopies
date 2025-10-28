
using DelimitedFiles: readdlm
using Plots
plotlyjs()

data = readdlm("examples/data/data_timestep.txt")

plot(data[1:end,1], data[1:end,2], yscale=:log10, c=1, label="")
plot!(data[1:end,1], data[1:end,2], st=:scatter, yscale=:log10, c=1,
      label="",xlabel="N",ylabel="timestep",title="(N-1) degree N \
      polynomials in N variables")
#=
num_rows,_ = size(data)

for idx in 2:(num_rows-1)
    println(data[idx,2]/data[idx-1,2])
end
=#
