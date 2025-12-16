using DelimitedFiles: readdlm
using Plots
# plotlyjs()
gr()

data = readdlm("examples/profiling/data/data_check_jac_complexity.txt")

base = 2

# scaling like num_vars
p = plot(base .^ data[1:end,1], data[1:end,2], xscale=:log2, yscale=:log2,
         legends=:topleft, label="gradient", xlabel="number of variables",
         ylabel="time (s)", title="Cost of computing derivatives")
plot!(base .^ data[1:7,1], ones(length(data[1:7,1])) * 2^(-14), linestyle=:dash, label="constant")
plot!(base .^ data[7:end,1], (base .^ data[7:end,1])/base^21, linestyle=:dash, label="linear")

# scaling like num_vars
plot!(base .^ data[1:end,1], data[1:end,3], label="jacobian")
plot!(base .^ data[1:7,1], ((base .^ data[1:7,1]) .^ 2)/base^13, linestyle=:dash, label="quadratic")
plot!(base .^ data[7:end,1], ((base .^ data[7:end,1]) .^ 3)/base^20, linestyle=:dash, label="cubic")

savefig(p, "deriv_complexity_waring.png")
