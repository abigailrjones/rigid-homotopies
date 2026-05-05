using DelimitedFiles: readdlm
using Plots
# plotlyjs()
gr()

data = readdlm("examples/profiling/data/data_check_waring_complexity.txt")

base = 2

# scaling like num_vars
p = plot(data[1:end,1], data[1:end,2], xscale=:log2, yscale=:log2,
         legends=:bottomright, label="polynomial", xlabel="number of variables",
         ylabel="time (s)", title="Cost of evaluation")
plot!(data[1:end,1], (data[1:end,1])/base^24, label="linear", linestyle=:dash)

# scaling like num_vars^2
plot!(data[1:end,1], data[1:end,3], xscale=:log2, yscale=:log2, legends=:bottomright, label="system")
plot!(data[1:end,1], ((data[1:end,1]) .^ 2)/base^24, label="quadratic", linestyle=:dash)

savefig(p, "eval_complexity_waring.png")
