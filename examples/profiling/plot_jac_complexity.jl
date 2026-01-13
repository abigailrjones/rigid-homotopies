using DelimitedFiles: readdlm
using Plots
# plotlyjs()
gr()

data = readdlm("examples/profiling/data/data_check_jac_complexity.txt")

base = 2

p1 = plot(data[1:end,1], data[1:end,2],
         xscale=:log2, yscale=:log2,
         legends=:topleft, label="grad-for", xlabel="number of variables",
         ylabel="time (s)")
plot!(p1, data[1:end,1], data[1:end,4], label="grad-rev")
plot!(p1, data[1:end,1], data[1:end,6], alpha=0.25, label="old-grad")
plot!(p1, data[1:end,1], (data[1:end,1])/base^20, linestyle=:dashdot, linecolor=:black, label="linear")

p2 = plot(data[1:end,1], data[1:end,3],
         xscale=:log2, yscale=:log2,
         legends=:topleft, label="jac-for", xlabel="number of variables",
         ylabel="time (s)")
# plot!(data[1:end,1], data[1:end,3], label="jac-for")
plot!(p2, data[1:end,1], data[1:end,5], label="jac-rev")
plot!(p2, data[1:end,1], data[1:end,7], alpha=0.25, label="old-jac")
plot!(p2, data[1:end,1], ((data[1:end,1]) .^ 2)/base^20, linestyle=:dash, linecolor=:black,  label="quadratic")


p3 = plot(data[1:end,1], ones(length(data[1:end,1]))/base^20, xscale=:log2,
          yscale=:log2, legends=:topleft, xlabel="number of
          variables", ylabel="time (s)", linestyle=:dashdotdot,
          linecolor=:black, label="constant")
plot!(data[1:end,1], (data[1:end,1])/base^20, linestyle=:dashdot, linecolor=:black, label="linear")
plot!(data[1:end,1], ((data[1:end,1]) .^ 2)/base^20, linestyle=:dash, linecolor=:black,  label="quadratic")
plot!(data[1:end,1], ((data[1:end,1]) .^ 3)/base^20, linestyle=:dot, linecolor=:black, label="cubic")

plot(p3,p1,p2)#,title="Cost of computing derivatives")

# savefig(p, "deriv_complexity_waring.png")
