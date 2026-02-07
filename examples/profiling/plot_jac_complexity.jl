using DelimitedFiles: readdlm
using Plots
# plotlyjs()
gr()

# Waring evaluation complexity
data = readdlm("examples/profiling/data/data_check_waring_complexity.txt")
base = 2

# scaling like num_vars
p0 = plot(data[1:end,1], data[1:end,2], xscale=:log2, yscale=:log2,
          legends=:topleft, label="polynomial", xlabel="number of variables",
          ylabel="time (s)", title="Cost of evaluating Waring")
plot!(p0, data[1:end,1], (data[1:end,1])/base^24, label="linear", linestyle=:dot, linecolor=:black, alpha=0.8)

# scaling like num_vars^2
plot!(p0, data[1:end,1], data[1:end,3], xscale=:log2, yscale=:log2, label="system")
plot!( p0, data[1:end,1], ((data[1:end,1]) .^ 2)/base^24, label="quadratic", linestyle=:dash, linecolor=:black, alpha=0.8)


# Gradient / jacobian complexity
data = readdlm("examples/profiling/data/data_check_jac_complexity.txt")

base = 2

p1 = plot(data[1:end,1], data[1:end,2],
         xscale=:log2, yscale=:log2,
         legends=:topleft, label="Enzyme (forward)", xlabel="number of variables",
         ylabel="time (s)",title="Gradient")
plot!(p1, data[1:end,1], data[1:end,4], label="Enzyme (reverse)")
plot!(p1, data[1:end,1], data[1:end,6], label="Zygote")
plot!(p1, data[1:end,1], (data[1:end,1])/base^20, linestyle=:dot, linecolor=:black, alpha=0.8, label="linear")

p2 = plot(data[1:end,1], data[1:end,3],
         xscale=:log2, yscale=:log2,
         legends=:topleft, label="Enzyme (forward)", xlabel="number of variables",
         ylabel="time (s)",title="Jacobian")
# plot!(data[1:end,1], data[1:end,3], label="jac-for")
plot!(p2, data[1:end,1], data[1:end,5], label="Enzyme (reverse)")
plot!(p2, data[1:end,1], data[1:end,7], label="Zygote")
plot!(p2, data[1:end,1], ((data[1:end,1]) .^ 2)/base^20, linestyle=:dash, linecolor=:black, alpha=0.8, label="quadratic")


#=
p3 = plot(data[1:end,1], ones(length(data[1:end,1]))/base^20, xscale=:log2,
          yscale=:log2, legends=:topleft, xlabel="number of
          variables", ylabel="time (s)", linestyle=:dashdotdot,
          linecolor=:black, label="constant")
plot!(data[1:end,1], (data[1:end,1])/base^20, linestyle=:dot, linecolor=:black, label="linear")
plot!(data[1:end,1], ((data[1:end,1]) .^ 2)/base^20, linestyle=:dash, linecolor=:black,  label="quadratic")
plot!(data[1:end,1], ((data[1:end,1]) .^ 3)/base^20, linestyle=:dashdot, linecolor=:black, label="cubic")
=#

l = @layout [ a{0.5h}
             [grid(1,2)] ]

p = plot(p0,p1,p2,layout=l,
         titlefontsize=9,
         legendfontsize=5,
         xtickfontsize=7,ytickfontsize=7,
         xguidefontsize=7,yguidefontsize=7)

savefig(p, "complexity_waring.png")
