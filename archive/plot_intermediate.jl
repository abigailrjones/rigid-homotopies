using Colors
using PlotlyJS
using Plots
using Plots.PlotMeasures
using DelimitedFiles

data = readdlm("examples/data/new_intermediate_roots.txt")

function plot_zero_path(data, upper, lower)#, filename)
    DATA = data[upper:lower-1,:]
    num_rows, num_cols = size(DATA)
    # num_cols = 2

    pal = palette([:navy, :green1], num_cols)
    p = plot3d()#camera=(90,90))
    for i in 2:2:num_cols
        plot!(p, DATA[1:num_rows,1], DATA[1:num_rows,i], DATA[1:num_rows,i+1], seriescolor=pal[i])#, margin=1px)
    end
    z_lower, _ = zlims(p)
    _, y_higher = ylims(p)
    for i in 2:2:num_cols
        plot!(p, DATA[1:num_rows,1], DATA[1:num_rows,i], z_lower*ones(num_rows), seriescolor=coloralpha(pal[i],0.25))#, margin=1px)
        plot!(p, DATA[1:num_rows,1], y_higher*ones(num_rows), DATA[1:num_rows,i+1], seriescolor=coloralpha(pal[i],0.25))#, margin=1px)
    end
    for i in 2:2:num_cols
        plot!(p, DATA[1:num_rows,1], DATA[1:num_rows,i], DATA[1:num_rows,i+1], seriescolor=pal[i])#, margin=1px)
    end
    #=
    p = plot()
    for i in 2:2:num_cols
        plot!(p, DATA[1:num_rows,1], DATA[1:num_rows,i], seriescolor=pal[i])#, margin=1px)
        plot!(p, DATA[1:num_rows,1], DATA[1:num_rows,i+1], seriescolor=pal[i])#, margin=1px)
    end
    =#

    plot!(legend=:none)
    xlabel!("time")
    ylabel!("real")
    zlabel!("imag")
    # title!("$(data[lower][1])")
    println("Number of iterations: $(round(data[lower][1],sigdigits=3))")

    # savefig(p, filename)
    return p
end

p1 = plot_zero_path(data, 2, 3240)
p2 = plot_zero_path(data, 3242, 5808)
p3 = plot_zero_path(data, 5810, 39765)
p4 = plot_zero_path(data, 39767, 49482)

p = plot(p2,p3,layout=(1,2),xtickfontsize=4,ytickfontsize=4,ztickfontsize=4,
         xguidefontsize=6,yguidefontsize=6,zguidefontsize=6,dpi=300);#,margin=0mm)

savefig(p, "newzeropath.png");

