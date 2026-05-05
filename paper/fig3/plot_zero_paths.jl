using Plots
using Plots.PlotMeasures
using Colors
using LaTeXStrings
using DelimitedFiles

function plot_zero_path(data)
    # time condnum gammafrob dt root
    num_rows, num_cols = size(data)
    num_rows -= 1

    p = plot3d(camera=(15,20),right_margin=-10mm,size=(600,400))

    z_lower = minimum(data[1:num_rows,6:2:num_cols])
    y_lower = minimum(data[1:num_rows,5:2:num_cols])
    y_higher = maximum(data[1:num_rows,5:2:num_cols])
    for i in 5:2:num_cols
        # floor shadow
        if num_rows > 1000
            len = length(data[1:4:num_rows,1])
            plot!(p, data[1:4:num_rows,1], data[1:4:num_rows,i],
                  z_lower*ones(len), line_z=data[1:4:num_rows,3], lw=1.25,
                  alpha=0.05, legend=:none)
        else
            plot!(p, data[1:num_rows,1], data[1:num_rows,i],
                  z_lower*ones(num_rows), line_z=data[1:num_rows,3], lw=1.25,
                  alpha=0.05, legend=:none)
        end
        # max gammafrob line
        # plot!(p, ones(num_rows)*data[argmax(data[1:num_rows,3]),1], range(y_lower,y_higher,length=num_rows), z_lower*ones(num_rows), linecolor=:black, linestyle=:dot, alpha=0.25, lw=0.75)

        # wall shadow
        # plot!(p, data[1:num_rows,1], y_higher*ones(num_rows), data[1:num_rows,i+1], line_z=data[1:num_rows,3], alpha=0.25)#, margin=1px)
    end
    for i in 5:2:num_cols
        plot!(p, data[1:num_rows,1], data[1:num_rows,i], data[1:num_rows,i+1], line_z=data[1:num_rows,3], lw=2, cbar=false)
    end

    xlabel!(L"t")
    ylabel!("Re")
    zlabel!("Im")
    println("Number of iterations: $(round(data[end,1],sigdigits=3))")

    return p
end

for filename in ["waring345", "waring334"]
    pathname = "$filename.txt"
    data = readdlm(pathname)
    num_rows, num_cols = size(data)
    num_rows -= 1

    default(titlefont=(10,"Computer Modern"),guidefont=(8,"Computer Modern"),
            tickfont=(6,"Computer Modern"),framestyle=:axes)
    p1 = plot_zero_path(data)
    z_min, z_max = extrema(data[1:num_rows,3])

    colorbar_data = collect(range(z_min, z_max, length=100))
    p2 = heatmap([1], colorbar_data, reshape(colorbar_data, :, 1), cbar=false,
                 xaxis=false,
                 ymirror=true, left_margin=-10mm,
                 title=L"\hat{\gamma}_{\mathrm{Frob}}",
                 yguidefontrotation=270,
                 cmap=:plasma,
                 framestyle=:grid)
    xaxis = Plots.get_axis(Plots.get_subplot(p2,1),:x)
    yaxis = Plots.get_axis(Plots.get_subplot(p2,1),:y)
    zaxis = Plots.get_axis(Plots.get_subplot(p2,1),:z)
    xaxis[:gridalpha] = 0
    yaxis[:gridalpha] = 0
    zaxis[:gridalpha] = 0

    l = @layout [
                 a{0.95w} [ _
                           b{0.85h}
                            _ ]
                ]
    p = plot(p1,p2,layout=l,dpi=600,xticks=[0,0.5,1])
    savefig(p, "$filename.png");
end
