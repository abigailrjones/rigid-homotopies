using Plots
using Plots.PlotMeasures
using Colors
using LaTeXStrings
using DelimitedFiles

default(titlefont=(20,"Computer Modern"),guidefont=(15,"Computer Modern"),
        tickfont=(12,"Computer Modern"),framestyle=:axes)

filename = "data/data_plot_table1.txt"
data = readdlm(filename)

categories = ["("*val[4:6]*")" for val in data[1:9,1]]
n1 = [Float64(val) for val in data[1:9,2]]
n2 = [Float64(val) for val in data[10:end,2]]

bar(categories,n1,label=L"n=1",color=:darkorange,linecolor=:match,alpha=1.0,bar_width=0.5,dpi=600,fillrange=0)
bar!(categories,n2,label=L"n=2",color=:blue,linecolor=:match,alpha=1.0,bar_width=0.5,fillrange=0)
bar!(categories,n1,label=:none,color=:darkorange,linecolor=:match,alpha=1.0,bar_width=0.5,fillrange=0,legend=:topleft,legendfontsize=12,bottom_margin=2mm)

# yticks!([10^4,10^5,10^6])

xlabel!("\$(D,r)\$ configurations")
ylabel!("Iterations")
savefig("table1_bar.png")
