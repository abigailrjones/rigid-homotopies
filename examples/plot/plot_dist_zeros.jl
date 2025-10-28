
using DelimitedFiles: readdlm
using Plots
plotlyjs()

data = readdlm("examples/data/data_dist_zeros.txt")
println(length(data[1:end,end]))
# rows, cols = size(data)

# FIXME need to clean NaN case out so it doesn't distract
nan_idx = argmax(data[1:end,end])
if isnan(data[nan_idx,4])
    data_sans_nan = []
    append!(data_sans_nan, data[1:nan_idx-1,end])
    append!(data_sans_nan, data[nan_idx+1:end,end])
else
    data_sans_nan = data[1:end,end]
end

plot(bar(1:length(data_sans_nan),
         data_sans_nan[1:end,end],
         bar_width=0.8))
