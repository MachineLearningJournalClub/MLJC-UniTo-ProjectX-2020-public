using DelimitedFiles
using Plots

tensor = readdlm("C:/Users/jamesjun/Desktop/Simone/ProjectX2020/BCI/ProjectX2020/Interpolation/tensor.txt")
tensor = reshape(tensor, (420,420,31))
tensor = permutedims(tensor, [2,1,3])
tensor = tensor
heatmap(tensor[:,:,30])
contour(tensor[:,:,30], levels = 0.1:0.5)
