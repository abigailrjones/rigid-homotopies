include("../../rigid_hom.jl")
include("../../start_system.jl")
using Colors
using Plots
using DelimitedFiles

c1 = 4
c2 = 1
R = 1
Mf = [1/c1 0 0; 0 1/c2 0; 0 0 R*im]
f = WaringPoly(3,2,3,Mf)
# p1 = [R*c1,0,1]
p1 = [0,R*c2,1] # WORKS BETTER

c3 = 3
Mg = [1 1 0; 1 -1 0; 0 0 c3*exp((pi/4)*im)]
g = WaringPoly(3,4,3,Mg)
# p2 = [c3/2^0.25,0,1] # BETTER
p2 = [c3/2,c3/2,1] # WORKS

start_root = p2
M = build_unitary(p1,p2,3)
MM = M * (norm(p2)/norm(p1))
start_system = [M, [1 0 0; 0 1 0; 0 0 1.0+0*im]]
path = build_path(start_system)

solve([f,g],2,3,[2,4],1,start_system,start_root,path,use_heuristic=false,mid_print=true,filename="examples/visual/real_path.txt")
# solve([f,g],2,3,[2,4],1,use_heuristic=false,mid_print=true,filename="examples/visual/real_path.txt")

#=
data = readdlm("examples/visual/real_test.txt")
D = data[1:end-1,5:2:10]
=#


# ---------------- PLOTTING ------------------------

#=
xs = range(-5,5,length=100)
ys = range(-5,5,length=100)
F(x,y) = f([x,y,1])
G(x,y) = real(g([x,y,1]))

# pal = palette([:black, :steelblue1], 101)
data = range(-5,5,length=20)
X,Y,Z = mgrid(data,data,data)
vol = [f([X[idx],Y[idx],Z[idx]]) for idx in 1:length(X)]
trace = volume(x=X[:],y=Y[:],z=Z[:],value=vol[:],isomin=-1.0,isomax=1.0,isorange=0.5)
PlotlyJS.plot(trace)
# c = Plots.surface(xs,ys,F,fillalpha=0.05,contours_z_show=true,contours_z_start=0,contours_z_end=0.01,contours_z_size=1)#,ratio=1,legend=:none,color=pal[1],lw=2)
# df = [F(x,y) for x in xs, y in ys]
# contour!(xs,ys,df,levels=[0])

# contour!(c,xs,ys,G,levels=[0],color=:black,lw=2)

# scatter!(c,[p1[1]],[p1[2]],color=pal[end],markerstrokewidth=0)
# scatter!(c,[p2[1]],[p2[2]],color=:black,markerstrokewidth=0)

shifted_F(x,y,W) = real(f(W * [x,y,1]))
idx = 1
for t in 1:-0.01:0
    # root = real(W' * p1)
    # scatter!(c,[D[idx,1]/D[idx,end]],[D[idx,2]/D[idx,end]],color=pal[idx],markerstrokewidth=0)
    # if(!isreal(W)) println("Not real at t=$t") end
    W, _ = path(t)
    Plots.surface!(c,xs,ys,(x,y)->shifted_F(x,y,W)+t*100,fillalpha=0.05,contours_z_show=true,contours_z_start=t,contours_z_end=t*100+1e-8,contours_z_size=1,color=pal[idx],lw=2)
    global idx += 1
end
=#
