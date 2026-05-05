include("../../rigid_hom.jl")
include("../../start_system.jl")
using Colors
using LaTeXStrings
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
p2 = [c3/2^0.25,0,1] # BETTER
# p2 = [c3/2,c3/2,1] # WORKS

start_root = p2
M = build_unitary(p1,p2,3) * (norm(p2)/norm(p1))
start_system = [M, [1 0 0; 0 1 0; 0 0 1.0+0*im]]
path = build_path(start_system)

solve([f,g],2,3,[2,4],100,start_system,start_root,path,use_heuristic=true,mid_print=true,filename="examples/visual/real_test.txt",initial_dt=0.1)

data = readdlm("examples/visual/real_test.txt")
D = data[1:end-1,5:2:10]


# ---------------- PLOTTING ------------------------

xs = range(-4,6,length=100)
ys = range(-6,5,length=100)
F(x,y) = f([x,y,1])
G(x,y) = real(g([x,y,1]))

pal = palette([:black, :steelblue1], 11)
c = Plots.contour(xs,ys,F,levels=[0],ratio=1,legend=:none,color=pal[1],lw=2)#,label=L"f",legend=:topright)
Plots.contour!(c,xs,ys,G,levels=[0],color=:black,lw=2)#,label=L"g")

# Plots.scatter!(c,[p1[1]],[p1[2]],color=pal[end],markerstrokewidth=0,markershape=:star,markersize=10)
Plots.scatter!(c,[p2[1]],[p2[2]],color=:black,markerstrokewidth=0)

shifted_F(x,y,W) = real(f(W * [x,y,1]))
idx = 1
for t in 1:-0.1:0
    W, _ = path(t)
    # root = real(W' * p1)
    Plots.scatter!(c,[D[idx,1]/D[idx,end]],[D[idx,2]/D[idx,end]],color=pal[idx],markerstrokewidth=0)
    if(!isreal(W)) println("Not real at t=$t") end
    Plots.contour!(c,xs,ys,(x,y)->shifted_F(x,y,W),levels=[0],color=pal[idx],lw=2)
    global idx += 1
end

savefig(c, "examples/visual/images/real_track.png")
