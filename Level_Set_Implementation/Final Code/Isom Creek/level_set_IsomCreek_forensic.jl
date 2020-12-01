#___  ____       ___ _____   _   _       _ _
#|  \/  | |     |_  /  __ \ | | | |     (_| |
#| .  . | |       | | /  \/ | | | |_ __  _| |_ ___
#| |\/| | |       | | |     | | | | '_ \| | __/ _ \
#| |  | | |___/\__/ | \__/\ | |_| | | | | | || (_) |
#_______\_____\____/ \____/  _____|_|___|_____\_____ _____ _____
#| ___ \        (_)         | | \ \ / / / __  |  _  / __  |  _  |
#| |_/ _ __ ___  _  ___  ___| |_ \ V /  `' / /| |/' `' / /| |/' |
#|  __| '__/ _ \| |/ _ \/ __| __|/   \    / / |  /| | / / |  /| |
#| |  | | | (_) | |  __| (__| |_/ /^\ \ ./ /__\ |_/ ./ /__\ |_/ /
#\_|  |_|  \___/| |\___|\___|\__\/   \/ \_____/\___/\_____/\___/
#              _/ |
#             |__/
#
# This code is part of the proposal of the team "MLJC UniTo" - University of Turin
# for "ProjectX 2020" Climate Change for AI.
# The code is licensed under MIT 3.0
# Please read readme or comments for credits and further information.

# Compiler: Julia 1.5

# Short description of this file: Level set implementation for Isom Creek simulation
#with a final boundary condition

using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using DelimitedFiles

print("Precompiling Done")

##  DECLARATIONS
@parameters t x y θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y

## DOMAINS AND OPERATORS

#Real world data to model mesh scaling

Lx = 10000   #Domain x width in meters
Ly = 10000   #Domain y width in meters
Lt = 3600    #Total simulation time in seconds

xRes = 30    #Resolution of x axis in meters, mesh cell width
yRes = 30    #Resolution of y axis in meters, mesh cell width
tRes = 60    #Time resultion, seconds

# Discretization
xwidth      = 10.0      #ft
ywidth      = 10.0
tmax        = 10.0      #min
xScale      = 1.0
yScale      = 1.0
xMeshNum    = max(round(Lx/xRes),500)
yMeshNum    = max(round(Ly/yRes),500)
tMeshNum    = max(round(Lt/tRes),20)
dx  = xwidth/xMeshNum
dy  = ywidth/yMeshNum
dt  = tmax/tMeshNum

#Domain, fire position and shape

shape      = ["zeroIsVertex","zeroIsCenter"]
x0         = [5.0]      #Fire ingnition coordinates
y0         = [5.0]
xSpread    = [1.0]      #Fire shape factors
ySpread    = [0.7]
tIgnition  = [10.0]      #Fire's time of ignition
amplitude  = [2.5]      #Fire's initial spread (radius if circle)

domainShape = shape[1]

if domainShape == shape[1]
    domains = [t ∈ IntervalDomain(0.0,tmax),
           x ∈ IntervalDomain(0.0,xwidth),
           y ∈ IntervalDomain(0.0,ywidth)]
           xs = 0.0 : dx : xwidth
           ys = 0.0 : dy : ywidth
elseif domainShape == shape[2]
    domains = [t ∈ IntervalDomain(0.0,tmax),
           x ∈ IntervalDomain(-xwidth*0.5,xwidth*0.5),
           y ∈ IntervalDomain(-ywidth*0.5,ywidth*0.5)]
           xs = -xwidth*0.5: dx : xwidth*0.5
           ys = -ywidth*0.5: dy : ywidth*0.5
end

#Terrain interpolation parameters

p0 = 316.977
p1 = -31.7087
p2 = 42.2348
p3 = -12.7828
p4 = 1.26132
p5 = -0.0386497
p6 = 38.5245
p7 = 9.65925
p8 = -4.35081
p9 = 0.554218
p10 = -0.0245952
p11 = -21.5973
p12 = 1.79071
p13 = 2.49292
p14 = -0.231523

#Terrain gradient's components
Dxz = 0.01*(p1 + 2*p2*x + 3*p3*x*x + 4*p4*x*x*x + 5*p5*x*x*x*x + p11*y + 2*p12*y*x + p13*y*y + 2*p14*y*y*x)
Dyz = 0.01*(p6 + 2*p7*y + 3*p8*y*y + 4*p9*y*y*y + 5*p10*y*y*y*y + p11*x + p12*x*x + 2*p13*x*y + 2*p14*x*x*y)

#Domain discretization
x_s     = [x for x in xs][1:end-1]
y_s     = [y for y in ys][1:end-1]
hgt(x,y) = 0.01*(p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x + p5*x*x*x*x*x + p6*y + p7*y*y + p8*y*y*y + p9*y*y*y*y + p10*y*y*y*y*y +
           p11*x*y + p12*x*x*y + p13*x*y*y + p14*x*x*y*y)
z_s     = [hgt(x,y) for x in x_s for y in y_s]
z_s     = reshape(z_s,(500,500))

Plots.plot(x_s,y_s,z_s, st=:surface)    #terrain plot

Uwind = [0.0, 0.0]  #wind vector

gn   = (Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^0.5 #gradient's norm
∇u   = [Dx(u(t,x,y,θ)), Dy(u(t,x,y,θ))]
∇z   = [Dxz,Dyz]
n    = ∇u/gn              #normal versor
normalized = ((Uwind[1]*n[1] + Uwind[2]*n[2])^2)^0.5 #inner product between wind and normal vector

## FUEL PARAMETERS

#Fuel parameters given by namelist.fire from WRF-Sfire
windrf = [0.36, 0.36, 0.44,  0.55,  0.42,  0.44,  0.44, 0.36, 0.36, 0.36,  0.36,  0.43,  0.46, 1e-7]
fgi    = [0.166, 0.897, 1.076, 2.468, 0.785, 1.345, 1.092, 1.121, 0.780, 2.694, 2.582, 7.749, 13.024, 1.e-7]
fueldepthm = [0.305, 0.305, 0.762, 1.829, 0.61,  0.762, 0.762, 0.061, 0.061, 0.305, 0.305, 0.701, 0.914, 0.305]
savr = [3500., 2784., 1500., 1739., 1683., 1564., 1562., 1889., 2484., 1764., 1182., 1145., 1159., 3500.]
fuelmce = [0.12, 0.15, 0.25, 0.20, 0.20, 0.25, 0.40, 0.30, 0.25, 0.25, 0.15, 0.20, 0.25, 0.12]
fueldens = [32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.] #from namelist.fire: "! 32 if solid, 19 if rotten"
st = [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555]
se = [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
cmbcnst  = 17.433e+06
fuelmc_g = 0.08  #or 0.09
weight = [7.,  7.,  7., 180., 100., 100., 100., 900., 900., 900., 900., 900., 900., 7.]

FuelNumber = 9
fuel_scale = 900

a     = windrf[FuelNumber]          #those numbers should be dependent on position
zf    =
z0    =
w     = weight[FuelNumber]/60               #seconds to minutes
wl    = fgi[FuelNumber]*0.204816144         #kg*m^-2 to lb*ft^-2
δm    = fueldepthm[FuelNumber]*3.28084      #m to ft
sigma = savr[FuelNumber]/3.28084            #m^-1 to ft^-1
Mx    = fuelmce[FuelNumber]
ρP    = fueldens[FuelNumber]*0.0624279606   #kg*m^-3 to lb*ft^-3
ST    = st[FuelNumber]
SE    = se[FuelNumber]
h     = cmbcnst*2.6839192e-5    #J*m^-3 to BTU*ft^-3
Mf    = fuelmc_g

## FIRE SPREAD RATE EQUATIONS

river_1 = 1 - 1/(1 + exp(-(x - r0)*1000))=#
tanϕ = sum(∇z.*n)
βop  = 3.348*sigma^(-0.8189)        #from Rothermel, eq (37)
U    = normalized                   #wind correction factor
w0   = wl/(1 + Mf)
ρb   = w0/δm                        #different from paper for units reasons
β    = ρb/ρP
ξ    = exp((0.792 + 0.618*sigma^0.5)*(β+0.1))/(192 + 0.25965*sigma)
ηs   = 0.174*(SE^(-0.19))
ηM   = 1 - 2.59*Mf/Mx + 5.11*(Mf/Mx)^2 - 3.52*(Mf/Mx)^3
wn   = w0/(1 + ST)
Γmax = (sigma^(1.5))/(495 + 0.594*sigma^(1.5))
A    = 1/(4.77*sigma^(0.1) - 7.27)
Γ    = Γmax*(β/βop)^A*exp(A*(1 - β/βop))
ϵ    = exp(-138/sigma)
Qig  = 250*β + 1116*Mf
C    = 7.47*exp(-0.133*sigma^0.55)
Ua   = a*U
E    = 0.715*exp(-0.000359*sigma)
IR   = Γ*wn*h*ηM*ηs
R0   = IR*ξ/(ρb*ϵ*Qig)              #spread rate without wind
ϕw   = C*max(Ua^β, (β/βop)^E)       #wind factor
ϕS   = 5.275*β^(-0.3)*tanϕ^2        #slope factor
S    = fuel_scale*R0*(1 + ϕw + ϕS)         #fire spread rate

eq = Dt(u(t,x,y,θ)) + S*gn ~ 0      #LEVEL SET EQUATION

initialCondition = (((xScale*(x-x0[1]))^2)*xSpread[1] + ((yScale*(y-y0[1]))^2)*ySpread[1])^0.5 - amplitude[1]   #Distance from ignition

#Multiple ignition points
#=
if length(x0) > 2
    for b = 2:length(x0)
        initialCondition = min(initialCondition, (((xScale*(x-x0[b]))^2)*xSpread[b] + ((yScale*(y-y0[b]))^2)*ySpread[b])^0.5 - amplitude[b])
    end
end
=#

bcs = [u(tIgnition[1],x,y,θ) ~ initialCondition]  #from literature

## NEURAL NETWORK
n = 16   #neuron number
maxIters = 3000     #number of iterations

chain = FastChain(FastDense(3,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))   #Neural network from Flux library

q_strategy = NeuralPDE.QuadratureTraining(algorithm =CubaCuhre(),reltol=1e-8,abstol=1e-8,maxiters=100)  #Training strategy

discretization = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain,strategy = q_strategy)


indvars = [t,x,y]   #phisically independent variables
depvars = [u]       #dependent (target) variable

dim = length(domains)

losses = []
cb = function (p,l)     #loss function handling
    println("Current loss is: $l")
    append!(losses, l)
    return false
end

pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
prob = discretize(pde_system, discretization)

a_1 = time_ns()

res = GalacticOptim.solve(prob, GalacticOptim.ADAM(0.08), cb = cb, maxiters=maxIters) #allow_f_increase = false,

b_1 = time_ns()
print(string("Training time = ",(b_1-a_1)/10^9))
#initθ = res.minimizer

pars = open(readdlm,"/Julia_implementation/LevelSetEq/Isom Creek/params_level_set_Isom_Creek.txt")  #to import parameters from previous training (change also line 249 accordingly)
pars

phi = discretization.phi


##VISUALIZATION

extrapolate  = false     #allows extrapolating values outside the original domain
printBCSComp = true     #prints initial condition comparison and training loss plot

tStepFactor = 2 #Used to tune the time scale, if =tMeshNum/tmax the time step is the physical unit time
FPS = 5     #GIF frame per second

if extrapolate
    timeFactor  = 2 #used to extrapolate the prediction outside the domain
    xAxisFactor = 1.25 #IF IsZeroCenter THE RESULTING DOMAIN WILL BE (xAxisFactor * yAxisFactor times)^2 TIMES LARGER !!!
    yAxisFactor = 1.25
else
    timeFactor  = 1 #used to extrapolate the prediction outside the domain
    xAxisFactor = 1 #IF IsZeroCenter THE RESULTING DOMAIN WILL BE (xAxisFactor * yAxisFactor times)^2 TIMES LARGER !!!
    yAxisFactor = 1
end

if domainShape == shape[1]
    xs = 0.0 : dx : xwidth*xAxisFactor
    ys = 0.0 : dy : ywidth*yAxisFactor
elseif domainShape == shape[2]
    xs = -xwidth*0.5*xAxisFactor : dx : xwidth*0.5*xAxisFactor
    ys = -ywidth*0.5*yAxisFactor : dy : ywidth*0.5*yAxisFactor
end
ts = 1 : dt*tStepFactor : tmax*timeFactor

u_predict = [reshape([first(phi([t,x,y],pars)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]  #matrix of model's prediction
outfile = "u_predict.txt"
writedlm(outfile, u_predict)

maxlim = maximum(maximum(u_predict[t]) for t = 1:length(ts))
minlim = minimum(minimum(u_predict[t]) for t = 1:length(ts))


result = @animate for time = 1:length(ts)   #Animation of the level set function
    Plots.plot(xs, ys, u_predict[time],st=:surface,camera=(30,30), zlim=(minlim,maxlim), clim=(minlim,maxlim),
                title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n t = ",
                round((time - 1)/tMeshNum*tStepFactor*tmax, digits = 3)))
end
gif(result, "isom_creek_surface_forensic.gif", fps = FPS)

result_level = @animate for time = 1:length(ts)     #Animation of the level set contour at z=0
    Plots.contour(xs, ys, u_predict[time::Int], levels = [0], title = string("Fireline \\n t = ",
    round((time - 1)/tMeshNum*tStepFactor*tmax, digits = 3)), legend = false, size = (600,600))
end
gif(result_level, "isom_creek_contour_forensic.gif", fps = FPS)


if printBCSComp
    zbcs(x,y) = (((xScale*(x-x0[1]))^2)*xSpread[1] + ((yScale*(y-y0[1]))^2)*ySpread[1])^0.5 - amplitude[1]

    z_s = reshape([zbcs(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    target = reshape(z_s, (length(xs),length(ys)))
    diff = (u_predict[end] - target).^2
    MSE = sum(diff)/(length(xs)*length(ys))     #Mean square difference

    bcsPlot = Plots.plot(xs,ys,z_s, st=:surface,  title = "Initial Condition")    #camera=(30,30)
    bcsPredict = Plots.plot(xs, ys, u_predict[end],st=:surface, zlim=(minlim,maxlim), clim=(minlim,maxlim),
        title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n t = ",0))    #initial condition prediction

    bcsDiff = Plots.plot(xs,ys,diff, st=:surface,  title = string("MSE = ", MSE))   #difference between prediction and target

    bcsFirelinePredict = Plots.contour(xs, ys, u_predict[end], levels = [0], title = string(" Fireline \\n t = ", 0)) #initial fireline prediction

    bcsFireline = Plots.contour(xs, ys, z_s, levels = [0], title = "BCS fireline ignition")     #target initial fireline

    trainingPlot = Plots.plot(1:(maxIters + 1), losses, yaxis=:log, title = string("Training time = 270 s",
        "\\n Iterations: ", maxIters, "   NN: 3>16>1"), ylabel = "log(loss)", legend = false) #loss plot

    bcsComparisonPlots = Plots.plot(bcsPlot, bcsPredict, bcsDiff, bcsFireline,bcsFirelinePredict, trainingPlot, size = (1500,600))
    Plots.savefig("isom_creek_bcs_comparison.pdf")
    bcsComparisonPlots
end


##IMPORT THE WRF OUTPUT

tensor = readdlm("/WRF_tensor_new/tensor_isom.txt")
tensor = reshape(tensor, (1440,1440,73))
tensor = permutedims(tensor, [2,1,3])

#Plots.contour(tensor[:,:,30], levels = 0.1:0.5)

# axis_scale_to_mesh_cell_ID()
xs =  0 : dx*144 : 1440
ys =  0 : dx*144 : 1440

#=
gif_tensor = @animate for time = 1:31
    Plots.plot(tensor[:,:,time], legend = false, levels = [0.01], size = (600,600))
end
gif(gif_tensor, "WRF_out_one_fire.gif", fps = FPS)
=#

                                                                
##PLOTS FOR COMPARISON BETWEEN WRF AND PINNs                                                             

for i = 3:20
    timel = i
    tensor_p = Plots.contour(tensor[:,:,Int(floor((8 + timel)))], levels = [0.01], tick = true, grid = true, size = (400,400), colorbar=false, color="red", label=["WRF output"],
            xlabel = "Space domain [m]", legend=true, title=string("Isom Creek Fire at t = ", ((8+timel)*15), " min"))
    pred_p   = Plots.contour!(xs, ys, u_predict[Int(floor(timel))], levels = [0], tick = true, grid = true, size = (400,400), color="blue", label=["PINNs output"], legend=true)
    Plots.savefig(string("/Visualization/Levelset isom creek/", i ,".png"))
end


#h_tensor = heatmap(tensor_p)
#h_pred = heatmap(u_predict[1],levels = [0.6])


##SAVE PARAMETERS TO REPRODUCE THE RESULTS WITHOUT TRAINING AGAIN
param = initθ

outfile = "/Julia_implementation/LevelSetEq/Isom Creek/params_level_set_Isom_Creek.txt"
open(outfile, "w") do f
  for i in param
    println(f, i)
  end
end
