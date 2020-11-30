#=
TO DO
Dare significato alla griglia   x y t
Far accettare z
includere dep temporale di wind
dipendenza spaziale del fuel
più punti di inizione (boundary) anche in tempi diversi
=#

using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot

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
x0         = [-2.0]      #Fire ingnition coordinates
y0         = [-2.0]
xSpread    = [1.0,1.0]      #Fire shape factors
ySpread    = [1.0,1.0]
tIgnition  = [0.0,0.0]      #Fire's time of ignition
amplitude  = [(max(0.5,10*dx))*sqrt(xScale*yScale),(max(0.5,10*dx))*sqrt(xScale*yScale)]      #Fire's initial spread (radius if circle)

domainShape = shape[2]

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

#Operators
#=

fire_mountain_height=100.,     ! (m) ideal mountain height
 fire_mountain_start_x=900.,   ! (m) coord of start of the mountain from lower left corder (just like ignition)
 fire_mountain_start_y=1000.,   ! (m) coord of start of the mountain from lower left corder (just like ignition)
 fire_mountain_end_x=1400.,     ! (m) coord of end of the mountain from lower left corder (just like ignition)
 fire_mountain_end_y=1600.,     ! (m) coord of end of the mountain from lower left corder (just like ignition)

=#

#z = H*exp(-x^2 - y^2)
H = 0.25
Dxz = -H*x*exp(- 0.5x^2 - 0.5y^2)
Dyz = -H*y*exp(- 0.5x^2 - 0.5y^2)

Uwind = [-10.0, 0.0]  #wind vector

gn   = (Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^0.5 #gradient's norm
∇u   = [Dx(u(t,x,y,θ)), Dy(u(t,x,y,θ))]
∇z   = [Dxz,Dyz]
n    = ∇u/gn              #normal versor
normalized = ((Uwind[1]*n[1] + Uwind[2]*n[2])^2)^0.5 #(((Uwind[1]*Dx(u(t,x,y,θ))) + (Uwind[2]*Dy(u(t,x,y,θ))))^2)^0.5/(Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^0.5

## FUEL PARAMETERS

#Fuel parameters given by namelist.fire
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

FuelNumber = 3

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

tanϕ = sum(∇z.*n)
βop  = 3.348*sigma^(-0.8189)        #from Rothermel, eq (37)
U    = normalized                   #wind correction factor
w0   = wl/(1 + Mf)
ρb   = w0/δm                        #different from paper for units reasons
β    = ρb/ρP
ξ    = exp((0.792 + 0.618*sigma^0.5)*(β+0.1))/(192 + 0.25965*sigma)
ηs   = 0.174*(SE^(-0.19))           #probably a typo in the paper
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
S    = R0*(1 + ϕw + ϕS)             #fire spread rate

eq = Dt(u(t,x,y,θ)) + S*gn ~ 0      #level set equation

initialCondition = (((xScale*(x-x0[1]))^2)*xSpread[1] + ((yScale*(y-y0[1]))^2)*ySpread[1])^0.5 - amplitude[1]
#=
if length(x0) > 2
    for b = 2:length(x0)
        initialCondition = min(initialCondition, (((xScale*(x-x0[b]))^2)*xSpread[b] + ((yScale*(y-y0[b]))^2)*ySpread[b])^0.5 - amplitude[b])
    end
end
=#

#ATTEMPT with MATRIX BC

x_s     = [x for x in xs][50:157]
y_s     = [y for y in ys][100:350]
xNum = length(x_s)
yNum = length(y_s)
zz(x,y) = -12/π*atan(cot(x-1.)) + 2.
z_s     = [zz(x,y) for x in x_s for y in y_s]
z_s     = reshape(z_s,(xMeshNum,yMeshNum))

Plots.plot(x_s,y_s,z_s)

r = 1:xMeshNum

bcs = [u(tIgnition[1],x,y,θ) ~ initialCondition,
       u(0,x,y,θ) ~ z_s[x,y] for x in collect(r) for y in collect(r)]  #from literature


## NEURAL NETWORK
n = 16
maxIters = 5000

chain = FastChain(FastDense(3,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1))
q_strategy = NeuralPDE.QuadratureTraining(algorithm =CubaCuhre(),reltol=1e-8,abstol=1e-8,maxiters=100)
discretization = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain,strategy = q_strategy)

indvars = [t,x,y]

depvars = [u]

dim = length(domains)
losses = []

cb = function (p,l)
    println("Current loss is: $l")
    append!(losses, l)
    return false
end

pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
prob = discretize(pde_system, discretization)

a_1 = time_ns()

res = GalacticOptim.solve(prob, GalacticOptim.ADAM(0.05), progress = true, cb = cb, maxiters=maxIters) #allow_f_increase = false,
#initθ = res.minimizer

#discretization2 = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain, initθ; strategy = q_strategy)
#initθ == discretization2.initθ
#prob2 = NeuralPDE.discretize(pde_system,discretization2)
#res = GalacticOptim.solve(prob2, BFGS(), cb = cb, maxiters=300)
b_1 = time_ns()
print(string("Training time = ",(b_1-a_1)/10^9))

phi = discretization.phi

## VISUALIZATION

extrapolate  = false
printBCSComp = true

tStepFactor = 5 #Used to tune the time scale, if =tMeshNum/tmax the time step is the unit time
FPS = 8

if extrapolate
    timeFactor  = 2.0 #used to extrapolate the prediction outside the domain
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
ts = 0 : dt*tStepFactor : tmax*timeFactor

u_predict = [reshape([first(phi([t,x,y],res.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]

maxlim = maximum(maximum(u_predict[t]) for t = 1:length(ts))
minlim = minimum(minimum(u_predict[t]) for t = 1:length(ts))

result = @animate for time = 1:length(ts)
    Plots.plot(xs, ys, u_predict[time],st=:surface,camera=(30,30), zlim=(minlim,maxlim), clim=(minlim,maxlim),
                title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n t = ",
                round((time - 1)/tMeshNum*tStepFactor*tmax, digits = 3)))
end
gif(result, "level_set.gif", fps = FPS)

if maxlim > 0 && minlim < 0
    print("GOOD")
    result_level = @animate for time = 1:length(ts)
        Plots.contour(xs, ys, u_predict[time::Int], levels = [0], title = string("Fireline \\n t = ",
        round((time - 1)/tMeshNum*tStepFactor*tmax, digits = 3)), legend = false)
    end
    gif(result_level, "fireline.gif", fps = FPS)
else
    print("BAD")    #in this case the level set is either always negative or always positive, so the result is meaningless
end
##
if printBCSComp
    zbcs(x,y) = (((xScale*(x-x0[1]))^2)*xSpread[1] + ((yScale*(y-y0[1]))^2)*ySpread[1])^0.5 - amplitude[1]

    z_s = reshape([zbcs(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    target = reshape(z_s, (length(xs),length(ys)))
    diff = (u_predict[1] - target).^2
    MSE = sum(diff)/(length(xs)*length(ys))

    bcsPlot = Plots.plot(xs,ys,z_s, st=:surface,  title = "Initial Condition")    #camera=(30,30)

    bcsPredict = Plots.plot(xs, ys, u_predict[1],st=:surface, zlim=(minlim,maxlim), clim=(minlim,maxlim),
        title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n t = ",0))

    bcsDiff = Plots.plot(xs,ys,diff, st=:surface,  title = string("MSE = ", MSE))

    bcsFirelinePredict = Plots.contour(xs, ys, u_predict[1], levels = [0], title = string("Fireline \\n t = ", 0))

    bcsFireline = Plots.contour(xs, ys, z_s, levels = [0], title = "BCS fireline ignition")

    trainingPlot = Plots.plot(1:(maxIters + 1), losses, yaxis=:log, title = string("Training time = ",round((b_1-a_1)/10^9), " s",
        "\\n Iterations: ", maxIters, " Hidden neurons: ", n), ylabel = "log(loss)", legend = false)

    bcsComparisonPlots = Plots.plot(bcsPlot, bcsPredict, bcsDiff, bcsFireline,bcsFirelinePredict, trainingPlot, size = (1500,600))
    png(bcsComparisonPlots, "bcs_comparison.png")
    bcsComparisonPlots
end

##
##
##
