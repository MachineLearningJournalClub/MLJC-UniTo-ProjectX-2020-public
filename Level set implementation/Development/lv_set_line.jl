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

#eq = Dt(u(t,x,y,θ)) + (Dx(u(t,x,y,θ))*Dx(u(t,x,y,θ)) + Dy(u(t,x,y,θ))*Dy(u(t,x,y,θ)))^0.5 ~ 0      #level set equation

#FireIgnition = -1.5(exp(-0.1*(x-x0)^2-0.1*(y-y0)^2))+1

#operators
#z = x^2 - y^2
Dxz = 0.1#2*x
Dyz = 0.1#-2*y

gn   = (Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^0.5 #gradient norm
∇u   = [Dx(u(t,x,y,θ)), Dy(u(t,x,y,θ))]
∇z   = [Dxz,Dyz]
n    = ∇u/gn #normal to the fire region

## FUEL PARAMETERS

#Fuel parameters given by file (convert to Imperial unit system!!!!)
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

FuelNumber = 1

#=
a     = windrf[FuelNumber]          #thoose numbers should be dependent on position
zf    =
z0    =
w     = weight[FuelNumber]
wl    = fgi[FuelNumber]
δm    = fueldepthm[FuelNumber]
sigma = savr[FuelNumber]
Mx    = fuelmce[FuelNumber]
ρP    = fueldens[FuelNumber]
ST    = st[FuelNumber]
SE    = se[FuelNumber]
h     = cmbcnst[FuelNumber]
Mf    = fuelmc_g[FuelNumber]
=#

a     = windrf[FuelNumber]          #thoose numbers should be dependent on position
zf    =
z0    =
w     = weight[FuelNumber]/60           #seconds to minutes
wl    = fgi[FuelNumber]*0.204816        #kg*m^-2 to lb*ft^-2
δm    = fueldepthm[FuelNumber]*3.28084  #m to ft
sigma = savr[FuelNumber]/3.28084        #m^-1 to ft^-1
Mx    = fuelmce[FuelNumber]   
ρP    = fueldens[FuelNumber]*0.062428   #kg*m^-3 to lb*ft^-3
ST    = st[FuelNumber]
SE    = se[FuelNumber]
h     = cmbcnst[FuelNumber]*0.000026839 #J*m^-3 to BTU*ft^-3
Mf    = fuelmc_g[FuelNumber]

## FIRE SPREAD RATE EQUATIONS

tanϕ = sum(∇z.*n)
βop  = 0.1                          #dummy
U    = 0                            #wind vector (dummy)
w0   = wl/(1 + Mf)
ρb   = w0/δm                        #different from paper for units reasons
β    = ρb/ρP
ξ    = exp((0.792 + 0.618*sigma^0.5)*(β+0.1))/(192 + 0.25965*sigma)
ηs   = 0.174*(SE^(-0.19))               #probably mistaken in the paper
ηM   = 1 - 2.59*Mf/Mx + 5.11*(Mf/Mx)^2 - 3.52*(Mf/Mx)^3
wn   = w0/(1 + ST)
Γmax = (sigma^(1.5))/(495 + 0.594*sigma^(1.5))
A    = 1/(4.77*sigma^(0.1) - 7.27)
Γ    = Γmax*(β/βop)^A*exp(A*(1 - β/βop))
ϵ    = exp(-138/sigma)
Qig  = 250*β + 1116*Mf
C    = 7.47*exp(-0.133*sigma^0.55)
Ua   = a*U                          #U is the wind vector
E    = 0.715*exp(-0.000359*sigma)
IR   = Γ*wn*h*ηM*ηs
R0   = IR*ξ/(ρb*ϵ*Qig)              #spread rate without wind
ϕw   = C*max(Ua^β, (β/βop)^E)       #wind factor
ϕS   = 5.275*β^(-0.3)*tanϕ^2        #slope factor
S    = R0*(1 + ϕw + ϕS)               #fire spread rate --- should be between 0 and 1

eq = Dt(u(t,x,y,θ)) + S*gn ~ 0      #level set equation

#FireIgnition = -1.5(exp(-0.1*(x-x0)^2-0.1*(y-y0)^2))+1

## DOMAINS AND INITIAL/BOUNDARY CONDITIONS

# Discretization
xwidth      = 10.0
ywidth      = 10.0
tmax        = 10.0
xSemiAxis   = 1.0
ySemiAxis   = 1.0
xMeshNum    = 100
yMeshNum    = 100
tMeshNum    = 20
dx  = xwidth/xMeshNum
dy  = ywidth/yMeshNum
dt  = tmax/tMeshNum

#Fire position and shape

shape     = ["zeroIsVertex","zeroIsCenter"]
x0         = [0.0]      #Fire ingnition coordinates
y0         = [0.0]
xSemiAxis  = [2.0]      #Fire shape factors
ySemiAxis  = [0.2]
tIgnition  = [0.0]      #Fire's time of ignition
amplitude  = [1.0]      #Fire's initial spread (0.5*radius if circle)

domainShape = shape[2]

if domainShape == shape[1]
    domains = [t ∈ IntervalDomain(0.0,tmax),
           x ∈ IntervalDomain(0.0,xwidth),
           y ∈ IntervalDomain(0.0,ywidth)]
elseif domainShape == shape[2]
    domains = [t ∈ IntervalDomain(0.0,tmax),
           x ∈ IntervalDomain(-xwidth*0.5,xwidth*0.5),
           y ∈ IntervalDomain(-ywidth*0.5,ywidth*0.5)]
end

#initial condition from a paper

bcs = [u(tIgnition[1],x,y,θ) ~ (xSemiAxis[1]*(x-x0[1])^2 + ySemiAxis[1]*(y-y0[1])^2)^0.5 - amplitude[1]]



#=
bcs = [u(0,x,y,θ) ~ ((x - x0)^2 + (y - y0)^2)^0.5 - initialSpread*dx,
        u(dt,x,y,θ) ~ ((x - x0)^2 + (y - y0)^2)^0.5 - initialSpread*dx,
        u(2*dt,x,y,θ) ~ ((x - x0)^2 + (y - y0)^2)^0.5 - initialSpread*dx] #Initial and boundary conditions
=#
#=
bcs = [u(0,x,y,θ) ~ ((x - x0)^2 + (y - y0)^2)^0.5 - initialSpread*dx,
        Dt(u(0,x,y,θ)) ~ -S*gn] #Initial and boundary conditions
=#

#bcs = [u(0,x,y,θ) ~ ((x - x0)^2 + (y - y0)^2)^0.5 - initialSpread*dx] #Initial and boundary conditions
#bcs = [u(0,x,y,θ) ~ 1] #Initial and boundary conditions
#bcs = [u(0,x,y,θ) ~ (1 - 2*exp(-(((x - x0)/xSemiAxis)^20 + ((y - y0)/ySemiAxis)^20)))*amplitude]

#bcs = [u(0,x,y,θ) ~ (x^2 + y^2)^0.5 - 1] # Initial and boundary conditions
#bcs = [u(0,x,y,θ) ~ (xSemiAxis*(x-x0)^2 + ySemiAxis*(y-y0)^2)^0.5 - amplitude]
       #u(0,x,y,θ)  ~ -0.0015(exp(-(x-x0)^2-(y-y0)^2))+1
       #u(dt,x,y,θ) ~ -0.0015(exp(-(x-x0)^2-(y-y0)^2))+1
       #u(2*dt,x,y,θ) ~ -0.0015(exp(-(x-x0)^2-(y-y0)^2))+1]
       #u(t,x,0,θ) ~ 100
       #u(t,x,2,θ) ~ 100
       #u(t,0,y,θ) ~ 100
       #u(t,2,y,θ) ~ 100]

#bcs = [u(0,x,y,θ) ~ (1 - 2*exp(-(((x - x0)/xSemiAxis)^20 + ((y - y0)/ySemiAxis)^20)))*amplitude]

## NEURAL NETWORK
n = 16
chain = FastChain(FastDense(3,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) 
q_strategy = NeuralPDE.QuadratureTraining(algorithm =CubaCuhre(),reltol=1e-8,abstol=1e-8,maxiters=100)
discretization = NeuralPDE.PhysicsInformedNN([dt,dx,dy],chain,strategy = q_strategy)

indvars = [t,x,y]
depvars = [u]

dim = length(domains)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
prob = discretize(pde_system, discretization)

# optimizer
# opt = GalacticOptim.ADAM(0.01)
# opt = BFGS()

a_1 = time_ns()

res = GalacticOptim.solve(prob, GalacticOptim.ADAM(0.03), progress = true, allow_f_increase = false, cb = cb, maxiters=1000) 
b_1 = time_ns()
show((b_1-a_1)/10^9)

phi = discretization.phi

ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]


## VISUALIZATION

extrapolate  = true
printBCSComp = true

tStepFactor = 10 #Used to tune the time scale, if =tMeshNum/tmax the time step is the unit time
FPS = 10

if extrapolate
    timeFactor  = 2.5 #used to extrapolate the prediction outside the domain
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
