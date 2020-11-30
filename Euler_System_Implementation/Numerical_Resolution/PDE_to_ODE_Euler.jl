using ModelingToolkit
using Plots, PyPlot
using DifferentialEquations, DiffEqOperators, LinearAlgebra, OrdinaryDiffEq, RecursiveArrayTools


@parameters x y η t θ
@variables u1(..), u2(..), u3(..), u4(..), u5(..), u6(..), u7(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y
@derivatives Dη'~η
@derivatives Dxx''~x
@derivatives Dyy''~y


#functions
u  = u1(x,y,η,t,θ)
v  = u2(x,y,η,t,θ)
w  = u3(x,y,η,t,θ)
ω  = u4(x,y,η,t,θ)
θm = u5(x,y,η,t,θ)
qm = u6(x,y,η,t,θ)
z  = u7(x,y,η,t,θ)


#physical constants
g  = 9.81 #constant
γ  = 1.4 #air cp/cv=1.4
ρd = 1 #dry air density
p0 = 100000 #reference pressure 10^5 Pa
Rd = 8.31 #gas constant for dry air
ηc = 0.2 #page 8 of Advanced Research WRF M4

#simulation inputs
sum_of_qs = 0.1 #read from file
#dx =
#dy =
#dz =
Δx = 1 #dummy
Δy = 1 #dummy

#scale factors
dist = x^2 + y^2 #dummy for distance on earth
Dxd  = 2*x #Dx(dist)
Dyd  = 2*y #Dy(dist)
mx   = Δx/dist
my   = Δy/dist

#definitions
αd = 1/ρd
p  = p0*(Rd*θm/(p0*αd))^γ
c1 = 2*ηc*ηc/((1 - ηc)*(1 - ηc)*(1 - ηc))
c2 = -ηc*(4 + ηc + ηc*ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
c3 = 2*(1 + ηc + ηc*ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
c4 = -(1 + ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
B  = c1 + c2*η + c3*η*η + c4*η*η*η
ps = 10 #dummy
pt = 1 #dummy
pd = B*(ps - pt) + (η - B)*(p0 - pt) + pt
μd = (ps - pt)*(c2 + 2*c3*η + 3*c4*η*η) + (p0 - pt)*(1 - c2 - 2*c3*η - 3*c4*η*η) #[μd = Dη(pd)]
D2pd = (ps - pt)*(2*c3 + 6*c4*η) + (p0 - pt)*(-2*c3 - 6*c4*η) # (maybe find η as a function of pd)

#ω  = Dt(η)
U  = μd*u/my
V  = μd*v/mx
W  = μd*w/my
Ω  = μd*ω/my
Θm = μd*θm
Qm = μd*qm
ϕ  = g*z
α  = αd/(1 + sum_of_qs)

#geographic parametrization
Ωe = 1 #dummy #angular rotation rate of earth
ψ  = 1 #dummy #latitude on earth
re = 1000 #dummy #radius of earth
αr = 1 #dummy #local rotation angle between y-axis and the meridians
f  = 2*Ωe*sin(ψ)
e  = 2*Ωe*cos(ψ)

#external forcings
Dymx = -Δx/(dist*dist)*Dyd
Dxmy = -Δy/(dist*dist)*Dxd


# Initial and boundary conditions
bcs =       [u1(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u2(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u3(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u4(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u5(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u6(x,y,η,0,θ) ~ (x^2 + y^2 + η^2),
             u7(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)]

PhyBcs =     []


# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0),
           η ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]# Discretization

# Discretization
dx = 0.1; dy= 0.1; dη = 0.1; dt = 0.1


##### Da qua si passa da PDE a ODE seguendo http://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/
const N_ = 10

const X = reshape([i for i in 1:10 for j in 1:10 for k in 1:10], N_,N_,N_)
const Y = reshape([j for i in 1:10 for j in 1:10 for k in 1:10], N_,N_,N_)
# \eta maiuscolo
const H = reshape([k for i in 1:10 for j in 1:10 for k in 1:10], N_,N_,N_)

const Mx = Tridiagonal([1.0 for i in 1:N_-1],[-2.0 for i in 1:N_],[1.0 for i in 1:N_-1])
const My = copy(Mx)
const Mη = copy(Mx)

# 1st order differentiation tensors
MMx = zeros(N_,N_,N_)

for i in 1:N_
    MMx[i, 1:end,1:end] = Mx
end

MMy = zeros(N_,N_,N_)

for i in 1:N_
    MMy[1:end, i, 1:end] = My
end

MMη = zeros(N_,N_,N_)

for i in 1:N_
    MMη[1:end, 1:end, i] = Mη
end


# Boundary conditions on derivatives, enforcing Neumann BC
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0
Mη[1,2] = 2.0
Mη[end,end-1] = 2.0

# rifare un attimo in base a quali prodotti abbiamo
const MMxu_1 = zeros(N_,N_,N_);
const MMyu_2 = zeros(N_,N_,N_);
const MMyu_1 = zeros(N_,N_,N_);
const MMηu_1 = zeros(N_,N_,N_);
const MMηu_4 = zeros(N_,N_,N_);
const MMxu_5 = zeros(N_,N_,N_);
const MMηu_5 = zeros(N_,N_,N_);
const MMxu_7 = zeros(N_,N_,N_);
const MMxu_2 = zeros(N_,N_,N_);
const MMηu_2 = zeros(N_,N_,N_);
const MMyu_5 = zeros(N_,N_,N_);
const MMyu_7 = zeros(N_,N_,N_);
const MMxu_3 = zeros(N_,N_,N_);
const MMyu_3 = zeros(N_,N_,N_);
const MMηu_3 = zeros(N_,N_,N_);

MMxu_1

# valori iniziali
u0 = zeros(N_,N_,N_,7)

# da completare, a partire dai prodotti per definire le derivazioni
function ff(du,u,p,t)
  u_1 = @view  u[:,:,:,1]
  u_2 = @view  u[:,:,:,2]
  u_3 = @view  u[:,:,:,3]
  u_4 = @view  u[:,:,:,4]
  u_5 = @view  u[:,:,:,5]
  u_6 = @view  u[:,:,:,6]
  u_7 = @view  u[:,:,:,7]

  du_1 = @view du[:,:, :, 1]
  du_2 = @view du[:,:, :, 2]
  du_3 = @view du[:,:, :, 3]
  du_4 = @view du[:,:, :, 4]
  du_5 = @view du[:,:, :, 5]
  du_6 = @view du[:,:, :, 6]
  du_7 = @view du[:,:, :, 7]

  #definire qua derivate
  mul!(MMxu_1,MMx,u_1)
  mul!(MMyu_2,MMy,u_2)
  mul!(MMyu_1,MMy,u_1)
  mul!(MMηu_1,MMη,u_1)
  mul!(MMηu_4,MMη,u_4)
  mul!(MMxu_5,MMx,u_5)
  mul!(MMηu_5,MMη,u_5)
  mul!(MMxu_7,MMx,u_7)
  mul!(MMxu_2,MMx,u_2)
  mul!(MMηu_2,MMη,u_2)
  mul!(MMyu_5,MMy,u_5)
  mul!(MMyu_7,MMy,u_7)
  mul!(MMxu_3,MMx,u_3)
  mul!(MMyu_3,MMy,u_3)
  mul!(MMηu_3,MMη,u_3)
  mul!(MMηu_7,MMη,u_7)
  mul!(MMxu_6,MMx,u_6)
  mul!(MMyu_6,MMy,u_6)
  mul!(MMηu_6,MMη,u_6)

  #= VARIABLES MAPPING
  u  = u1(x,y,η,t,θ)
  v  = u2(x,y,η,t,θ)
  w  = u3(x,y,η,t,θ)
  ω  = u4(x,y,η,t,θ)
  θm = u5(x,y,η,t,θ)
  qm = u6(x,y,η,t,θ)
  z  = u7(x,y,η,t,θ)
  =#

  U  = μd*u_1/my
  V  = μd*u_2/mx
  W  = μd*u_3/my
  Ω  = μd*u_4/my
  Θm = μd*u_5
  Qm = μd*u_6
  ϕ  = g*u_7

  divVu_1 = μd*u_1/Δy*(u_1*Dxd + 2*dist*MMxu_1) + μd/Δx*(u_1*u_2*Dyd + dist*u_1*MMyu_2 + dist*u_2*MMyu_1) + dist/Δy*(D2pd*u_1*u_4 + μd*MMηu_1*u_4 + μd*u_1*MMηu_4)
  divVu_2 = μd/Δy*(u_1*u_2*Dxd + dist*u_1*MMxu_2 + dist*u_2*MMxu_1) + μd/Δx*(u_2*u_2*Dyd + dist*2*u_2*MMyu_2) + 1/my*(D2pd*u_3*u_2 + μd*MMηu_4*u_2 + μd*u_4*MMηu_2)
  divVu_3 = μd/Δy*(Dxd*u_1*u_3 + dist*MMxu_1*u_3 + dist*u_1*MMxu_3) + μd/Δx*(Dyd*u_2*u_3 + dist*MMyu_2*u_3 + dist*u_2*MMyu_3) + 1/my*(D2pd*u_4*u_3 + μd*MMηu_4*u_3 + μd*u_4*MMηu_3)
  divVu_5 = μd/Δy*(Dxd*u_1*u_5 + dist*MMxu_1*u_5 + dist*u_1*MMxu_5) + μd/Δx*(Dyd*u_2*u_5 + dist*MMyu_2*u_5 + dist*u_2*MMyu_5) + 1/my*(D2pd*u_4*u_5 + μd*MMηu_4*u_5 + μd*u_4*MMηu_5)
  divVu_6 = μd/Δy*(Dxd*u_1*u_6 + dist*MMxu_1*u_6 + dist*u_1*MMxu_6) + μd/Δx*(Dyd*u_2*u_6 + dist*MMyu_2*u_6 + dist*u_2*MMyu_6) + 1/my*(D2pd*u_4*u_6 + μd*MMηu_4*u_6 + μd*u_4*MMηu_6)

  innerV∇ϕ = U*g*MMxu_7 + V*g*MMyu_7 + Ω*g*MMηu_7

  FU  =  (mx/my)*(f + u_1*(my/mx)*Dymx - u_2*Dxmy)*V - (u_1/re + e*cos(αr))*W
  FV  = -(my/mx)*((f + u_1*(my/mx)*Dymx - u_2*Dxmy*U + (u_2/re - e*sin(αr))*W))
  FW  = e*(U*cos(αr) - (mx/my)*V*sin(αr) + 1/re*(u_1*U + (mx/my)*u_2*V))
  FΘm = 0 #dummy
  FQm = 0 #dummy


  @. du_1 = - my/μd*(divVu_1 + μd*α*p0*(Rd/(p0*αd))^γ*γ*u_5^(γ-1)*MMxu_5 + (α/αd)*p0*(Rd/(p0*αd))^γ*γ*u_5^(γ-1)*MMηu_5*g*MMxu_7 - FU)
  @. du_2 = - mx/μd*(divVu_2 + μd*α*p0*(Rd/(p0*αd))^γ*γ*u_5^(γ-1)*MMyu_5 + (α/αd)*p0*(Rd/(p0*αd))^γ*γ*u_5^(γ-1)*MMηu_5*g*MMyu_7 - FV)
  @. du_3 = - my/μd*(divVu_3 - g*((α/αd)*p0*(Rd/(p0*αd))^γ*γ*u_5^(γ-1)*MMηu_5 - μd) - FW)

  @. du_5 = - 1/μd*(divVu_5 - FΘm)    #coupling with SFIRE
  @. divV = 0              # ?????
  @. du_7 = - 1/g*(1.0/μd)*(innerV∇ϕ - g*W)
  @. du_6 = - 1/μd*(divVu_6 - FQm)
end


prob = ODEProblem(ff,u0,(0.0,10.0))
sol = solve(prob,ROCK2(),progress=true,save_everystep=false,save_start=false)
#=
pde_system = PDESystem(eqs,bcs,domains,[x,y,η,t],[u1,u2,u3,u4,u5,u6,u7])

OrdinaryDiffEq.solve(pde_system)
#prob = OrdinaryDiffEq.PDEProblem(pde_system)


xs,ys,ηs,ts = [domain.domain.lower:dx:domain.domain.upper for (dx,domain) in zip([dx,dy,dη,dt],domains)]
#u_predict = [reshape([first(phi([x,y,η,t],res.minimizer)) for x in xs  for y in ys for η in ηs], (length(xs),length(ys),length(ηs)))  for t in ts ]
u_predict  = [reshape([phi([x,y,η,t],res.minimizer)[i] for x in xs for y in ys for η in ηs for t in ts],
              (length(xs),length(ys),length(ηs),length(ts))) for i in 1:7]

u_predict
=#
