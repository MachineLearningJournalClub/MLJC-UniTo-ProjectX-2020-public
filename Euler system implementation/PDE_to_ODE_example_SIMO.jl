using OrdinaryDiffEq, RecursiveArrayTools, LinearAlgebra

# Define the constants for the PDE
const α₂ = 1.0
const α₃ = 1.0
const β₁ = 1.0
const β₂ = 1.0
const β₃ = 1.0
const r₁ = 1.0
const r₂ = 1.0
const D = 100.0
const γ₁ = 0.1
const γ₂ = 0.1
const γ₃ = 0.1
const N = 100
const X = reshape([i for i in 1:100 for j in 1:100],N,N)
const Y = reshape([j for i in 1:100 for j in 1:100],N,N)
const α₁ = 1.0.*(X.>=80)

const Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
const My = copy(Mx)
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0

# Define the initial condition as normal arrays
u0 = zeros(N,N,3)

const MyA = zeros(N,N);
const AMx = zeros(N,N);
const DA = zeros(N,N)

MyA

# Define the discretized PDE as an ODE function
function f(du,u,p,t)
   A = @view  u[:,:,1]
   B = @view  u[:,:,2]
   C = @view  u[:,:,3]
  dA = @view du[:,:,1]
  dB = @view du[:,:,2]
  dC = @view du[:,:,3]
  mul!(MyA,My,A)
  mul!(AMx,A,Mx)
  @. DA = D*(MyA + AMx)
  @. dA = DA + α₁ - β₁*A - r₁*A*B + r₂*C
  @. dB = α₂ - β₂*B - r₁*A*B + r₂*C
  @. dC = α₃ - β₃*C + r₁*A*B - r₂*C
end

# Solve the ODE
prob = ODEProblem(f,u0,(0.0,100.0))
sol = solve(prob,ROCK2(),progress=true,save_everystep=false,save_start=false)

using Plots; pyplot()
p1 = surface(X,Y,sol[end][:,:,1],title = "[A]")
p2 = surface(X,Y,sol[end][:,:,2],title = "[B]")
p3 = surface(X,Y, sol[end][:,:,3],title = "[C]")
plot(p1,p2,p3,layout=grid(3,1))
