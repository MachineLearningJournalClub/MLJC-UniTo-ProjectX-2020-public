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

# Compiler: Julia 1.5.3

# Short description of this file: Numerical Approach to diffusion equation

#Warning: Reset Julia Kernel (pre-compilation required) to avoid errors in
#redefinitions of constants.

import Pkg; Pkg.add("OrdinaryDiffEq"); Pkg.add("LinearAlgebra"); Pkg.add("SparseArrays"); Pkg.add("BenchmarkTools");
using OrdinaryDiffEq, LinearAlgebra, SparseArrays, BenchmarkTools, Plots, DifferentialEquations
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
const N = 128
const X = reshape([i for i in 1:N for j in 1:N],N,N)
const Y = reshape([j for i in 1:N for j in 1:N],N,N)
const α₁ = 1.0.*(X.>=4*N/5)

const Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
const My = copy(Mx)
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0

# Define the discretized PDE as an ODE function
const MyA = zeros(N,N)
const AMx = zeros(N,N)
const DA = zeros(N,N)
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

u0 = zeros(N,N,3)
prob = ODEProblem(f,u0,(0.0,10.0))
#CPU: Intel Core i5 4440 - 4 threads
@btime solve(prob, ROCK4(), reltol = 1e-8, abstol=1e-8, saveat=0.1); # 1.111 s (3452 allocations: 285.59 MiB)
@btime solve(prob, Tsit5(), reltol = 1e-8, abstol=1e-8, saveat=0.1); # 2.886 s (275 allocations: 44.27 MiB)
@btime solve(prob, DP5(), reltol = 1e-8, abstol=1e-8, saveat=0.1);   # 2.591 s (269 allocations: 43.14 MiB)
@btime solve(prob, RK4(), relto1 = 1e-8, abstol = 1e-8, saveat=0.1); # 3.471 s (268 allocations: 43.14 MiB)

@btime solve(prob, ROCK4(), reltol = 1e-7, abstol=1e-7, saveat=0.1); # 604.227 ms (2972 allocations: 195.58 MiB)
@btime solve(prob, Tsit5(), reltol = 1e-7, abstol=1e-7, saveat=0.1); # 2.871 s (275 allocations: 44.27 MiB)
@btime solve(prob, DP5(), reltol = 1e-7, abstol=1e-7, saveat=0.1);   # 2.498 s (269 allocations: 43.14 MiB)
@btime solve(prob, RK4(), relto1 = 1e-7, abstol = 1e-7, saveat=0.1); # 3.547 s (268 allocations: 43.14 MiB)

@btime solve(prob, ROCK4(), reltol = 1e-9, abstol=1e-9, saveat=0.1); # 1.289 s (4324 allocations: 449.13 MiB)
@btime solve(prob, Tsit5(), reltol = 1e-9, abstol=1e-9, saveat=0.1); # 2.282 s (275 allocations: 44.27 MiB)
@btime solve(prob, DP5(), reltol = 1e-9, abstol=1e-9, saveat=0.1);   # 2.039 s (269 allocations: 43.14 MiB)
@btime solve(prob, RK4(), relto1 = 1e-9, abstol = 1e-9, saveat=0.1); # 2.834 s (268 allocations: 43.14 MiB)

@btime solve(prob, ROCK4(), reltol = 1e-6, abstol=1e-6, saveat=0.1); # 351.243 ms (2708 allocations: 146.07 MiB)
@btime solve(prob, Tsit5(), reltol = 1e-6, abstol=1e-6, saveat=0.1); # 2.189 s (275 allocations: 44.27 MiB)
@btime solve(prob, DP5(), reltol = 1e-6, abstol=1e-6, saveat=0.1);   # 2.069 s (269 allocations: 43.14 MiB)
@btime solve(prob, RK4(), relto1 = 1e-6, abstol = 1e-6, saveat=0.1); # 2.887 s (268 allocations: 43.14 MiB)

@btime solve(prob, ROCK4(), reltol = 1e-10, abstol=1e-10, saveat=0.1); # 1.962 s (5900 allocations: 744.69 MiB)
@btime solve(prob, Tsit5(), reltol = 1e-10, abstol=1e-10, saveat=0.1); # 2.462 s (275 allocations: 44.27 MiB)
@btime solve(prob, DP5(), reltol = 1e-10, abstol=1e-10, saveat=0.1);   # 2.133 s (269 allocations: 43.14 MiB)
@btime solve(prob, RK4(), relto1 = 1e-10, abstol = 1e-10, saveat=0.1); # 2.882 s (268 allocations: 43.14 MiB)
