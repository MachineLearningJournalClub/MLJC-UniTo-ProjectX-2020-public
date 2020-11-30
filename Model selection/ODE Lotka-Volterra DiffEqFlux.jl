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
#= Short description of this file: LOTKA-VOLTERRA EQUATIONS.

Numerical approach using DifferentialEquations.jl and Neural ODE layer approach
using DiffEqFlux.jl based on Chris Rackauckas' paper " DiffEqFlux.jl -
A Julia Library for Neural Differential Equations"
(https://arxiv.org/abs/1902.02376 )
=#

using Flux, DiffEqFlux, DifferentialEquations, Plots

#= To solve the equations numerically, we first define a problem type by giving
the equation, the initial condition, and timespan to solve over.
=#
function lotka_volterra(du,u,p,t) #Define equations
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0] #initial condition
tspan = (0.0,10.0) #timespan
p = [1.5,1.0,3.0,1.0] #parameters
prob = ODEProblem(lotka_volterra,u0,tspan,p)

#= Solve the ODE problem using 5th order Tsitouras method (Tsit5) and plot the
solution.
=#
sol = solve(prob,Tsit5())
plot(sol)

# Data generation from the above ODE solution
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector
t = 0:0.1:10.0
scatter!(t,A) #plot (t,A) over the ODE's solution

#= Build a neural network with the function as our single layer,
and define a loss function as the squared distance of the above generated data
from 1.
 =#

p = param([2.2, 1.0, 2.0, 0.4]) # Initial Parameter Vector
function predict_rd() # Our 1-layer neural network
  diffeq_rd(p,prob,Tsit5(),saveat=0.1)[1,:]
end
loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

#= 100 epoch neural network training to minimize loss function and thus get the
optimized parameters.
=#

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end
# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, [p], data, opt, cb = cb)

#= Flux trains the neural network, finding its parameters (p) that minimize the
cost function: the forward pass of the neural network includes solving an ODE.
=#
