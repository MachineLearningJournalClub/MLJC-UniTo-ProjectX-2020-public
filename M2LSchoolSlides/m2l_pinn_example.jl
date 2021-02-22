using Flux , Statistics, Plots

NN_ODE = Chain(x -> [x], # Take a scalar as input and transform it into an array
           Dense(1,16,tanh),#input layer
           Dense(16,1), #output layer
           first) # Take first value, i.e. return a scalar

# initial condition u_0 = 1. + all the prior info is HERE
g(t) = 1f0 + t*NN_ODE(t)

# small increment
ϵ = sqrt(eps(Float32))
# g(t+ϵ)-g(t))/ϵ --> difference quotient, derivative definition as a limit
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.ADAM(0.01)
data = Iterators.repeated((),250)
iter = 0

cb = function() #callback function, taking a look at training
    global iter += 1
    if iter % 50 == 0
        display(loss())
    end
end

display(loss())
Flux.train!(loss, Flux.params(NN_ODE), data, opt; cb = cb)


t = 0:1f-2:1f0
plot(t,g.(t),label="NN")
plot!(t, 1f0 .+sin.(2π.*t)/2π, label = "True Solution")

t = 0:1f-2:1.5f0
plot(t,g.(t),label="NN")
plot!(t, 1f0 .+sin.(2π.*t)/2π, label = "True Solution")
