using ApproxFun
d = ChebyshevInterval()^2                   # Defines a rectangle
Δ = Laplacian(d)                            # Represent the Laplacian
f = ones(∂(d))                              # one at the boundary
u = \([Dirichlet(d); Δ+100I], [f;0.];       # Solve the PDE
                tolerance=1E-5)
surface(u)                                  # Surface plot
