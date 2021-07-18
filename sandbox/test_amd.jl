using Random

grid = RegularRectilinearGrid(size=(3,3,3), extent=(1,1,1))

for cn in [nothing, 1]
    amd = AnisotropicMinimumDissipation(Cn=cn)
    display(amd)
    model = IncompressibleModel(grid=grid, closure=amd)

    local rng = MersenneTwister(12345);
    u₀ = rand(rng, size(model.grid)...)
    set!(model, u=u₀)

    #display(model.velocities.u.data)

    simulation = Simulation(model, Δt=1, stop_iteration=2)
    @info "Starting run"
    run!(simulation)

    display(model.velocities.u.data)
end
