using Oceananigans
using Oceananigans.Units
using Random
using Printf

grid = RegularRectilinearGrid(size=(16, 16, 64), extent=(32, 32, 64), topology=(Periodic, Periodic, Bounded))

buoyancy_flux = 1e-8 # m² s⁻³
N² = 1e-4 # s⁻²
buoyancy_gradient_bc = GradientBoundaryCondition(N²)
buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux)
buoyancy_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

mixed_layer_depth = 32 # m
initial_buoyancy(x, y, z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth

wizard = TimeStepWizard(cfl=0.5, Δt=0.1seconds, max_change=1.1, max_Δt=2minutes,
                        min_Δt=0.01seconds)
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt.Δt))

#for cn in [nothing, 1]
    cn=1
    amd = AnisotropicMinimumDissipation(Cn=cn)
    display(amd)
    model = NonhydrostaticModel(grid=grid,
                                closure=amd,
                                buoyancy=Buoyancy(model=BuoyancyTracer()),
                                tracers=(:b,),
                                )

    rng = MersenneTwister(12345);
    u₀ = 0.1 .* rand(rng, size(model.grid)...)
    set!(model, u=u₀, b=initial_buoyancy)

    simulation = Simulation(model, Δt=wizard, stop_time=10hours,
                        iteration_interval=20, progress=progress)

    using Oceananigans.AbstractOperations
    u, v, w = model.velocities
    b = model.tracers.b
    w2 = ComputedField(w^2)
    bz = ComputedField(∂z(b))

    fname = isnothing(cn) ? "avg.test_amd_original.nc" : "avg.test_amd.nc"
    simulation.output_writers[:avg] = NetCDFOutputWriter(model, (; w2, bz), 
                                                         filepath=fname, 
                                                         schedule=TimeInterval(20minutes),
                                                         mode="c",
                                                         array_type = Array{Float64}
                                                        )

    @info "Starting run"
    run!(simulation)

    display(model.velocities.u.data)
#end
