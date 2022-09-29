using Oceananigans
using Oceananigans.Units
using Printf

Nz = 32          # number of points in the vertical direction
Lz = 32          # (m) domain depth

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom

# Normalized height ranging from 0 to 1
h(k) = (k - 1) / Nz

# Linear near-surface generator
ζ₀(k) = 1 + (h(k) - 1) / refinement

# Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

grid = RectilinearGrid(size = (Nz, Nz, Nz),
                          x = (0, 100),
                          y = (0, 100),
                          z = z_faces)

#++++ Model and IC
closure = AnisotropicMinimumDissipation(Cn=1)
#closure = SmagorinskyLilly()
model = NonhydrostaticModel(advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = closure,
                            )

# Random noise damped at top and bottom

# Temperature initial condition: a stable density gradient with random noise superposed.
bᵢ(x, y, z) = 1e-5 * z

# Velocity initial condition: random noise scaled by the friction velocity.
amplitude = 1e-3
uᵢ(x, y, z) = amplitude * randn()

# `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, b=bᵢ,)
#----


#++++ Simulation
simulation = Simulation(model, Δt=10.0, stop_time=4hours)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim),
                                prettytime(sim),
                                prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w),
                                prettytime(sim.run_wall_time))
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))
#----


#++++ Output
b = model.tracers.b
eddy_viscosity = (; νₑ = model.diffusivity_fields.νₑ, dbdz=Field(@at (Center, Center, Center) ∂z(b)))

closure_name = string(typeof(closure).name.wrapper) # Get closure's name
simulation.output_writers[:netcdf] = NetCDFOutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                                                        filename = "amd_test_$closure_name.nc",
                                                        indices = (:, grid.Ny/2, :),
                                                        schedule = TimeInterval(1minute),
                                                        overwrite_existing = true)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                                                    filename = "amd_test_$closure_name",
                                                    indices = (:, grid.Ny/2, :),
                                                    schedule = TimeInterval(1minute),
                                                    overwrite_existing = true)
#----


#++++ Run simulation
run!(simulation)
#----


#++++ Plotting
using JLD2
using Plots

jld2writer = simulation.output_writers[:jld2]
file = jldopen(jld2writer.filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

@info "Making a neat movie from $(jld2writer.filepath)"

xC, yC, zC = nodes(jld2writer.outputs[:b])

function eigenplot(ν, b, σ, t; ν_lim=maximum(abs, ν)+1e-16, b_lim=maximum(abs, b)+1e-16)

    kwargs = (xlabel="x", ylabel="z", linewidth=0, label=nothing, color = :balance,)

    ν_title(t) = t == nothing ? @sprintf("vorticity") : @sprintf("vorticity at t = %.2f", t)

    plot_ν = heatmap(xC, zC, clamp.(ν, -ν_lim, ν_lim)';
                      levels = range(-ν_lim, stop=ν_lim, length=20),
                       xlims = (xC[1], xC[grid.Nx]),
                       ylims = (zC[1], zC[grid.Nz]),
                       clims = (-ν_lim, ν_lim),
                       title = ν_title(t), kwargs...)

    b_title(t) = t == nothing ? @sprintf("buoyancy") : @sprintf("buoyancy at t = %.2f", t)

    plot_b = heatmap(xC, zC, clamp.(b, 0, b_lim)';
                    levels = range(-b_lim, stop=b_lim, length=20),
                     xlims = (xC[1], xC[grid.Nx]),
                     ylims = (zC[1], zC[grid.Nz]),
                     clims = (-b_lim, b_lim),
                     title = b_title(t), kwargs...)

    return plot(plot_ν, plot_b, layout=(1, 2),)
end

anim_total = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."

    t = file["timeseries/t/$iteration"]
    b_snapshot = file["timeseries/b/$iteration"][:, 1, :]
    dbdz_snapshot = file["timeseries/dbdz/$iteration"][:, 1, :]
    ν_snapshot = file["timeseries/νₑ/$iteration"][:, 1, :]

    eigenmode_plot = eigenplot(ν_snapshot, dbdz_snapshot, nothing, t; ν_lim=1e-4, b_lim=1e-5)

    plot(eigenmode_plot, size=(1200, 600))
end

videofile_name = replace(jld2writer.filepath, "jld2"=>"mp4")
mp4(anim_total, videofile_name, fps = 15) # hide

