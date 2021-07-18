
grid = RegularRectilinearGrid(size=(2,2,2), extent=(1,1,1))

amd = AnisotropicMinimumDissipation(Cn=1)
model = IncompressibleModel(grid=grid, closure=amd)

simulation = Simulation(model, Δt=1, stop_iteration=2)
@info "Starting run"
run!(simulation)
