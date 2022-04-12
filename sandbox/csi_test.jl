using Pkg
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters, Oceananigans.Fields
using SpecialFunctions: erf
using CUDA: has_cuda_gpu

Nz = 2^3
Ny = 20Nz

AMD = false
AMD2 = false
AMD3 = true

f_0 = 1e-4
Ly = 6_000 # m
Lz = 80 # m
sponge_frac = 1/16

u_0 = -0.2
N2_inf = 1e-5
σ_y = 800
σ_z = 80
y_0 = +Ly/2
z_0 = 0
νz = 5e-4

LES = true


#++++ Figure out architecture
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
#-----

#++++ Set GRID
Nx = Ny÷32
Lx = (Ly / Ny) * Nx
topology = (Periodic, Bounded, Bounded)

grid = RectilinearGrid(arch, size=(Nx, Ny, Nz),
                       x=(0, Lx),
                       y=(0, Ly),
                       z=(-Lz, 0), 
                       topology=topology)
@info "" grid
#-----


#++++ Calculate secondary parameters
b₀ = u_0 * f_0
ρ₀ = 1027
T_inertial = 2*π/f_0
Ro_r = - √2 * u_0 * (z_0/σ_z-1) * exp(-1/8) / (2*f_0*σ_y)
Ri_r = N2_inf * σ_z^2 * exp(1/4) / u_0^2
νh = νz * (grid.Δyᵃᶜᵃ / grid.Δzᵃᵃᶜ)^(4/3)

secondary_params = merge((LES=Int(LES), ρ_0=ρ₀, b_0=b₀,), (; Ro_r, Ri_r, T_inertial, νh))

global_attributes = secondary_params
@info "" global_attributes
#-----



# Set up Geostrophic flow
#++++++
const n2_inf = N2_inf
const Hz = grid.Lz
const Hy = grid.Ly
const sig_z = σ_z
const sig_y = σ_y
const u₀ = u_0
const y₀ = y_0
const z₀ = z_0
const f₀ = f_0
@inline fy(ψ) = exp(-ψ^2)
@inline intgaussian(ψ) = √π/2 * (erf(ψ) + 1)
@inline umask(Y, Z) = Z * fy(Y)
@inline bmask(Y, Z) = (1/sig_z) * (sig_y * intgaussian(Y))

u_g(x, y, z, t) = +u₀ * umask((y-y₀)/sig_y, ((z-z₀)/sig_z +1))
@inline background_strat(z) = n2_inf * (z+Hz)
b_g(x, y, z, t) = -f₀ * u₀ * bmask((y-y₀)/sig_y, ((z-z₀)/sig_z +1)) + background_strat(z)
#-----

# Setting BCs
#++++
U_top_bc = FluxBoundaryCondition(0)
U_bot_bc = FluxBoundaryCondition(0)
B_bc = GradientBoundaryCondition(N2_inf)

ubc = FieldBoundaryConditions(top = U_top_bc, bottom = U_bot_bc,)
vbc = FieldBoundaryConditions()
wbc = FieldBoundaryConditions()
bbc = FieldBoundaryConditions(bottom = B_bc, top = B_bc,)
#-----


# Set-up sponge layer
#++++
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
@inline mask2nd(X) = heaviside(X) * X^2
const frac = sponge_frac

@inline function north_mask(x, y, z)
    y₁ = Hy; y₀ = y₁ - Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
@inline function south_mask(x, y, z)
    y₁ = 0; y₀ = y₁ + Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
@inline full_mask(x, y, z) = north_mask(x, y, z) + south_mask(x, y, z)

const rate = 1/10minutes
full_sponge_0 = Relaxation(rate=rate, mask=full_mask, target=0)
full_sponge_u = Relaxation(rate=rate, mask=full_mask, target=u_g)
forcing = (u=full_sponge_u, v=full_sponge_0, w=full_sponge_0)
#-----



# Set up ICs and/or Background Fields
#++++
const amplitude = 1e-6
u_ic(x, y, z) = u_g(x, y, z, 0) + amplitude*randn()
v_ic(x, y, z) = + amplitude*randn()
w_ic(x, y, z) = + amplitude*randn()
b_ic(x, y, z) = b_g(x, y, z, 0) #+ 1e-8*randn()
#-----


# Define model!
#++++
if LES
    import Oceananigans.TurbulenceClosures: SmagorinskyLilly, AnisotropicMinimumDissipation
    νₘ, κₘ = 1.0e-6, 1.5e-7
    if AMD
        sufix = "AMD"
        closure = AnisotropicMinimumDissipation(ν=νₘ, κ=κₘ, Cn=nothing)
    elseif AMD2
        sufix = "AMD2"
        closure = AnisotropicMinimumDissipation(ν=νₘ, κ=κₘ, Cn=0.1)
    elseif AMD3
        sufix = "AMD3"
        closure = AnisotropicMinimumDissipation(ν=νₘ, κ=κₘ, Cn=nothing, Pr=1)
    else
        sufix = "SMA"
        closure = SmagorinskyLilly(C=0.16, ν=νₘ, κ=κₘ)
    end
else
    import Oceananigans.TurbulenceClosures: AnisotropicDiffusivity, IsotropicDiffusivity
    closure = AnisotropicDiffusivity(νh=νh, κh=νh, νz=νz, κz=νz)
end
model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            coriolis = FPlane(f=f₀),
                            tracers = (:b,),
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (b=bbc, u=ubc, v=vbc, w=wbc),
                            forcing = forcing,
                            closure = closure,
                            )
println("\n", model, "\n")
#-----


# Adding the ICs
#++++
set!(model, u=u_ic, v=v_ic, w=w_ic, b=b_ic)

v̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
w̄ = sum(model.velocities.w.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
model.velocities.v.data.parent .-= v̄
model.velocities.w.data.parent .-= w̄
#-----


# Define time-stepping
#++++
u_scale = abs(u₀)
Δt = 1/2 * min(grid.Δzᵃᵃᶜ, grid.Δyᵃᶜᵃ) / u_scale
wizard = TimeStepWizard(cfl=0.9,
                        diffusive_cfl=0.9,
                        max_change=1.02, min_change=0.2, max_Δt=Inf, min_Δt=0.1seconds)
#----


# Finally define Simulation!
#++++
stop_time = 3*T_inertial
start_time = 1e-9*time_ns()
using Oceanostics: SingleLineProgressMessenger
simulation = Simulation(model, Δt=Δt, 
                        stop_time=stop_time,
                        wall_time_limit=23.5hours,
                        stop_iteration=Inf,)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

progress=SingleLineProgressMessenger(LES=LES, initial_wall_time_seconds=start_time)
simulation.callbacks[:messenger] = Callback(progress, IterationInterval(10))

@info "" simulation
#-----


# DIAGNOSTICS
#++++
using Oceanostics.FlowDiagnostics: ErtelPotentialVorticity
u, v, w = model.velocities
b = model.tracers.b


dbdz = Field(@at (Center, Center, Face) ∂z(b))
ω_x = Field(∂y(w) - ∂z(v))
PV = Field(ErtelPotentialVorticity(model))

outputs_vid = (; u, v, w, b, dbdz, ω_x, PV)

simulation.output_writers[:vid_writer] =
    NetCDFOutputWriter(model, outputs_vid,
                       filepath = "vid.csi_$sufix.nc",
                       schedule = TimeInterval(60minutes),
                       mode = "c",
                       array_type = Array{Float32},
                       field_slicer = FieldSlicer(i=1, with_halos=false),
                       )
#-----


# Run the simulation!
#+++++
if has_cuda_gpu() run(`nvidia-smi`) end

@printf("---> Starting run!\n")
run!(simulation)

using Oceananigans.OutputWriters: write_output!
write_output!(checkpointer, model)
#-----
