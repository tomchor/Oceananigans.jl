"""
    NonTraditionalBetaPlane{FT} <: AbstractRotation

A Coriolis implementation that accounts for the latitudinal variation of both
the locally vertical and the locally horizontal components of the rotation vector.
The "traditional" approximation in ocean models accounts for only the locally
vertical component of the rotation vector (see [`BetaPlane`](@ref)).

This implementation is based off of section 5 of Dellar (2011). It conserve energy,
angular momentum, and potential vorticity.

References
==========
Dellar, P. (2011). Variations on a beta-plane: Derivation of non-traditional
    beta-plane equations from Hamilton's principle on a sphere. Journal of
    Fluid Mechanics, 674, 174-195. doi:10.1017/S0022112010006464
"""
struct NonTraditionalBetaPlane{FT} <: AbstractRotation
    fz :: FT
    fy :: FT
    β  :: FT
    γ  :: FT
    R  :: FT
    Cz :: FT
end

"""
    NonTraditionalBetaPlane(FT=Float64;
        fz=nothing, fy=nothing, β=nothing, γ=nothing,
        rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth)

The user may directly specify `fz`, `fy`, `β`, `γ`, and `radius` or the three
parameters `rotation_rate`, `latitude`, and `radius` that specify the rotation rate
and radius of a planet, and the central latitude (where y = 0) at which the
non-traditional `β`-plane approximation is to be made.

By default, the `rotation_rate` and planet `radius` is assumed to be Earth's.
"""
function NonTraditionalBetaPlane(FT=Float64;
        fz=nothing, fy=nothing, β=nothing, γ=nothing,
        rotation_rate=Ω_Earth, latitude=nothing, radius=R_Earth, Cz=1)

    Ω, φ, R =  rotation_rate, latitude, radius

    use_f = !all(isnothing.((fz, fy, β, γ))) && isnothing(latitude)
    use_planet_parameters = !isnothing(latitude) && all(isnothing.((fz, fy, β, γ)))

    if !xor(use_f, use_planet_parameters)
        throw(ArgumentError("Either the keywords fz, fy, β, γ, and radius must be specified, " *
                            "*or* all of rotation_rate, latitude, and radius."))
    end

    if use_planet_parameters
        fz =  2Ω*sind(φ)
        fy =  2Ω*cosd(φ)
        β  =  2Ω*cosd(φ)/R
        γ  = -4Ω*sind(φ)/R
    end

    return NonTraditionalBetaPlane{FT}(fz, fy, β, γ, R, Cz)
end

@inline two_Ωʸ(P, y, z) = P.fy * (1 - P.Cz *  z/P.R) + P.γ * y
@inline two_Ωᶻ(P, y, z) = P.fz * (1 + P.Cz * 2z/P.R) + P.β * y

# This function is eventually interpolated to fcc to contribute to x_f_cross_U.
@inline two_Ωʸw_minus_two_Ωᶻv(i, j, k, grid, coriolis, U) =
    @inbounds (  two_Ωʸ(coriolis, grid.yC[j], grid.zC[k]) * ℑzᵃᵃᶜ(i, j, k, grid, U.w)
               - two_Ωᶻ(coriolis, grid.yC[j], grid.zC[k]) * ℑyᵃᶜᵃ(i, j, k, grid, U.v))

@inline x_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    ℑxᶠᵃᵃ(i, j, k, grid, two_Ωʸw_minus_two_Ωᶻv, coriolis, U)

@inline y_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    @inbounds  two_Ωᶻ(coriolis, grid.yF[k], grid.zC[k]) * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)

@inline z_f_cross_U(i, j, k, grid, coriolis::NonTraditionalBetaPlane, U) =
    @inbounds -two_Ωʸ(coriolis, grid.yC[j], grid.zF[k]) * ℑxzᶜᵃᶠ(i, j, k, grid, U.u)

Base.show(io::IO, β_plane::NonTraditionalBetaPlane{FT}) where FT =
    print(io, "NonTraditionalBetaPlane{$FT}: ",
          @sprintf("fz = %.2e, fy = %.2e, β = %.2e, γ = %.2e, R = %.2e, Cz=%.2e",
                   β_plane.fz, β_plane.fy, β_plane.β, β_plane.γ, β_plane.R, β_plane.Cz))
