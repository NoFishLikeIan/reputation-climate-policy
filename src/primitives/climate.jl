Base.@kwdef struct Climate{T <: Real}
    γ::T = 1e-2 # Damage coefficient
    ζ::T = 4.8e-4 # TCRE    
end

const badclimate = Climate(γ = 2e-2, ζ = 7.49e-4)
const goodclimate = Climate(γ = 5e-3, ζ = 2.4e-4)

function temperature(m, climate::Climate)
    climate.ζ * m
end

"Fraction of output `y` destroyed by cumulative warming `m`"
function d(m, climate::Climate)
    1 - exp(-(climate.γ / 2) * temperature(m, climate)^2)
end

function d′(m, climate::Climate)
    climate.γ * climate.ζ^2 * m * (1 - d(m, climate))
end
