Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 1e-2
    δ::T = 40.
end

function l(τ, government::Government)
    government.δ * τ^2 / 2
end