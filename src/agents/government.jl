Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 0.02
    δ::T = 40.
end

function l(τ, government::Government)
    government.δ * τ^2 / 2
end

function l′(τ, government::Government)
    government.δ * τ
end

function w(m, τ, government::Government, climate::Climate)
    government.y₀ * d(m, climate) + l(τ, government)
end
