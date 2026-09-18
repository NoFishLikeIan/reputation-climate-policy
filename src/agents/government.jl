Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = realgovernmentdiscount
    δ::T = defaultδᴾ
end

function lᴾ(τ, government::Government)
    government.δ * τ^2 / 2
end

function lᴾ′(τ, government::Government)
    government.δ * τ
end

function householdsurplus(τ, household::Household, firm::Firm)
    labour = n(τ, household, firm)
    y(labour, firm) - v(labour, household)
end

function lᴱ(τ, household::Household, firm::Firm)
    householdsurplus(zero(τ), household, firm) - householdsurplus(τ, household, firm)
end

function lᴱ′(τ, household::Household, firm::Firm)
    -τ * e′(τ, household, firm)
end

function l(τ, household::Household, firm::Firm, government::Government)
    lᴱ(τ, household, firm) + lᴾ(τ, government)
end

function l′(τ, household::Household, firm::Firm, government::Government)
    lᴱ′(τ, household, firm) + lᴾ′(τ, government)
end

function w(τ, m, a, u, household::Household, firm::Firm, government::Government, climate::Climate)
    government.y₀ * d(m, climate) + k(a, u, firm) + l(τ, household, firm, government)
end
