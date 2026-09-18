abstract type AbstractHousehold{T <: Real} end

Base.@kwdef struct Household{T} <: AbstractHousehold{T}
    ν::T = defaultν
    φᴸ::T = defaultφᴸ
    r::T = realhouseholddiscount
end

function v(n, household::Household)
    household.ν * n^(1 + household.φᴸ) / (1 + household.φᴸ)
end

function v′(n, household::Household)
    household.ν * n^household.φᴸ
end

function n(W, household::Household)
    (W / household.ν)^(1 / household.φᴸ)
end
