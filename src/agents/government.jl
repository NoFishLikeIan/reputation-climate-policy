Base.@kwdef struct Government{T <: Real}
    r::T = realgovernmentdiscount
    δ::T = defaultδᴾ
end

function w(τ, m, a, u, household::AbstractHousehold, firm::Firm, _::Government, climate::Climate)
    wage = ω(τ, firm)
    labour = n(wage, household)
    output = y(labour, firm) * (1 - d(m, climate))

    return k(a, u, firm) + v(labour, household) - output
end
