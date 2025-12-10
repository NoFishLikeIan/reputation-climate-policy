function totalsocialcosts(τ₀, θ, firm::Firm, government::Government, dimensions::NTuple{2, Int}; kwargs...)
    n, T = dimensions
    firmvaluefunction = FirmValue(ones(n, T), ones(n, T) ./ 2)
    return totalsocialcosts(τ₀, θ, firm, government, firmvaluefunction; kwargs...)
end
function totalsocialcosts(τ₀, θ::S, firm::Firm, government::Government, firmvaluefunction::FirmValue; tolerance = Error{S}(1e-6, 1e-6), a₀ = zero(S), howardkwargs...) where {S <: Real}
    n, T = size(firmvaluefunction.V)
    A = range(zero(S), one(S); length = n)
    
    howard!(firmvaluefunction, τ(T, τ₀, θ), A, firm, tolerance, tolerance; howardkwargs...)
    backwardinduction!(firmvaluefunction, τ₀, θ, A, firm)
    
    aₜ = copy(a₀)
    w = zero(S)
    
    for t in 1:(T - 1)
        ϕₜ = interpolate(firmvaluefunction.Φ, (aₜ, t), (A, 1:T))
        w += socialcost(aₜ, ϕₜ, firm, government) * government.β^(t - 1)
        aₜ = f(aₜ, ϕₜ, firm)
    end
    
    return w
end

function simulatepath(τ₀, θ, firm::Firm; n = 101, T = 150, tolerance = Error(1e-6, 1e-6), a₀ = 0.)
    valuefunction = FirmValue(ones(n, T), ones(n, T) ./ 2)
    A = range(0., 1.; length = n)
    
    howard!(valuefunction, τ(T, τ₀, θ), A, firm, tolerance, tolerance)
    backwardinduction!(valuefunction, τ₀, θ, A, firm)
    
    ϕitp = linear_interpolation((A, 1:T), valuefunction.Φ; extrapolation_bc = Line())
    
    abatement = zeros(T)
    abatement[1] = a₀
    ϕ = similar(abatement)
    
    for t in 1:(T - 1)
        aₜ = abatement[t]
        ϕ[t] = interpolate(valuefunction.Φ, (aₜ, t), (A, 1:T),)
        abatement[t + 1] = f(aₜ, ϕ[t], firm)
    end
    
    return abatement, ϕ
end
