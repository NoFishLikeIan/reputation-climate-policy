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