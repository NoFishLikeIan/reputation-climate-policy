mutable struct Error{S <: Real}
    relative::S
    absolute::S
end

struct FirmValue{S <: Real}
    V::Matrix{S}
    Φ::Matrix{S}
end

# Error utilities
function zero!(error::Error{S}) where S
    error.absolute = zero(S)
    error.relative = zero(S)
    return error
end
function typemax!(error)
    error.absolute = typemax(S)
    error.relative = typemax(S)
    return error
end

function Base.isless(error::Error{S}, tolerance::Error{S}) where S
    (error.absolute < tolerance.absolute) && (error.relative < tolerance.relative)
end

function errorupdate!(error::Error{S}, x::S, y::S) where S
    a = abs(x - y)
    if a > error.absolute error.absolute = a end
    
    r = abs(x / y) - 1
    if r > error.relative error.relative = r end
end

function Base.show(io::IO, error::Error{S}) where S
    Printf.@printf io "Error(abs=%.2e, rel=%.2e)" error.absolute error.relative
end

function terminalvalueupdate!(V, error::Error, Φ, τ, A, firm::Firm)
    @inbounds for i in eachindex(V)
        ϕ = Φ[i]
        a = A[i]
        V′ = c(a, ϕ, τ, firm) + firm.β * interpolate(V, f(a, ϕ, firm), A)
        
        errorupdate!(error, V′, V[i])
        V[i] = V′
    end
    
    return V, error
end

function steadystatevalue!(V, Φ, τ, A, firm::Firm{S}, tolerance::Error{S}; maxiters = 10_000) where S
    error = Error{S}(NaN, NaN)
    
    for iter in 1:maxiters
        zero!(error)
        terminalvalueupdate!(V, error, Φ, τ, A, firm)
        
        if error < tolerance
            return V, error, iter
        end
    end
    
    return V, error, maxiters
end

# Optimal terminal policy
function optimalterminalpolicy!(Φ, error::Error, V, τ, A, firm::Firm)
    @inbounds for i in eachindex(V)
        a = A[i]
        _, ϕ′ = gssmin(ϕ -> c(a, ϕ, τ, firm) + firm.β * interpolate(V, f(a, ϕ, firm), A), 0., 1.)
        
        errorupdate!(error, ϕ′, Φ[i])
        Φ[i] = ϕ′
    end
    
    return Φ, error
end

function howard!(V, Φ, τₜ, A, firm::Firm{S}, valuetolerance::Error{S}, policytolerance::Error{S}; maxouteriters = 10_000, innneriters = 100, verbose = false, kwargs...) where S
    policyerror = Error{S}(NaN, NaN)
    innertolerance = Error{S}(0, 0)

    for iter in 1:maxouteriters
        steadystatevalue!(V, Φ, τₜ, A, firm, innertolerance; maxiters = innneriters, kwargs...)
        zero!(policyerror)
        optimalterminalpolicy!(Φ, policyerror, V, τₜ, A, firm)

        if verbose
            @printf "Iteration %i, error: absolute %.2e, relative %.2e\n" iter policyerror.absolute policyerror.relative
        end

        if policyerror < policytolerance
            return V, Φ, policyerror, iter
        end
    end

    if verbose
        @warn @sprintf "Failed convergence in %i iterations with error: absolute %.2e, relative %.2e" maxouteriters  error.absolute error.relative
    end
    
    return V, Φ, policyerror, maxouteriters
end

function howard!(valuefunction::FirmValue, τ, A, firm, valuetolerance, policytolerance; kwargs...)
    V̄ = @view valuefunction.V[:, end]
    Φ̄ = @view valuefunction.Φ[:, end]
    
    return howard!(V̄, Φ̄, τ, A, firm, valuetolerance, policytolerance; kwargs...)
end

function backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm::Firm)
    @inbounds for i in eachindex(Vₜ)
        aᵢ = A[i]
        vₜ, ϕₜ = gssmin(ϕ -> c(aᵢ, τₜ, ϕ, firm) + firm.β * interpolate(Vₜ₊₁, f(aᵢ, ϕ, firm), A), 0., 1.)
        
        Vₜ[i] = vₜ
        Φₜ[i] = ϕₜ
    end
end

function backwardinduction!(V::AbstractMatrix, Φ::AbstractMatrix, τ₀, θ, A, firm::Firm)
    T = size(V, 2)
    
    for t in reverse(1:(T - 1))
        Vₜ = @view V[:, t]
        Vₜ₊₁ = @view V[:, t + 1]
        Φₜ = @view Φ[:, t]
        τₜ = τ(t, τ₀, θ)
        
        backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm)
    end
    
    return V, Φ
end
function backwardinduction!(valuefunction::FirmValue, τ₀, θ, A, firm::Firm)
    backwardinduction!(valuefunction.V, valuefunction.Φ, τ₀, θ, A, firm)

    return valuefunction
end
