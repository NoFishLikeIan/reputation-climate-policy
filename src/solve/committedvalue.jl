function pushat!((I, J, V), v, (i, j))
    push!(I, i)
    push!(J, j)
    push!(V, v)
end

function initscheme(u::TU, mgrid::MG, climate::Climate, government::Government, firm::Firm) where {T <: Real, MG <: AbstractRange{T}, TU <: AbstractVector{T}}

    welfarecosts = Vector{T}(undef, size(mgrid))
    scheme = (Int[], Int[], T[])
    Δm = step(mgrid)
    n = length(mgrid)
    
    @inbounds for (i, m) in enumerate(mgrid)
        j = ifelse(i < n, i + 1, i - 1)

        ∂ₘu = abs(u[j] - u[i]) / Δm

        τᶜ = optimalcommittedtax(∂ₘu, government, firm)
        s = e(a(τᶜ, firm), firm) / Δm

        pushat!(scheme, -s, (i, i))
        pushat!(scheme, s, (i, j))

        welfarecosts[i] = w(m, τᶜ, a(τᶜ, firm), climate, government, firm)
    end

    return scheme, welfarecosts
end

function updatescheme!(scheme, welfarecosts::AbstractVector, u::AbstractVector, mgrid::MG, climate::Climate, government::Government, firm::Firm) where {T <: Real, MG <: AbstractRange{T}}
    V = scheme[3]
    Δm = step(mgrid)
    n = length(mgrid)
    
    @inbounds for (i, m) in enumerate(mgrid)
        j = ifelse(i < n, i + 1, i - 1)

        ∂ₘu = abs(u[j] - u[i]) / Δm

        τᶜ = optimalcommittedtax(∂ₘu, government, firm)
        s = e(a(τᶜ, firm), firm) / Δm

        idx = (2i - 1, 2i)
        V[idx[1]] = -s
        V[idx[2]] = s
        
        welfarecosts[i] = w(m, τᶜ, a(τᶜ, firm), climate, government, firm)
    end

    return scheme, welfarecosts
end

function comittedhjbstep!(nextuᶜ, welfarecosts, scheme, uᶜ, Δt⁻¹, mgrid, climate::Climate, government::Government, firm::Firm)
    updatescheme!(scheme, welfarecosts, uᶜ, mgrid, climate, government, firm)

    n = length(mgrid)
    A = SA.sparse(scheme[1], scheme[2], scheme[3], n, n)

    nextuᶜ .= ((government.r + Δt⁻¹) * LA.I - A) \ (government.r * welfarecosts + Δt⁻¹ * uᶜ)

    return nextuᶜ
end

function solvehjb!(uᶜ::UT, mgrid, climate::Climate, government::Government, firm::Firm; maxiters = 1000, abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹ = 100.) where {T, UT <: AbstractVector{T}}

    scheme, welfarecosts = initscheme(uᶜ, mgrid, climate, government, firm)
    errors = similar(uᶜ)
    nextuᶜ = copy(uᶜ)
    abserror = T(Inf)
    relerror = T(Inf)

    for i in 1:maxiters
        comittedhjbstep!(nextuᶜ, welfarecosts, scheme, uᶜ, Δt⁻¹, mgrid, climate, government, firm)

        errors = nextuᶜ .- uᶜ
        abserror = maximum(abs, errors)
        relerror = maximum(abs, errors ./ uᶜ)

        if abserror < abstol && relerror < reltol
            return uᶜ, (i, abserror, relerror)
        end

        if verbose > 0 
            @printf "Iteration %d, errors: abs = %.2e, rel = %.2e\r" i abserror relerror
        end

        uᶜ .= nextuᶜ
    end


    if verbose > 0
        @warn @sprintf "Convergence failed after %d iterations with errors: abs = %.2e (%.2e), rel = %.2e (%.2e)" maxiters abserror abstol relerror reltol
    end

    return uᶜ, (maxiters, abserror, relerror)
end

function computeglobalpolicy(u::AbstractVector, mgrid, government::Government, firm::Firm)

    policy = similar(u)
    Δm = step(mgrid)
    n = length(mgrid)

    @inbounds for i in eachindex(mgrid)
        j = ifelse(i < n, i + 1, i - 1)

        ∂ₘu = abs(u[j] - u[i]) / Δm

        policy[i] = optimalcommittedtax(∂ₘu, government, firm)
    end

    return policy
end
