function committedmderivative(u::TU, mgrid, i) where {T, TU <: AbstractVector{T}}
    Δm = step(mgrid)
    n = length(mgrid)
    
    return (i < n ? u[i + 1] - u[i] : u[i] - u[i - 1]) / Δm
end

function buildcommittedsystem(u::TU, mgrid::MG, climate::Climate, government::Government, firm::Firm, Δt⁻¹) where {T <: Real, MG <: AbstractRange{T}, TU <: AbstractVector{T}}

    I = Int[]
    J = Int[]
    V = T[]

    rhs = similar(u)
    Δm = step(mgrid)
    n = length(mgrid)
    
    @inbounds for (i, m) in enumerate(mgrid)
        ∂ₘu = committedmderivative(u, mgrid, i)
        τᶜ = optimalcommittedtax(∂ₘu, government, firm)
        aᶜ = a(τᶜ, government, firm)
        dm = e(aᶜ, firm)
        welfarecost = w(m, τᶜ, aᶜ, climate, government, firm)

        if i < n && dm > 0
            pushatstencil!((I, J, V), (i, i), government.r + Δt⁻¹ + dm / Δm)
            pushatstencil!((I, J, V), (i, i + 1), -dm / Δm)
            rhs[i] = government.r * welfarecost + Δt⁻¹ * u[i]
        else
            pushatstencil!((I, J, V), (i, i), government.r + Δt⁻¹)
            rhs[i] = government.r * welfarecost + dm * ∂ₘu + Δt⁻¹ * u[i]
        end
    end

    return SA.sparse(I, J, V, n, n), rhs
end

function comittedhjbstep!(nextuᶜ, uᶜ, Δt⁻¹, mgrid, climate::Climate, government::Government, firm::Firm)
    A, rhs = buildcommittedsystem(uᶜ, mgrid, climate, government, firm, Δt⁻¹)

    nextuᶜ .= A \ rhs

    return nextuᶜ
end

function solvehjb!(uᶜ::UT, mgrid, climate::Climate, government::Government, firm::Firm; maxiters = 1000, abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹ = 100.) where {T, UT <: AbstractVector{T}}

    errors = similar(uᶜ)
    nextuᶜ = copy(uᶜ)
    abserror = T(Inf)
    relerror = T(Inf)

    for i in 1:maxiters
        comittedhjbstep!(nextuᶜ, uᶜ, Δt⁻¹, mgrid, climate, government, firm)

        errors = nextuᶜ .- uᶜ
        abserror = maximum(abs, errors)
        relerror = maximum(abs.(errors) ./ max.(abs.(uᶜ), eps(T)))

        uᶜ .= nextuᶜ

        if abserror < abstol && relerror < reltol
            return uᶜ, (i, abserror, relerror)
        end

        if verbose > 0 
            @printf "Iteration %d, errors: abs = %.2e, rel = %.2e\r" i abserror relerror
        end

    end


    if verbose > 0
        @warn @sprintf "Convergence failed after %d iterations with errors: abs = %.2e (%.2e), rel = %.2e (%.2e)" maxiters abserror abstol relerror reltol
    end

    return uᶜ, (maxiters, abserror, relerror)
end

function computeglobalpolicy(u::AbstractVector, mgrid, government::Government, firm::Firm)

    policy = similar(u)

    @inbounds for i in eachindex(mgrid)
        ∂ₘu = committedmderivative(u, mgrid, i)

        policy[i] = optimalcommittedtax(∂ₘu, government, firm)
    end

    return policy
end
