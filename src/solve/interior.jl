function discretehamiltonian(τ, i, j, u, φgrid, mgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    Δφ = step(φgrid)
    Δm = step(mgrid)
    nφ, _ = size(u)

    φ = φgrid[i]
    m = mgrid[j]

    aᵢ = aᵇ(τ, φ, τᶜ, government, firm)
    χᵢ = χ(τ, τᶜ, signal)

    dm = e(aᵢ, firm)
    dφ = beliefdrift(χᵢ, φ)
    d²φ = beliefdiffusion(χᵢ, φ)^2 / 2

    v = government.r * w(m, τ, aᵢ, climate, government, firm)

    if dm > 0
        v += dm * (u[i, j + 1] - u[i, j]) / Δm
    end

    if d²φ > 0 && 1 < i < nφ
        v += d²φ * (u[i - 1, j] - 2u[i, j] + u[i + 1, j]) / Δφ^2
    end

    if dφ > 0 && i < nφ
        v += dφ * (u[i + 1, j] - u[i, j]) / Δφ
    elseif dφ < 0 && i > 1
        v += dφ * (u[i, j] - u[i - 1, j]) / Δφ
    end

    return v
end

function optimalinteriortax(i, j, u, φgrid, mgrid, τᶜ, signal::Signal{T}, climate::Climate{T}, government::Government{T}, firm::Firm{T}) where T
    
    maxτ = government.y₀ * firm.ν * firm.e₀

    obj = @closure τ -> discretehamiltonian(τ, i, j, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
    result = Optim.optimize(obj, 0, maxτ, brent)

    return Optim.minimizer(result)
end

function updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    nφ, nm = size(u)

    @inbounds for j in 1:(nm - 1)
        m = mgrid[j]
        τᶜⱼ = τᶜ(m)

        for i in 1:nφ
            policy[i, j] = optimalinteriortax(i, j, u, φgrid, mgrid, τᶜⱼ, signal, climate, government, firm)
        end
    end

    policy[:, nm] .= policy[:, nm - 1]

    return policy
end

function initialinteriorvalue(φgrid, mgrid, u̲grid::TU, ūgrid::TU) where {T, TU <: AbstractVector{T}}
    u = Matrix{T}(undef, length(φgrid), length(mgrid))

    @inbounds for j in eachindex(mgrid), i in eachindex(φgrid)
        φ = φgrid[i]
        u[i, j] = (1 - φ) * u̲grid[j] + φ * ūgrid[j]
    end

    return u
end

function buildinteriorsystem(policy, u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm) where {T, TU <: AbstractArray{T}}
    nφ, nm = size(u)
    n = nφ * nm

    I = Int[]; sizehint!(I, 5n)
    J = Int[]; sizehint!(J, 5n)
    V = T[]; sizehint!(V, 5n)

    rhs = similar(vec(u))
    Δφ = step(φgrid)
    Δm = step(mgrid)

    @inbounds for j in 1:nm
        for i in 1:nφ
            row = interiorindex(i, j, nφ)

            if j == nm
                pushatstencil!((I, J, V), (row, row), one(T))
                φ = φgrid[i]
                rhs[row] = (1 - φ) * u̲grid[end] + φ * ūgrid[end]
                continue
            end

            φ = φgrid[i]
            m = mgrid[j]
            τ = policy[i, j]

            τᶜⱼ = τᶜ(m)

            aᵢ = aᵇ(τ, φ, τᶜⱼ, government, firm)
            χᵢ = χ(τ, τᶜⱼ, signal)
            
            dm = e(aᵢ, firm)
            dφ = beliefdrift(χᵢ, φ)
            d²φ = beliefdiffusion(χᵢ, φ)^2 / 2

            diagonal = government.r + Δt⁻¹

            if dm > 0
                rate = dm / Δm
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i, j + 1, nφ)), -rate)
            end

            if d²φ > 0 && 1 < i < nφ
                rate = d²φ / Δφ^2
                diagonal += 2rate
                pushatstencil!((I, J, V), (row, interiorindex(i - 1, j, nφ)), -rate)
                pushatstencil!((I, J, V), (row, interiorindex(i + 1, j, nφ)), -rate)
            end

            if dφ > 0 && i < nφ
                rate = dφ / Δφ
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i + 1, j, nφ)), -rate)
            elseif dφ < 0 && i > 1
                rate = -dφ / Δφ
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i - 1, j, nφ)), -rate)
            end

            pushatstencil!((I, J, V), (row, row), diagonal)

            rhs[row] = government.r * w(m, τ, aᵢ, climate, government, firm) + Δt⁻¹ * u[i, j]
        end
    end

    return SA.sparse(I, J, V, n, n), rhs
end

function interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm)
    A, rhs = buildinteriorsystem(policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)
    nextu .= reshape(A \ rhs, size(u))

    return nextu
end

function iterateinteriorhjb!(u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm, Δt⁻¹, iterations; verbose = 0) where {T, TU <: AbstractMatrix{T}}
    policy = similar(u)
    nextu = copy(u)
    errors = similar(u)
    abserror = T(Inf)
    relerror = T(Inf)

    for iter in 1:iterations
        updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
        interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)

        errors .= nextu .- u
        abserror = maximum(abs, errors)
        relerror = maximum(abs.(errors) ./ max.(abs.(u), eps(T)))

        u .= nextu

        if verbose > 1
            @printf "Interior iteration %d, Δt⁻¹ = %.2f, errors: abs = %.4e, rel = %.4e\r" iter Δt⁻¹ abserror relerror
        end
    end

    updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)

    return u, policy, (iterations, abserror, relerror)
end

function solveinteriorfixedpoint!(u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm; inneriterations = 1_000, maxstages = 8, growthfactor = 2., abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹₀ = 100.) where {T, TU <: AbstractMatrix{T}}

    policy = similar(u)
    abserror = T(Inf)
    relerror = T(Inf)
    totaliterations = 0

    Δt⁻¹ = Δt⁻¹₀

    for stage in 1:maxstages
        _, policy, (iterations, abserror, relerror) = iterateinteriorhjb!(u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal, climate, government, firm, Δt⁻¹, inneriterations; verbose)
        
        totaliterations += iterations

        if verbose > 0
            @printf "Exterior stage %d, Δt⁻¹ = %.2f, errors: abs = %.4e, rel = %.4e\n" stage Δt⁻¹ abserror relerror
        end

        if abserror < abstol && relerror < reltol
            return u, policy, (totaliterations, abserror, relerror)
        end

        Δt⁻¹ *= growthfactor
    end

    if verbose > 0
        @warn @sprintf "Exterior convergence failed after %d stages and %d iterations with errors: abs = %.4e (%.4e), rel = %.4e (%.4e)" maxstages totaliterations abserror abstol relerror reltol
    end

    return u, policy, (totaliterations, abserror, relerror)
end
