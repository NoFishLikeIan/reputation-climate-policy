function computeoverensemble(solution, fn, timesteps)
    T = length(timesteps)

    firstsol = solution[1]
    y₁ = fn(firstsol.u[1], firstsol.prob.p, firstsol.t[1])
    TY = typeof(y₁)

    n = length(solution)
    computed = Matrix{TY}(undef, n, T)

    Threads.@threads for i in 1:n
        sol = solution[i]
        yᵢ = @view computed[i, :]

        for (j, t) in enumerate(timesteps)
            x = sol(t)
            yᵢ[j] = fn(x, sol.prob.p, t)
        end
    end

    return computed
end

function quantileoverderived(computed, qs)

    


end