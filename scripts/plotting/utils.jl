function safesavefigure(figure, figurepath)
    paths = publicationfigurepaths(figurepath)

    for path in paths
        safesaveonefigure(figure, path)
    end
end

function safesaveonefigure(figure, figurepath)
    try
        Plots.savefig(figure, figurepath)
    catch e
        if isa(e, SystemError)
            @warn e
        else
            throw(e)
        end
    end
end

function publicationfigurepaths(figurepath)
    root, ext = splitext(figurepath)

    if isempty(ext)
        return ["$figurepath.png", "$figurepath.pdf"]
    end

    pngpath = lowercase(ext) == ".png" ? figurepath : "$root.png"
    pdfpath = "$root.pdf"

    return pngpath == pdfpath ? [pngpath] : [pngpath, pdfpath]
end

function publicationdefaults!()
    Plots.default(
        fontfamily = "Computer Modern",
        linewidth = 2.25,
        dpi = 450,
        label = false,
        background_color = :white,
        background_color_legend = :white,
        foreground_color = :black,
        grid = true,
        gridalpha = 0.18,
        minorgrid = true,
        minorgridalpha = 0.08,
        framestyle = :box,
        tickfontsize = 9,
        guidefontsize = 11,
        titlefontsize = 12,
        legendfontsize = 9,
        colorbar_tickfontsize = 9,
        colorbar_titlefontsize = 11,
        margin = 5Plots.mm,
        size = (640, 440),
    )
end

function loadsolution(path)
    @assert isfile(path) "Solution file not found: $path"

    solutionfile = JLD2.jldopen(path)

    @assert haskey(solutionfile, "committed") "Missing committed group in $path"
    @assert haskey(solutionfile, "interior") "Missing interior group in $path"
    @assert haskey(solutionfile, "upper") "Missing upper-boundary group in $path"

    @unpack uᶜ, committedpolicy, mgrid = solutionfile["committed"]
    committedmgrid = mgrid

    @unpack φgrid, mgrid, u, interiorpolicy, u̲grid, ūgrid = solutionfile["interior"]
    @unpack ū, mrestrictedgrid = solutionfile["upper"]

    close(solutionfile)

    return (; committedmgrid, uᶜ, committedpolicy, φgrid, mgrid, u, interiorpolicy, u̲grid, ūgrid, ū, mrestrictedgrid)
end

function policyinterpolants(solution)
    τᶜ = Itp.linear_interp(solution.committedmgrid, solution.committedpolicy; extrap = Itp.ClampExtrap())
    τ = Itp.linear_interp((solution.φgrid, solution.mgrid), solution.interiorpolicy; extrap = Itp.ClampExtrap())
    u = Itp.linear_interp((solution.φgrid, solution.mgrid), solution.u; extrap = Itp.ClampExtrap())

    return (; τᶜ, τ, u)
end

function sigmafigurespecs(climate, government, firm, signal)
    defaultσ = Signal().σ
    specs = [(defaultσ, L"Baseline $\sigma$"), (2defaultσ, L"$2 \times$ baseline $\sigma$")]

    if !any(isapprox(signal.σ, σ; rtol = 1e-8) for (σ, _) in specs)
        push!(specs, (signal.σ, L"Current $\sigma$"))
    end

    return sort(specs; by = first)
end

function committedemissions(τᶜ, m, government, firm)
    return e(a(τᶜ(m), government, firm), firm)
end

function committedtrajectory(τᶜ, timesteps, government, firm)
    mpath = similar(collect(timesteps))
    τpath = similar(mpath)
    epath = similar(mpath)

    mpath[1] = m₀
    τpath[1] = τᶜ(mpath[1])
    epath[1] = committedemissions(τᶜ, mpath[1], government, firm)

    for i in 2:length(timesteps)
        Δt = timesteps[i] - timesteps[i - 1]
        mpath[i] = mpath[i - 1] + Δt * epath[i - 1]
        τpath[i] = τᶜ(mpath[i])
        epath[i] = committedemissions(τᶜ, mpath[i], government, firm)
    end

    return (; m = mpath, τ = τpath, e = epath)
end

function computepolicies(x, p, _)
    τ, τᶜ, government, firm, _ = p
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(x)
    aₜ = aᵇ(τₜ, φ, τᶜₜ, government, firm)
    τᵉ = τᶜₜ * φ + τₜ * (1 - φ)

    return (φ = φ, m = m, τ = τₜ, τᶜ = τᶜₜ, a = aₜ, e = e(aₜ, firm), τᵉ = τᵉ)
end

function simulatepolicies(solution, government, firm, signal; φ₀grid = [0.3, 0.6, 0.9], horizon = 80., trajectories = 500)
    itps = policyinterpolants(solution)
    parameters = (itps.τ, itps.τᶜ, government, firm, signal)
    timesteps = range(0, horizon; step = 1 / 6)

    solutions = map(φ₀grid) do φ₀
        prob = SDE.SDEProblem(F!, G!, [φ₀, m₀], (0, horizon), parameters)
        ensembleprob = SDE.EnsembleProblem(prob)
        SDE.solve(ensembleprob; trajectories)
    end

    policies = map(solution -> computeoverensemble(solution.u, computepolicies, timesteps), solutions)

    return (; timesteps, policies)
end

function medianpath(policies, key)
    return vec(Statistics.median(getindex.(policies, key), dims = 1))
end

function quantilepath(policies, key, q)
    return vec(mapslices(x -> Statistics.quantile(x, q), getindex.(policies, key); dims = 1))
end

function plotmedian!(fig, timesteps, policies, key; scale = identity, kwargs...)
    y = scale.(medianpath(policies, key))
    ylo = scale.(quantilepath(policies, key, 0.1))
    yhi = scale.(quantilepath(policies, key, 0.9))

    Plots.plot!(fig, timesteps, y; ribbon = (y .- ylo, yhi .- y), fillalpha = 0.12, kwargs...)
end
