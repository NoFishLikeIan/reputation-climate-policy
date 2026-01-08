using Revise

using DotEnv, UnPack
using CSV, JLD2
using DataFrames, TidierData
using Printf
using Plots, LaTeXStrings

include("../src/utils.jl")

Plots.default(linewidth = 2, dpi = 180, label = false)
plotpath = "figures"; if !ispath(plotpath) mkpath(plotpath) end

ar6filepath = "data/AR6_Scenarios_Database_World_v1.1.csv"; @assert isfile(ar6filepath)

ar6data = CSV.read(ar6filepath, DataFrame);

scenarios = @chain ar6data begin
    @filter(Region == "World")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    # Scenarios filters
    @filter(!occursin(r"EN_NPi.*", Scenario))
    @filter(!occursin("CEMICS_SSP2-Npi", Scenario))
    @filter(!occursin(r"Delayed Transition", Scenario))
    @filter(Scenario == "EN_NoPolicy" || 
            occursin(r"PkBudg\d+", Scenario) || # Carbon budget scenarios
            occursin(r"EN_INDCi.*_\d+f?$", Scenario) ||
            occursin(r"NGFS2_.*(2°C|Net-Zero)", Scenario) || # NGFS climate scenarios
            occursin(r"CEMICS_SSP\d-1p5C", Scenario) || # 1.5°C scenarios
            occursin(r"CEMICS_SSP\d-2C", Scenario)) # 2°C scenarios
    @group_by(Model, Scenario)
end

begin # Construct scenarios dataframes
    npscenario = "EN_NoPolicy"  # True no-policy baseline from REMIND-MAgPIE
    variablemap = Dict(
        :c => "Investment|Energy Supply|Electricity|Renewables (incl. Biomass)",
        :E => "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)",
        :ϕ => "Capacity Additions|Electricity|Renewables (incl. Biomass)",
        :p => "Price|Carbon"
    )

    parsefloat(year) = parse(Float64, year)

    dfs = Dict{String, DataFrame}()
    for (k, scenario) in pairs(scenarios)
        df = @chain DataFrame(scenario) begin
            @select(!(:Model, :Scenario, :Region, :Unit))
            stack(Not(:Variable), variable_name="Year", value_name="Value")
            unstack(:Variable, :Value)
            dropmissing!()
            @mutate(Year = parsefloat(Year))
            @filter(Year ≥ 2020.)
        end

        if all(variable in names(df) for variable in values(variablemap))
            dfs[k.Scenario] = df
        end
    end

    dfnp = dfs[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs)
    nmodels = length(dfs)
end

begin # Firms' investment parameters
    αs = Float64[]
    κs = Float64[]
    νs = Float64[]
    for (k, (scenario, df)) in enumerate(dfs)
        # aₜ [.]
        emissionsfactor = 1e-3 # MtCO₂ -> GtCO₂
        E = df[:, variablemap[:E]] * emissionsfactor
        Eⁿᵖ = interpolate(dfnp[:, variablemap[:E]] * emissionsfactor, df.Year, dfnp.Year)
        abated = @. 1 - E / Eⁿᵖ

        # Limit to 0 ≤ aₜ ≤ 1
        abidx = eachindex(abated)[@. (0 < abated < 1)]
        abated = abated[abidx]

        # c(ϕₜ) [t$ / year]
        costfactor = 1e-3 # b$ → t$
        costs = df[:, variablemap[:c]] * costfactor
        costsⁿᵖ = interpolate(dfnp[:, variablemap[:c]] * costfactor, df.Year, dfnp.Year)
        excesscosts = costs - costsⁿᵖ
        
        # ϕₜ [tW / year]
        capacityfactor = 1e-3 # gW → tW
        installedcapacity = df[:, variablemap[:ϕ]] .* capacityfactor
        installedcapacityⁿᵖ = interpolate(dfnp[:, variablemap[:ϕ]] .* capacityfactor, df.Year, dfnp.Year)
        excessedinstalledcapacity = installedcapacity - installedcapacityⁿᵖ

        # SCC [t$ / GtCO₂]
        sccfactor = 1e-3 # ($/tCO₂) → (t$ / GtCO₂)
        scc = df[:, variablemap[:p]] * sccfactor

        # Estimation of α [year / tW]
        t = abidx[1:(end - 1)]
        impulseabatement = @. (abated[t + 1] - abated[t]) / (1 - abated[t])
        yearlytime = range(extrema(df.Year[t])..., step = 5.) # Interpolate data every five years

        # FIXME: Work out whether linearly interpolating makes sense.
        Aₜ = interpolate(impulseabatement, yearlytime, df.Year[t])
        φₜ = interpolate(excessedinstalledcapacity[t], yearlytime, df.Year[t])
        
        α = (φₜ'φₜ) \ (φₜ'Aₜ)
        
        # Estimation of κ [t$ / tW] and ν [(t$ / tW)²]
        pₜ = interpolate(scc[t], yearlytime, df.Year[t])
        Eₜⁿᵖ = interpolate(Eⁿᵖ[t], yearlytime, df.Year[t])
        aₜ = interpolate(abated[t], yearlytime, df.Year[t])

        λₜ = @. pₜ * Eₜⁿᵖ * α * (1 - aₜ)
        x = [ones(length(φₜ)) φₜ]
        κ, ν = (x'x) \ (x'λₜ)

        push!(αs, α)
        push!(κs, κ)
        push!(νs, ν)
    end

    @printf "α, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(αs) std(αs) median(αs)
    @printf "κ, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(κs) std(κs) median(κs)
    @printf "ν, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(νs) std(νs) median(νs)
end

let # Plot histograms
    p1 = histogram(αs, bins=12, xlabel=L"\alpha \; [\mathrm{year/tW}]", legend=false, c = :white)
    p2 = histogram(κs, bins=12, xlabel=L"\kappa \; [\mathrm{trUSD/tW}]", legend=false, c = :white)
    p3 = histogram(νs, bins=12, xlabel=L"\nu \; [\mathrm{trUSD/tW}^2]", legend=false, c = :white)
    
    estfig = plot(p1, p2, p3, layout=(1,3), size=(1200, 400), margins = 10Plots.mm)

    savefig(estfig, joinpath(plotpath, "calibration", "estimation.png"))

    estfig
end

begin
    temperaturelabel = percentile -> @sprintf "AR6 climate diagnostics|Surface Temperature (GSAT)|FaIRv1.6.2|%.1fth Percentile" percentile;

    Δt = diff(dfnp.Year)
    ΔE = similar(Δt)
    for (i, Δtᵢ) in enumerate(Δt)
        ΔEᵢ = mean(dfnp[[i, i + 1], variablemap[:E]]) / 1000
        ΔE[i] = ΔEᵢ * Δtᵢ
    end


    TCRs = Float64[]

    percentiles = [5.0, 10.0, 16.7, 33.0, 50.0, 67.0, 83.3, 90.0, 95.0]

    for percentile in percentiles
        ΔT = diff(dfnp[:, temperaturelabel(percentile)])
        TCR = (ΔE'ΔE) \ (ΔE'ΔT)

        @printf "TCR (%.1f percentile) ≈ %.4e\n" percentile TCR

        push!(TCRs, TCR)
    end
end