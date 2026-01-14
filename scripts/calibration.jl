using Revise

using DotEnv, UnPack
using CSV, JLD2
using DataFrames, TidierData
using Printf
using Plots, LaTeXStrings

include("../src/utils.jl")

Plots.default(linewidth = 2, dpi = 180, label = false)
plotpath = "figures/calibration"; if !ispath(plotpath) mkpath(plotpath) end

ar6filepath = "data/AR6_Scenarios_Database_R10_regions_v1.1.csv"; @assert isfile(ar6filepath)
ar6data = CSV.read(ar6filepath, DataFrame);

scenarios = @chain ar6data begin
    @filter(Region == "R10EUROPE")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end

begin
    parsefloat(year) = parse(Float64, year)

    dfs = Dict{String, DataFrame}()
    for (key, scenario) in pairs(scenarios)
        df = @chain DataFrame(scenario) begin
            @select(!(:Model, :Scenario, :Region, :Unit))
            stack(Not(:Variable), variable_name="Year", value_name="Value")
            unstack(:Variable, :Value, combine=mean)
            dropmissing!()
            @mutate(Year = parsefloat(Year))
            @filter(Year ≥ 2020.)
        end

        dfs[key.Scenario] = df
    end

    npscenario = "EN_NoPolicy"
    dfnp = dfs[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs)
end

renewables = [ "Hydro", "Nuclear", "Solar", "Storage Capacity", "Wind" ]
begin
    αs_eu = Float64[]
    κs_eu = Float64[]
    νs_eu = Float64[]
    for (k, (scenario, df)) in enumerate(dfs)
        emissionsfactor = 1e-3
        E = df[:, "Emissions|Kyoto Gases"] * emissionsfactor
        Eⁿᵖ = interpolate(dfnp[:, "Emissions|Kyoto Gases"] * emissionsfactor, df.Year, dfnp.Year)
        abated = @. 1 - E / Eⁿᵖ

        abidx = eachindex(abated)[@. (0 < abated < 1)]
        abated = abated[abidx]
        
        capacityfactor = 1e-3

        installedcapacity = zeros(size(df, 1))
        installedcapacityⁿᵖ = zeros(size(df, 1))

        for renewable in renewables
            variable = "Capacity Additions|Electricity|$renewable"
            installedcapacity .+= df[:, variable] .* capacityfactor
            installedcapacityⁿᵖ .+= interpolate(dfnp[:, variable] .* capacityfactor, df.Year, dfnp.Year)
        end

        excessedinstalledcapacity = installedcapacity - installedcapacityⁿᵖ

        sccfactor = 1e-3
        scc = df[:, "Price|Carbon"] * sccfactor

        t = abidx[1:(end - 1)]
        impulseabatement = @. (abated[t + 1] - abated[t]) / (1 - abated[t])
        yearlytime = range(extrema(df.Year[t])..., step = 5.)

        Aₜ = interpolate(impulseabatement, yearlytime, df.Year[t])
        φₜ = interpolate(excessedinstalledcapacity[t], yearlytime, df.Year[t])
        
        α = (φₜ'φₜ) \ (φₜ'Aₜ)
        
        pₜ = interpolate(scc[t], yearlytime, df.Year[t])
        Eₜⁿᵖ = interpolate(Eⁿᵖ[t], yearlytime, df.Year[t])
        aₜ = interpolate(abated[t], yearlytime, df.Year[t])

        λₜ = @. pₜ * Eₜⁿᵖ * α * (1 - aₜ)
        x = [ones(length(φₜ)) φₜ]
        κ, ν = (x'x) \ (x'λₜ)

        push!(αs_eu, α)
        push!(κs_eu, κ)
        push!(νs_eu, ν)
    end

    println("EU estimation")
    @printf "α, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(αs_eu) std(αs_eu) median(αs_eu)
    @printf "κ, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(κs_eu) std(κs_eu) median(κs_eu)
    @printf "ν, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(νs_eu) std(νs_eu) median(νs_eu)
end

ar6filepath_world = "data/AR6_Scenarios_Database_World_v1.1.csv"; @assert isfile(ar6filepath_world)
ar6data_world = CSV.read(ar6filepath_world, DataFrame);

scenarios_world = @chain ar6data_world begin
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end

begin
    parsefloat(year) = parse(Float64, year)

    dfs_world = Dict{String, DataFrame}()
    for (k, scenario) in pairs(scenarios_world)
        df = @chain DataFrame(scenario) begin
            @select(!(:Model, :Scenario, :Region, :Unit))
            stack(Not(:Variable), variable_name="Year", value_name="Value")
            unstack(:Variable, :Value, combine=first)
            dropmissing!()
            @mutate(Year = parsefloat(Year))
            @filter(Year ≥ 2020.)
        end

        dfs_world[k.Scenario] = df
    end

    dfnp_world = dfs_world[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs_world)
end

begin
    αs_world = Float64[]
    κs_world = Float64[]
    νs_world = Float64[]
    for (k, (scenario, df)) in enumerate(dfs_world)
        emissionsfactor = 1e-3
        E = df[:, "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"] * emissionsfactor
        Eⁿᵖ = interpolate(dfnp_world[:, "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"] * emissionsfactor, df.Year, dfnp_world.Year)
        abated = @. 1 - E / Eⁿᵖ

        abidx = eachindex(abated)[@. (0 < abated < 1)]
        abated = abated[abidx]
        
        capacityfactor = 1e-3
        installedcapacity = df[:, "Capacity Additions|Electricity|Renewables (incl. Biomass)"] .* capacityfactor
        installedcapacityⁿᵖ = interpolate(dfnp_world[:, "Capacity Additions|Electricity|Renewables (incl. Biomass)"] .* capacityfactor, df.Year, dfnp_world.Year)
        excessedinstalledcapacity = installedcapacity - installedcapacityⁿᵖ

        if !("Price|Carbon" in names(df))
            continue 
        end

        sccfactor = 1e-3
        scc = df[:, "Price|Carbon"] * sccfactor

        t = abidx[1:(end - 1)]
        impulseabatement = @. (abated[t + 1] - abated[t]) / (1 - abated[t])
        yearlytime = range(extrema(df.Year[t])..., step = 5.)

        Aₜ = interpolate(impulseabatement, yearlytime, df.Year[t])
        φₜ = interpolate(excessedinstalledcapacity[t], yearlytime, df.Year[t])
        
        α = (φₜ'φₜ) \ (φₜ'Aₜ)
        
        pₜ = interpolate(scc[t], yearlytime, df.Year[t])
        Eₜⁿᵖ = interpolate(Eⁿᵖ[t], yearlytime, df.Year[t])
        aₜ = interpolate(abated[t], yearlytime, df.Year[t])

        λₜ = @. pₜ * Eₜⁿᵖ * α * (1 - aₜ)
        x = [ones(length(φₜ)) φₜ]
        κ, ν = (x'x) \ (x'λₜ)

        push!(αs_world, α)
        push!(κs_world, κ)
        push!(νs_world, ν)
    end

    @printf "α, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(αs_world) std(αs_world) median(αs_world)
    @printf "κ, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(κs_world) std(κs_world) median(κs_world)
    @printf "ν, Average ≈ %.4f (%.4f), median ≈ %.4f\n" mean(νs_world) std(νs_world) median(νs_world)
end

let
    αbins = range(min(minimum(αs_eu), minimum(αs_world)), max(maximum(αs_eu), maximum(αs_world)), length=13)
    κbins = range(min(minimum(κs_eu), minimum(κs_world)), max(maximum(κs_eu), maximum(κs_world)), length=13)
    νbins = range(min(minimum(νs_eu), minimum(νs_world)), max(maximum(νs_eu), maximum(νs_world)), length=13)
    
    p1 = histogram(αs_eu, bins=αbins, alpha=0.6, xlabel=L"\alpha \; [\mathrm{year/tW}]", legend=true, c=:darkblue, linewidth = 0., title = "Abatement efficiency")
    histogram!(p1, αs_world, bins=αbins, alpha=0.6, c=:darkorange, linewidth = 0.)
    
    p2 = histogram(κs_eu, bins=κbins, alpha=0.6, xlabel=L"\kappa \; [\mathrm{trUSD/tW}]", legend=true, c=:darkblue, title = "Price")
    histogram!(p2, κs_world, bins=κbins, alpha=0.6, c=:darkorange, linewidth = 0.)
    
    p3 = histogram(νs_eu, bins=νbins, label="EU", alpha=0.6, xlabel=L"\nu \; [\mathrm{trUSD/tW}^2]", legend=true, c=:darkblue, linewidth = 0., title = "Adjustment costs")
    histogram!(p3, νs_world, bins=νbins, label="World", alpha=0.6, c=:darkorange, linewidth = 0.)
    
    estfig = plot(p1, p2, p3, layout=(1,3), size=(1400, 400), margins = 10Plots.mm)

    savefig(estfig, joinpath(plotpath, "investment-combined.png"))

    estfig
end