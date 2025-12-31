using Revise

using DotEnv, UnPack
using CSV, JLD2
using DataFrames, TidierData
using Printf

include("../src/utils.jl")

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
            occursin(r"PkBudg\d+", Scenario) ||          # Carbon budget scenarios
            occursin(r"EN_INDCi.*_\d+f?$", Scenario) ||  # Carbon budget scenarios  
            occursin(r"NGFS2_.*(2°C|Net-Zero)", Scenario) ||  # NGFS climate scenarios
            occursin(r"CEMICS_SSP\d-1p5C", Scenario) ||  # 1.5°C scenarios
            occursin(r"CEMICS_SSP\d-2C", Scenario))      # 2°C scenarios
    @group_by(Model, Scenario)
end

begin # Construct scenarios dataframes
    npscenario = "EN_NoPolicy"  # True no-policy baseline from REMIND-MAgPIE

    parsefloat(year) = parse(Float64, year)

    investmentkey = "Investment|Energy Supply|Electricity|Non-fossil"

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

        if investmentkey in names(df)
            dfs[k.Scenario] = df
        end
    end
    dfnp = dfs[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs)
    nmodels = length(dfs)
end

begin # Fill in data coefficients
    emissionkey = "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"

    ts = Float64[]
    investments = Float64[]
    abatements = Float64[]
    for (k, (scenario, df)) in enumerate(dfs)
        years = df.Year

        investment = df[:, investmentkey] ./ 1000 # t$ / year
        E = df[:, emissionkey]
        
        dfnpmatch = filter(r -> r.Year in years, dfnp)
        investmentnp = dfnpmatch[:, investmentkey] ./ 1000
        
        if maximum(investment - investmentnp) ≤ 0
            continue
        end
        
        Eⁿᵖ = dfnpmatch[:, emissionkey]

        excessinvestment = investment - investmentnp
        abated = @. 1 - E / Eⁿᵖ
        ts = df[:, :Year] .- 2020.

        abidx = 0 .< abated .< 1
        abated = abated[abidx]
        impulseabatement = @. (abated[2:end] - abated[1:(end - 1)]) / (1 - abated[1:(end - 1)])

        ts = ts[abidx][1:(end - 1)]
        excessinvestment = excessinvestment[abidx][1:(end - 1)]

        yearlyspace = 0:1:maximum(ts)
        impulseyearly = [interpolate(impulseabatement, t, ts) for t in yearlyspace]
        investedyearly = [interpolate(excessinvestment, t, ts) for t in yearlyspace]

        append!(ts, t[idxs])
        append!(investments, excessinvestment[idxs])
        append!(abatements, abated[idxs])
    end
end
