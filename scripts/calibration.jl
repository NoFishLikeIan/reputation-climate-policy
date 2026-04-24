using Revise

using DotEnv, UnPack
using CSV, JLD2
using LinearAlgebra
using DataFrames, TidierData
using Printf
using Plots, LaTeXStrings

includet("../src/constants.jl")

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/calibration"; if !ispath(plotpath) mkpath(plotpath) end

ar6filepath = "data/AR6_Scenarios_Database_R10_regions_v1.1.csv"; @assert isfile(ar6filepath)
ar6data = CSV.read(ar6filepath, DataFrame);
parsefloat(year) = parse(Float64, year)

struct Measurement <: Number
    val::Float64
    err::Float64
end

value(m::Measurement) = m.val
uncertainty(m::Measurement) = m.err

const variablekeys = Dict(
    :E => "Emissions|Kyoto Gases"
);

const nopolicyscenarios = Dict(
    "IMAGE 3.0" => "EN_NoPolicy",
    "IMAGE 3.2" => "SSP2-baseline",
    "GCAM 5.3" => "R_BAU",
    "GCAM 5.2" => "NGFS1_Current policies (Hot house world, Rep)",
    "GCAM-PR 5.3" => "PR_baseline",
    "GCAM5.2_NET" => "GCAM_1.5C_low_os_noDAC",
    "C3IAM 1.0" => "SSP2_BAU",
    "C3IAM 2.0" => "SSP2 BAU",
    "MESSAGEix-GLOBIOM_1.1" => "EN_NoPolicy",
    "MESSAGEix-GLOBIOM_1.2" => "COV_NoPolicyNoCOVID",
    "MESSAGEix-GLOBIOM_GEI 1.0" => "SSP2_noint_mc_50",
    "REMIND-MAgPIE 1.7-3.0" => "PEP_NoPolicy",
    "REMIND-MAgPIE 2.0-4.1" => "Diff_No-policy_baseline",
    "REMIND-MAgPIE 2.1-4.2" => "EN_NoPolicy",
    "REMIND-MAgPIE 2.1-4.3" => "DeepElec_SSP2_Npi",
    "REMIND-Buildings 2.0" => "BEG-Base",
    "REMIND-Transport 2.1" => "Transport_NPi_Conv",
    "REMIND 2.1" => "CEMICS_NPI",
    "REMIND-H13 2.1" => "START_BAU",
    "COFFEE 1.1" => "CO_BAU",
    "WITCH 4.6" => "DISCRATE_Ref_dr4p",
    "WITCH 5.0" => "EN_NoPolicy",
    "EPPA 6" => "Ref",
    "AIM/CGE 2.2" => "EN_NoPolicy",
    "GEM-E3_V2021" => "EN_NoPolicy"
);

function makedf(scenarios)
    dfsbymodel = Dict{String, Dict{String, DataFrame}}()
    
    for (key, scenario) in pairs(scenarios)
        model = key.Model
        
        if !(model in keys(nopolicyscenarios))
            continue
        end

        df = @chain DataFrame(scenario) begin
            @select(!(:Model, :Scenario, :Region, :Unit))
            stack(Not(:Variable), variable_name="Year", value_name="Value")
            unstack(:Variable, :Value, combine=mean)
            dropmissing!()
            @mutate(Year = parsefloat(Year))
            @filter(Year ≥ 2020.)
        end

        T, _ = size(df)
        hasemissions = variablekeys[:E] ∈ names(df)

        if (T ≥ 2) && hasemissions
            if !haskey(dfsbymodel, model)
                dfsbymodel[model] = Dict{String, DataFrame}()
            end
            dfsbymodel[model][key.Scenario] = df
        end
    end

    return dfsbymodel
end

function calibrate(modelscenarios, model; emissionsfactor = 1e-3, capacityfactor = 1e-3, sccfactor = 1e-3, renewables = ("Hydro", "Nuclear", "Solar", "Storage Capacity", "Wind"), labels = ("β (t)", "α₁ (aₜ)", "α₂ (φₜ)", "α₃ (φₜ×aₜ)"), mclabels = ("κ (φₜ)", "ν (φₜ²)"))

    npscenario = nopolicyscenarios[model]
    dfnp = get(modelscenarios, npscenario, nothing)
    
    if isnothing(dfnp)
        error("No policy scenario '$npscenario' not found for model $model")
    end

    aₜvec = Float64[]
    φₜvec = Float64[]
    pvec = Float64[]
    tvec = Float64[]

    for (scenarioname, df) in scenarios
        if scenarioname == npscenario continue end
        
        ts = df.Year[1]:5:df.Year[end]

        E = interpolate(df[:, "Emissions|Kyoto Gases"] * 1e-3 / CtoCO₂, ts, df.Year)
        Eⁿᵖ = interpolate(dfnp[:, "Emissions|Kyoto Gases"] * 1e-3 / CtoCO₂, ts, dfnp.Year)
        abated = Eⁿᵖ .- E

        additionalabatement = @. abated[2:end] - abated[1:(end - 1)] * (1 - abatementdepreciation)

        scc = df[:, "Price|Carbon"] * scctotax

        push!(aₜvec, abated[1:(end - 1)]...)
        push!(φₜvec, additionalabatement...)
        push!(pvec, scc[1:(end - 1)]...)
        push!(tvec, ts[1:(end - 1)]...)
    end
end

scenarioseu = @chain ar6data begin
    @filter(Region == "R10EUROPE")
    @group_by(Model, Scenario)
end;

dfbymodel = makedf(scenarioseu)

println("\n=== EU Estimation ===")
αs_eu, mcs_eu, costs_eu, p_eu, φ_eu, a_eu = calibrate(dfbymodel["IMAGE 3.0"], "IMAGE 3.0");

scenariosna = @chain ar6data begin
    @filter(Region == "R10NORTH_AM")
    @group_by(Model, Scenario)
end;

dfbymodelna = makedf(scenariosna)

println("\n=== North America Estimation ===")
αs_na, mcs_na, costs_na, p_na, φ_na, a_na = calibrate(dfbymodelna["IMAGE 3.0"], "IMAGE 3.0");

scenarioschina = @chain ar6data begin
    @filter(Region == "R10CHINA+")
    @group_by(Model, Scenario)
end;

dfbymodelchina = makedf(scenarioschina)

println("\n=== China Estimation ===")
αs_china, mcs_china, costs_china, p_china, φ_china, a_china = calibrate(dfbymodelchina["IMAGE 3.0"], "IMAGE 3.0");

scenariosindia = @chain ar6data begin
    @filter(Region == "R10INDIA+")
    @group_by(Model, Scenario)
end;

dfbymodelindia = makedf(scenariosindia)

println("\n=== India Estimation ===");
αs_india, mcs_india, costs_india, p_india, φ_india, a_india = calibrate(dfbymodelindia["IMAGE 3.0"], "IMAGE 3.0");

@recipe function f(::Type{T}, m::T) where T <: AbstractArray{<:Measurement}
    if !(get(plotattributes, :seriestype, :path) in (:contour, :contourf, :contour3d, :heatmap, :surface, :wireframe, :image))
        error_sym = Symbol(plotattributes[:letter], :error)
        plotattributes[error_sym] = uncertainty.(m)
    end
    value.(m)
end

begin
    regions = ["EU", "NA", "China", "India"]
    colors = [:darkblue, :darkgreen, beliefscolors[:green], :darkred]
    nregions = length(regions)
    
    αs_all = [αs_eu, αs_na, αs_china, αs_india]
    mcs_all = [mcs_eu, mcs_na, mcs_china, mcs_india]
    
    # Visual inspection: price vs φ_t and a_t
    p_all = [p_eu, p_na, p_china, p_india]
    φ_all = [φ_eu, φ_na, φ_china, φ_india]
    a_all = [a_eu, a_na, a_china, a_india]
    
    costplots = []
    for (i, region) in enumerate(regions)
        p1 = scatter(φ_all[i], p_all[i], label=false, title="$region: Price vs φₜ", legend=false, color=colors[i], markersize=4, alpha=0.6)
        p2 = scatter(a_all[i], p_all[i], label=false, title="$region: Price vs aₜ", legend=false, color=colors[i], markersize=4, alpha=0.6)
        p3 = scatter(φ_all[i] .* a_all[i], p_all[i], label=false, title="$region: Price vs φₜ×aₜ", legend=false, color=colors[i], markersize=4, alpha=0.6)
        push!(costplots, p1, p2, p3)
    end
    
    costfig = plot(costplots..., layout=(nregions, 3), size=(1200, 400*nregions))
    savefig(costfig, joinpath(plotpath, "cost-relationships.png"))
    
    # Create one plot per α coefficient
    α_names = [L"$\alpha_1 \; [.] \; (a_t)$", L"$\alpha_2 \; [\mathrm{yr} / \mathrm{TW}] \; (\phi_t)$", L"$\alpha_3 \; [\mathrm{yr} / \mathrm{TW}] \; ( \phi_t \times \alpha_2 )$"]
    α_plots = []
    
    for (i, α_name) in enumerate(α_names)
        vals = [value(αs[i+1]) for αs in αs_all]  # Skip β (index 1)
        errs = [uncertainty(αs[i+1]) for αs in αs_all]
        
        p = scatter(1:nregions, vals,
                   yerror = errs,
                   xticks = (1:nregions, regions),
                   title = α_name,
                   color = colors,
                   markersize = 8,
                   legend = false,
                   framestyle = :box,
                   size = (600, 300))
        
        push!(α_plots, p)
    end
    
    αfig = plot(α_plots..., layout = (1, 3), size = (1200, 400))
    savefig(αfig, joinpath(plotpath, "alpha-coefficients.png"))
    
    # Create one plot per marginal cost coefficient
    mc_names = [L"\kappa \; [\mathrm{TW} / \mathrm{tUSD}] \; (\phi_t)", L"\nu \; [\mathrm{TW}^2 / \mathrm{tUSD}^2] \; (\phi_t^2)"]
    mc_plots = []
    
    for (i, mc_name) in enumerate(mc_names)
        vals = [value(mcs[i]) for mcs in mcs_all]
        errs = [uncertainty(mcs[i]) for mcs in mcs_all]
        
        p = scatter(1:nregions, vals,
                   yerror = errs,
                   xticks = (1:nregions, regions),
                   title = mc_name,
                   color = colors,
                   markersize = 8,
                   legend = false,
                   framestyle = :box,
                   left_margin = 10Plots.mm,
                   right_margin = 10Plots.mm,
                   size = (600, 300))
        
        push!(mc_plots, p)
    end
    
    mcfig = plot(mc_plots..., layout = (1, 2), size = (1000, 400))
    savefig(mcfig, joinpath(plotpath, "mc-coefficients.png"))
end