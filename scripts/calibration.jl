using Revise

using DotEnv, UnPack
using CSV, JLD2
using LinearAlgebra
using DataFrames, TidierData
using Printf
using Plots, LaTeXStrings

include("../src/utils.jl")

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

function makedf(scenarios)
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

    dfnp = dfs[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs)

    return dfnp, dfs
end

function calibrate(scenarios; emissionsfactor = 1e-3, capacityfactor = 1e-3, sccfactor = 1e-3, renewables = ("Hydro", "Nuclear", "Solar", "Storage Capacity", "Wind"), labels = ("β (t)", "α₁ (aₜ)", "α₂ (φₜ)", "α₃ (φₜ×aₜ)"), mclabels = ("κ (φₜ)", "ν (φₜ²)"))
    dfnp, dfs = makedf(scenarios)

    aₜvec = Float64[]
    aₜ₊₁vec = Float64[]

    φvec = Float64[]
    tvec = Float64[]

    pvec = Float64[]
    Evec = Float64[]

    for (_, df) in dfs
        E = df[:, "Emissions|Kyoto Gases"] * emissionsfactor
        Eⁿᵖ = interpolate(dfnp[:, "Emissions|Kyoto Gases"] * emissionsfactor, df.Year, dfnp.Year)
        abated = @. 1 - E / Eⁿᵖ

        excessedinstalledcapacity = zeros(size(df, 1))
        for renewable in renewables
            variable = "Capacity Additions|Electricity|$renewable"
            installedcapacity = df[:, variable] .* capacityfactor
            installedcapacityⁿᵖ = interpolate(dfnp[:, variable] .* capacityfactor, df.Year, dfnp.Year)

            @. excessedinstalledcapacity +=  installedcapacity - installedcapacityⁿᵖ
        end

        scc = df[:, "Price|Carbon"] * sccfactor

        t = 1:(length(abated) - 1)
        yearlytime = range(df.Year[1], df.Year[end]; step = 5.)
        
        φ = interpolate(excessedinstalledcapacity, yearlytime, df.Year)
        a = interpolate(abated, yearlytime, df.Year)
        p = interpolate(scc, yearlytime, df.Year)
        emissions = interpolate(E, yearlytime, df.Year)

        push!(aₜvec, a[t]...)
        push!(aₜ₊₁vec, a[t .+ 1]...)
        push!(φvec, φ[t]...)
        push!(tvec, yearlytime[t]...)
        push!(Evec, emissions[t]...)
        push!(pvec, p[t]...)
    end

    X = [tvec aₜvec φvec φvec.*aₜvec]
    coeff = (X'X) \ (X'aₜ₊₁vec)
    β, α... = coeff
    n, k = size(X)
    
    residuals = (aₜ₊₁vec - X * coeff)
    σ² = (residuals'residuals) / (n - k)
    Σ = σ² * inv(X'X)
    se = sqrt.(diag(Σ))

    αmes = Measurement[]
    for (i, c) in enumerate(coeff)
        push!(αmes, Measurement(c, se[i]))
        @printf "%s = %.2e (%.2e)\n" labels[i] c se[i]
    end

    mb = @. pvec * Evec * (α[2] + α[3] * aₜvec)
    Z = [ ones(n) φvec ]

    mc = (Z'Z) \ (Z'mb)
    mcresiduals = mb - Z * mc
    mcσ² = (mcresiduals'mcresiduals) / (n - 2)
    mcΣ = mcσ² * inv(Z'Z)
    mcse = sqrt.(diag(mcΣ))

    mcmes = Measurement[]
    for (i, c) in enumerate(mc)
        push!(mcmes, Measurement(c, mcse[i]))
        @printf "%s = %.2e (%.2e)\n" mclabels[i] c mcse[i]
    end

    return αmes, mcmes
end

scenarios_eu = @chain ar6data begin
    @filter(Region == "R10EUROPE")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end;

println("\n=== EU Estimation ===")
αs_eu, mcs_eu = calibrate(scenarios_eu);

scenarios_na = @chain ar6data begin
    @filter(Region == "R10NORTH_AM")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end;

println("\n=== North America Estimation ===")
αs_na, mcs_na = calibrate(scenarios_na);

scenarios_china = @chain ar6data begin
    @filter(Region == "R10CHINA+")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end;

println("\n=== China Estimation ===")
αs_china, mcs_china = calibrate(scenarios_china);

scenarios_india = @chain ar6data begin
    @filter(Region == "R10INDIA+")
    @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
    @group_by(Model, Scenario)
end;

println("\n=== India Estimation ===");
αs_india, mcs_india = calibrate(scenarios_india);

@recipe function f(::Type{T}, m::T) where T <: AbstractArray{<:Measurement}
    if !(get(plotattributes, :seriestype, :path) in (:contour, :contourf, :contour3d, :heatmap, :surface, :wireframe, :image))
        error_sym = Symbol(plotattributes[:letter], :error)
        plotattributes[error_sym] = uncertainty.(m)
    end
    value.(m)
end

begin
    regions = ["EU", "NA", "China", "India"]
    colors = [:darkblue, :darkgreen, :darkorange, :darkred]
    nregions = length(regions)
    
    αs_all = [αs_eu, αs_na, αs_china, αs_india]
    mcs_all = [mcs_eu, mcs_na, mcs_china, mcs_india]
    
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