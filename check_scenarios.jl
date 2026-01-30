using Pkg
Pkg.activate(".")

using CSV, DataFrames, TidierData

ar6filepath = "data/AR6_Scenarios_Database_R10_regions_v1.1.csv"
ar6data = CSV.read(ar6filepath, DataFrame)

scenarios = ["R_MAC_45_n8", "EN_NoPolicy"]
region = "R10EUROPE"

for s in scenarios
    df = @chain ar6data begin
        @filter(Scenario == s, Variable == "Emissions|Kyoto Gases", Region == region)
        @select(Model, Scenario, Region, $(Symbol("2020")), $(Symbol("2025")), $(Symbol("2030")))
    end
    
    println("\n$s:")
    println(df)
end
