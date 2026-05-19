## Imports
using Revise

using XLSX, DataFrames
using Dates, TimeSeries
using Statistics
using Printf

using Plots; Plots.default(dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm, background = :transparent)

## Load data
etsdatapath = "data/emission-spot-primary-market-auction-report-2012-2025-data"
@assert ispath(etsdatapath)

combinedatetime(d, t) = (ismissing(d) || ismissing(t)) ? missing :  DateTime(Date(d), Time(t));

yearswitch = 2017
filenames = filter(filename -> occursin("xlsx", filename), readdir(etsdatapath))
dfs = DataFrame[]
for filename in filenames
    rawname, _ = splitext(filename)
    year = parse(Int, split(rawname, "-")[end-1])

    filepath = joinpath(etsdatapath, filename)

    # Data format changed
    columns = year < yearswitch ? "A:Q" : "B:AZ"
    first_row = year < yearswitch ? 3 : 6
    df = XLSX.readto(filepath, "Primary Market Auction", columns, DataFrame; first_row = first_row)

    timecol = "Auction Time" in names(df) ? "Auction Time" : "Time"
    
    if year < yearswitch
        df.DateTime = combinedatetime.(df.Date, df[:, timecol])
    else 
        df.DateTime = df.Time
    end
    
    select!(df, Not(["Date", timecol]))
    select!(df, :DateTime, Not([:DateTime]))

    push!(dfs, df)
end

## Concatenate data
allnames = union(names.(dfs)...)
for df in dfs, col in allnames
    if !(col in names(df))
        df[:, col] .= missing
    end
end

fulldf = vcat(dfs...)
eudf = fulldf[[occursin("EU", x) for x in fulldf[:, "Auction Name"]], Not("Auction Name")]
sort!(eudf, :DateTime)

pricedf = coalesce.(eudf[:, ["DateTime", "Auction Price €/tCO2"]], NaN)
priceta = TimeArray(pricedf; :timestamp => :DateTime)

## Estimating
const daysperyear = 365.25
const millisecondsperday = 1000 * 60 * 60 * 24
const millisecondsperyear = millisecondsperday * daysperyear

toyears(Δt::Day) = Dates.value(Δt) / daysperyear
toyears(Δt::Millisecond) = Dates.value(Δt) / millisecondsperyear

pricevec = dropnan(priceta)
r = percentchange(pricevec, :log)
Δt = toyears.(diff(timestamp(pricevec)))

α = only(values(sum(r)) / sum(Δt))
ε = r .- α * Δt
σ = @. ε^2 / Δt
@printf "Estimated annual σ = %.5f" mean(σ) |> values |> only