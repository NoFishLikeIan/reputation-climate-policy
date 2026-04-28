## Imports
using Revise

using XLSX, DataFrames
using Dates, TimeSeries
using Statistics, StateSpaceModels

using Plots; Plots.default(dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm)

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
dailyta = retime(priceta, Day(1), downsample = Mean())

## Estimating
dailydf = DataFrame(dailyta)
pricevec = dailydf[:, "Auction Price €/tCO2"]
model = LocalLevel(pricevec); StateSpaceModels.fit!(model)
smoothed = kalman_smoother(model)

noisetrajectory = (pricevec - get_smoothed_state(smoothed)) ./ pricevec
σ = √((365 / 7) * model.results.coef_table.coef[1])

dailydf[!, "Auction Price Noise"] = vec(noisetrajectory)
noisepriceta = TimeArray(dailydf; :timestamp => :timestamp)

## Plotting
tickyears = range(dailydf.timestamp[1], dailydf.timestamp[end]; step = Month(6))
xticklabels = [Dates.format(t, "yyyy-mm") for t in tickyears]

ymax = maximum(abs, dailydf[:, "Auction Price Noise"])

resfig = plot(dailydf.timestamp, dailydf[:, "Auction Price Noise"], label = false, ylabel = "Residual [€/tCO2]", xlabel = "Year", c = :black, xticks = (tickyears, xticklabels), xrotation = 45, ylims = (-ylims, ylims), dpi = 280, size = 500 .* (16 / 9 , 1))

savefig(resfig, "figures/calibration/ets/residuals.png")