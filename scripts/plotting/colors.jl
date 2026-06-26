using Colors
import Plots

const beliefscolors = Dict{Symbol, RGB}(
    :red => colorant"#9C3D3D",
    :coral => colorant"#D08067",
    :sand => colorant"#EEE3C5",
    :sage => colorant"#9BBE84",
    :green => colorant"#3C7D5E",
    :teal => colorant"#4E9D8A",
    :olive => colorant"#8F9F63",
    :brown => colorant"#A45E48",
    :dark => colorant"#2C3A33",
    :light => colorant"#F8F7F2",
    :text => colorant"#252525",
    :muted => colorant"#7A827C"
);

const beliefsgradientcolors = [
    beliefscolors[:red],
    beliefscolors[:coral],
    beliefscolors[:sand],
    beliefscolors[:sage],
    beliefscolors[:green],
]

const beliefsincreasingpalettecolors = [
    beliefscolors[:sand],
    beliefscolors[:sage],
    beliefscolors[:green],
]

const beliefsgradient = Plots.cgrad(beliefsgradientcolors)
const beliefsincreasingpalette = Plots.cgrad(beliefsincreasingpalettecolors)

function beliefspalette(n)
    return Plots.palette(beliefsgradient, n)
end
