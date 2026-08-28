const defaultpalette = Dict{Symbol, Colors.ColorTypes.RGB}(
    :damages => Colors.colorant"#9C3D3D",
    :guide => Colors.colorant"#7A827C",
    :mac => Colors.colorant"#9BBE84",
    :committed => Colors.colorant"#2C3A33",
    :abatement => Colors.colorant"#3C7D5E",
    :emissions => Colors.colorant"#8F9F63",
)

# ColorBrewer's even-numbered red-yellow-green scheme avoids a pale midpoint.
const beliefgradient = :RdYlGn_4
