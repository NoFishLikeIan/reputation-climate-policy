import ArgParse

function parameterargumentsettings()
    settings = ArgParse.ArgParseSettings(description = "Model parameter overrides.")

    ArgParse.@add_arg_table! settings begin
        "--e0", "--e₀"
            arg_type = Float64
            dest_name = "e₀"
            default = nothing
            help = "Baseline emissions."
        "--nu", "--ν"
            arg_type = Float64
            dest_name = "ν"
            default = nothing
            help = "Marginal abatement cost slope."
        "--omega", "--ω"
            arg_type = Float64
            dest_name = "ω"
            default = nothing
            help = "Free abatement share."
        "--l0", "--l₀"
            arg_type = Float64
            dest_name = "l₀"
            default = nothing
            help = "Benchmark transition loss."
        "--a0", "--a₀"
            arg_type = Float64
            dest_name = "a₀"
            default = nothing
            help = "Benchmark abatement."
        "--y0", "--y₀"
            arg_type = Float64
            dest_name = "y₀"
            default = nothing
            help = "Output."
        "--r"
            arg_type = Float64
            default = nothing
            help = "Discount rate."
        "--epsilon", "--eps", "--ϵ", "--ε"
            arg_type = Float64
            dest_name = "ϵ"
            default = nothing
            help = "Signal drift sensitivity."
        "--sigma", "--σ"
            arg_type = Float64
            dest_name = "σ"
            default = nothing
            help = "Signal volatility."
        "--gamma", "--γ"
            arg_type = Float64
            dest_name = "γ"
            default = nothing
            help = "Damage coefficient."
        "--zeta", "--ζ"
            arg_type = Float64
            dest_name = "ζ"
            default = nothing
            help = "TCRE."
    end

    return settings
end

function parseparameterarguments(args = ARGS)
    ArgParse.parse_args(args, parameterargumentsettings(); as_symbols = true)
end

function parameterkwargs(parsed, fields)
    kwargs = Dict{Symbol,Float64}()

    for field in fields
        value = get(parsed, field, nothing)
        value === nothing || (kwargs[field] = value)
    end

    return kwargs
end

function initmodels(args = ARGS)
    parsed = parseparameterarguments(args)

    firm = Firm(; parameterkwargs(parsed, (:e₀, :ν, :ω, :l₀, :a₀))...)
    government = Government(; parameterkwargs(parsed, (:y₀, :r))...)
    signal = Signal(; parameterkwargs(parsed, (:ϵ, :σ))...)
    climate = Climate(; parameterkwargs(parsed, (:γ, :ζ))...)

    return firm, government, signal, climate
end
