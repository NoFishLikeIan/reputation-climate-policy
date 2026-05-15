Base.@kwdef struct Signal{T <: Real}
    sigma::T = sqrt(defaulttax0)
    epsilon::T = 1e-2
end

signaldrift(tax, signal::Signal) = signal.epsilon * tax

function signalgap(tax, committedtax, signal::Signal)
    signaldrift(committedtax, signal) - signaldrift(tax, signal)
end

function beliefloading(tax, committedtax, signal::Signal)
    signalgap(tax, committedtax, signal) / signal.sigma
end
