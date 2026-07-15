abstract type AbstractState{N, T} <: StaticArraysCore.FieldVector{N, T} end

struct CommittedState{T} <: AbstractState{2, T}
    m::T
    a::T
end

struct PublicState{T} <: AbstractState{3, T}
    φ::T
    m::T
    a::T
end

struct FirmState{T} <: AbstractState{4, T}
    φ::T
    m::T
    a::T
    aᵢ::T
end


# Dynamics
function χ(τ, τᶜ, signal::Signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ) 
end

function beliefdrift(χ, φ)
    -φ^2 * (1 - φ) * χ^2
end

function beliefdiffusion(χ, φ)
    φ * (1 - φ) * χ
end