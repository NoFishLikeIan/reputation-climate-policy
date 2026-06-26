const CtoCO2 = 44 / 12
const taxfactor = 1e9 * 1e-12 # USD / tCO2e to trillion USD / GtCO2e

const m₀ = 2_500. # [GtCO2e] cumulative emissions since 1860
const a₀ = 5.9
const e₀ = 36 + a₀ # [GtCO2e / year] Acutal emissions in 2026 + estimated no-policy additional emissions

const y₀ = 197.231 # [trillion USD / year]
const defaultscc = 66 * taxfactor # [tUSD / GtCO2e]
const defaultdietzϕ = 3e-5 # Dietz-Venmans MAC slope ϕ
const τ₀ = 50 * taxfactor
const defaultresidualδ = 10.
const lresidual₀ = defaultresidualδ / 2 * (τ₀ * sqrt(e₀ * (e₀ - a₀)) / y₀)^2
const lretirement₀ = 5e-4 # Annual accelerated-retirement loss share at net zero
const l₀ = lretirement₀ # Backwards-compatible alias

const σ̂  = 0.38