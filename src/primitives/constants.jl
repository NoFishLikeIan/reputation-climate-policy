const CtoCO2 = 44 / 12
const taxfactor = CtoCO2 * 1e9 * 1e-12 # USD / tCO2 to trillion USD / GtC

const e₀ = 37.8 / CtoCO2
const y₀ = 197.231
const defaultscc = 66 / 1000
const defaultdietzϕ = 3e-5
const τ₀ = 3 * taxfactor
