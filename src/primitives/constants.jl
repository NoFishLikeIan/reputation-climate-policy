const ctoCO2 = 44 / 12
const taxfactor = ctoCO2 * 1e9 * 1e-12 # USD / tCO2 to trillion USD / GtC

const defaulte0 = 37.8 / ctoCO2
const defaulty0 = 197.231
const defaultscc = 66 / 1000
const defaultdietzphi = 3e-5
const defaulttax0 = 3 * taxfactor
