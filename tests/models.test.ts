import { describe, expect, it } from 'vitest'
import { models } from '../src/models'
import { buildCoefficients, buildSymbols, spectralModes } from '../src/solver/coefficients'

describe.each(models)('model $id', (model) => {
  const grid = { nx: 16, ny: 16, dx: model.domain.default / 16 }
  const params = Object.fromEntries(model.params.map((p) => [p.key, p.default]))

  it('has a unique id', () => {
    expect(models.filter((m) => m.id === model.id)).toHaveLength(1)
  })

  it('builds finite ETDRK4 coefficients and symbols at the default step', () => {
    const coefficients = buildCoefficients(model, params, grid, model.dt.default)
    expect(coefficients.every(Number.isFinite)).toBe(true)
    expect(buildSymbols(model, params, grid).every(Number.isFinite)).toBe(true)
  })

  it('has a linear part that is damped at very high wavenumbers', () => {
    for (let field = 0; field < model.fields; field++) expect(model.linearSymbol(1e6, field, params, 1e3, 0)).toBeLessThan(0)
  })

  it('creates an initial state of the right size', () => {
    const state = model.initial({ nx: grid.nx, ny: grid.ny, lx: model.domain.default, params })
    expect(state).toHaveLength(2 * model.fields * grid.nx * grid.ny)
    expect(state.every(Number.isFinite)).toBe(true)
  })

  it('keeps the spectral layout consistent with its field type', () => {
    expect(spectralModes(model, grid)).toBe((model.complex ? 16 : 9) * 16)
  })
})
