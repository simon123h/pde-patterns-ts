import { describe, expect, it } from 'vitest'
import { models } from '../src/models'
import { defaultModelSettings, loadSettings, sanitizeModelSettings, saveSettings, type SettingsStorage } from '../src/settings'

const fakeStorage = (): SettingsStorage & { data: Map<string, string> } => {
  const data = new Map<string, string>()
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v) }
}

describe.each(models)('settings of $id', (model) => {
  it('defaults stay in range and sanitizing nothing gives the defaults', () => {
    expect(sanitizeModelSettings(model, undefined)).toEqual(defaultModelSettings(model))
    expect(sanitizeModelSettings(model, null)).toEqual(defaultModelSettings(model))
    expect(sanitizeModelSettings(model, 'junk')).toEqual(defaultModelSettings(model))
  })

  it('keeps valid stored values', () => {
    const stored = { ...defaultModelSettings(model), dt: model.dt.default * 0.5, adaptive: true, stepsPerFrame: 3 }
    expect(sanitizeModelSettings(model, JSON.parse(JSON.stringify(stored)))).toEqual(stored)
  })

  it('clamps out-of-range numbers and ignores values of the wrong type', () => {
    const p = model.params[0]!
    const result = sanitizeModelSettings(model, {
      params: { [p.key]: p.max + 1e6, unknown: 5 },
      dt: -1,
      domain: 'big',
      stepsPerFrame: 1e9,
      viewIndex: 99,
      adaptive: 'yes',
      toleranceExp: NaN,
    })
    expect(result.params[p.key]).toBe(p.max)
    expect(result.params).not.toHaveProperty('unknown')
    expect(result.dt).toBeGreaterThan(0)
    expect(result.domain).toBe(model.domain.default)
    expect(result.stepsPerFrame).toBe(20)
    expect(result.viewIndex).toBe(model.views.length - 1)
    expect(result.adaptive).toBe(false)
    expect(result.toleranceExp).toBe(-3)
  })
})

describe('color range settings', () => {
  const model = models[0]!
  it('keeps a valid fixed range', () => {
    const result = sanitizeModelSettings(model, { rangeMode: 'manual', rangeLo: -2, rangeHi: 5 })
    expect(result).toMatchObject({ rangeMode: 'manual', rangeLo: -2, rangeHi: 5 })
  })
  it('ignores an inverted or invalid range and an unknown mode', () => {
    const result = sanitizeModelSettings(model, { rangeMode: 'sometimes', rangeLo: 3, rangeHi: 1 })
    expect(result).toMatchObject({ rangeMode: 'auto', rangeLo: 0, rangeHi: 1 })
  })
})

describe('blow-up protection setting', () => {
  const byId = (id: string) => models.find((m) => m.id === id)!
  it('starts on, except for the equations that opt out', () => {
    expect(defaultModelSettings(byId('swift-hohenberg')).blowupProtection).toBe(true)
    expect(defaultModelSettings(byId('kuramoto-sivashinsky')).blowupProtection).toBe(false)
    expect(defaultModelSettings(byId('nikolaevskiy')).blowupProtection).toBe(false)
  })
  it('can be switched either way and ignores other types', () => {
    expect(sanitizeModelSettings(byId('nikolaevskiy'), { blowupProtection: true }).blowupProtection).toBe(true)
    expect(sanitizeModelSettings(byId('swift-hohenberg'), { blowupProtection: false }).blowupProtection).toBe(false)
    expect(sanitizeModelSettings(byId('swift-hohenberg'), { blowupProtection: 'no' }).blowupProtection).toBe(true)
  })
})

describe('stored settings', () => {
  it('round-trips', () => {
    const storage = fakeStorage()
    saveSettings({ global: { model: 'cgle', gridSize: 128 }, models: { cgle: { dt: 0.05 } } }, storage)
    expect(loadSettings(storage)).toEqual({ global: { model: 'cgle', gridSize: 128 }, models: { cgle: { dt: 0.05 } } })
  })

  it('survives corrupt, foreign-version and missing data', () => {
    const empty = { global: {}, models: {} }
    const storage = fakeStorage()
    expect(loadSettings(storage)).toEqual(empty)
    storage.setItem('pde-patterns:settings', '{not json')
    expect(loadSettings(storage)).toEqual(empty)
    storage.setItem('pde-patterns:settings', JSON.stringify({ version: 99, global: { model: 'x' } }))
    expect(loadSettings(storage)).toEqual(empty)
    expect(loadSettings(undefined)).toEqual(empty)
  })

  it('ignores a storage that throws', () => {
    const broken: SettingsStorage = {
      getItem: () => {
        throw new Error('denied')
      },
      setItem: () => {
        throw new Error('full')
      },
    }
    expect(loadSettings(broken)).toEqual({ global: {}, models: {} })
    expect(() => saveSettings({ global: {}, models: {} }, broken)).not.toThrow()
  })
})
