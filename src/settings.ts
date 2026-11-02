import type { SpectralModel } from './models/types'

const KEY = 'pde-patterns:settings'
const VERSION = 1

/** Everything the control panel remembers per equation. */
export interface ModelSettings {
  params: Record<string, number>
  dt: number
  maxDt: number
  domain: number
  stepsPerFrame: number
  viewIndex: number
  adaptive: boolean
  toleranceExp: number
  /** Brush radius in percent of the domain length. */
  brushSize: number
  /** Color range: fitted to the field (`auto`) or fixed to [rangeLo, rangeHi]. */
  rangeMode: 'auto' | 'manual'
  rangeLo: number
  rangeHi: number
}

/** Settings shared by all equations. */
export interface GlobalSettings {
  model: string
  gridSize: number
  colormap: string
  brushStrength: number
}

export interface StoredSettings {
  global: Partial<GlobalSettings>
  models: Record<string, unknown>
}

/** The part of `Storage` that is used, so that tests can pass a fake. */
export type SettingsStorage = Pick<Storage, 'getItem' | 'setItem'>

/** The settings a model starts with: all its defaults. */
export function defaultModelSettings(model: SpectralModel): ModelSettings {
  return {
    params: Object.fromEntries(model.params.map((p) => [p.key, p.default])),
    dt: model.dt.default,
    maxDt: model.dt.max,
    domain: model.domain.default,
    stepsPerFrame: model.stepsPerFrame,
    viewIndex: 0,
    adaptive: false,
    toleranceExp: -3,
    brushSize: (model.paint?.sigma ?? 0.03) * 100,
    rangeMode: 'auto',
    rangeLo: 0,
    rangeHi: 1,
  }
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

/**
 * Merges stored settings onto the model's defaults. Anything that is missing, of the wrong type
 * or no longer in range (the model's ranges may have changed since it was saved) falls back to
 * the default or is clamped, so a stale or hand-edited entry can never break the app.
 */
export function sanitizeModelSettings(model: SpectralModel, raw: unknown): ModelSettings {
  const settings = defaultModelSettings(model)
  if (typeof raw !== 'object' || raw === null) return settings
  const stored = raw as Record<string, unknown>
  const number = (key: keyof ModelSettings, min: number, max: number) => {
    const value = stored[key]
    if (finite(value)) (settings[key] as number) = clamp(value, min, max)
  }
  const storedParams = typeof stored.params === 'object' && stored.params !== null ? (stored.params as Record<string, unknown>) : {}
  for (const p of model.params) {
    const value = storedParams[p.key]
    if (finite(value)) settings.params[p.key] = clamp(value, p.min, p.max)
  }
  number('dt', model.dt.min * 0.01, model.dt.max)
  number('maxDt', model.dt.min, model.dt.max * 20)
  number('domain', model.domain.min, model.domain.max)
  number('stepsPerFrame', 1, 20)
  number('toleranceExp', -6, -1)
  number('brushSize', 0.5, 15)
  if (finite(stored.viewIndex) && Number.isInteger(stored.viewIndex)) settings.viewIndex = clamp(stored.viewIndex, 0, model.views.length - 1)
  if (typeof stored.adaptive === 'boolean') settings.adaptive = stored.adaptive
  if (stored.rangeMode === 'auto' || stored.rangeMode === 'manual') settings.rangeMode = stored.rangeMode
  if (finite(stored.rangeLo) && finite(stored.rangeHi) && stored.rangeLo < stored.rangeHi) {
    settings.rangeLo = stored.rangeLo
    settings.rangeHi = stored.rangeHi
  }
  return settings
}

/** Reads the stored settings; anything unreadable gives empty settings. */
export function loadSettings(storage: SettingsStorage | undefined = defaultStorage()): StoredSettings {
  const empty: StoredSettings = { global: {}, models: {} }
  try {
    const parsed: unknown = JSON.parse(storage?.getItem(KEY) ?? 'null')
    if (typeof parsed !== 'object' || parsed === null) return empty
    const data = parsed as { version?: unknown; global?: unknown; models?: unknown }
    if (data.version !== VERSION) return empty
    return {
      global: typeof data.global === 'object' && data.global !== null ? (data.global as Partial<GlobalSettings>) : {},
      models: typeof data.models === 'object' && data.models !== null ? (data.models as Record<string, unknown>) : {},
    }
  } catch {
    return empty
  }
}

/** Writes the settings; storage may be unavailable or full (private windows), which is ignored. */
export function saveSettings(settings: StoredSettings, storage: SettingsStorage | undefined = defaultStorage()): void {
  try {
    storage?.setItem(KEY, JSON.stringify({ version: VERSION, ...settings }))
  } catch {
    // not persisting is fine
  }
}

function defaultStorage(): SettingsStorage | undefined {
  try {
    return globalThis.localStorage
  } catch {
    return undefined
  }
}
