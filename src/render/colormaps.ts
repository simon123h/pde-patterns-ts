export type RGB = [number, number, number]

export interface ColormapSpec {
  id: string
  label: string
  /** Diverging maps are centered on zero: the color range becomes symmetric, ±max|u|. */
  diverging: boolean
  /** Control points from t = 0 to t = 1, linearly interpolated in sRGB (0–255). */
  stops: RGB[]
}

/** ColorBrewer RdBu (11 classes), reversed so that low values are blue and high values red. */
const RD_BU_REVERSED: RGB[] = [
  [5, 48, 97],
  [33, 102, 172],
  [67, 147, 195],
  [146, 197, 222],
  [209, 229, 240],
  [247, 247, 247],
  [253, 219, 199],
  [244, 165, 130],
  [214, 96, 77],
  [178, 24, 43],
  [103, 0, 31],
]

/** ColorBrewer PRGn (11 classes), reversed: purple (low) – white – green (high). */
const PRGN: RGB[] = [
  [64, 0, 75],
  [118, 42, 131],
  [153, 112, 171],
  [194, 165, 207],
  [231, 212, 232],
  [247, 247, 247],
  [217, 240, 211],
  [166, 219, 160],
  [90, 174, 97],
  [27, 120, 55],
  [0, 68, 27],
]

/** ColorBrewer Spectral (11 classes), reversed so that low values are blue/purple and high red. */
const SPECTRAL_REVERSED: RGB[] = [
  [94, 79, 162],
  [50, 136, 189],
  [102, 194, 165],
  [171, 221, 164],
  [230, 245, 152],
  [255, 255, 191],
  [254, 224, 139],
  [253, 174, 97],
  [244, 109, 67],
  [213, 62, 79],
  [158, 1, 66],
]

/** ColorBrewer YlGnBu (9 classes). */
const YL_GN_BU: RGB[] = [
  [255, 255, 217],
  [237, 248, 177],
  [199, 233, 180],
  [127, 205, 187],
  [65, 182, 196],
  [29, 145, 192],
  [34, 94, 168],
  [37, 52, 148],
  [8, 29, 88],
]

/** Sampled from matplotlib's inferno. */
const INFERNO: RGB[] = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
]

/** Matt Zucker's polynomial fit of matplotlib's viridis, sampled into stops. */
function viridisStops(count = 32): RGB[] {
  const c = [
    [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
    [0.1050930431085774, 1.404613529898575, 1.384590162594685],
    [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
    [-4.634230498983486, -5.799100973351585, -19.33244095627987],
    [6.228269936347081, 14.17993336680509, 56.69055260068105],
    [4.776384997670288, -13.74514537774601, -65.35303263337234],
    [-5.435455855934631, 4.645852612178535, 26.3124352495832],
  ] as const
  return Array.from({ length: count }, (_, i) => {
    const t = i / (count - 1)
    return [0, 1, 2].map((ch) => {
      let v = 0
      for (let d = c.length - 1; d >= 0; d--) v = v * t + c[d]![ch]!
      return Math.round(255 * Math.min(1, Math.max(0, v)))
    }) as RGB
  })
}

export const COLORMAPS: ColormapSpec[] = [
  { id: 'viridis', label: 'viridis', diverging: false, stops: viridisStops() },
  { id: 'inferno', label: 'inferno', diverging: false, stops: INFERNO },
  { id: 'ylgnbu', label: 'YlGnBu (Brewer)', diverging: false, stops: YL_GN_BU },
  { id: 'gray', label: 'gray', diverging: false, stops: [[0, 0, 0], [255, 255, 255]] },
  { id: 'rdbu', label: 'blue–white–red (Brewer RdBu)', diverging: true, stops: RD_BU_REVERSED },
  { id: 'prgn', label: 'purple–white–green (Brewer PRGn)', diverging: true, stops: PRGN },
  { id: 'spectral', label: 'Spectral (Brewer, reversed)', diverging: false, stops: SPECTRAL_REVERSED },
]

export const PALETTE_SIZE = 256

/** All colormaps sampled into one table: map m, entry i at index m·256 + i, as RGBA floats in [0, 1]. */
export function buildPaletteTable(): Float32Array<ArrayBuffer> {
  const table = new Float32Array(COLORMAPS.length * PALETTE_SIZE * 4)
  COLORMAPS.forEach((map, m) => {
    for (let i = 0; i < PALETTE_SIZE; i++) {
      const x = (i / (PALETTE_SIZE - 1)) * (map.stops.length - 1)
      const lo = Math.min(Math.floor(x), map.stops.length - 2)
      const f = x - lo
      for (let ch = 0; ch < 3; ch++) {
        const a = map.stops[lo]![ch]!
        const b = map.stops[lo + 1]![ch]!
        table[(m * PALETTE_SIZE + i) * 4 + ch] = (a + (b - a) * f) / 255
      }
      table[(m * PALETTE_SIZE + i) * 4 + 3] = 1
    }
  })
  return table
}

/**
 * CSS gradient (bottom to top) of a colormap, for a color bar. The stops of a map are evenly spaced, as in the
 * palette table; the phase view uses the cyclic palette of the render shader instead.
 */
export function gradientCss(colormap: string, cyclic: boolean): string {
  if (cyclic) {
    const colors = Array.from({ length: 25 }, (_, i) => {
      const t = i / 24
      const c = [0, 0.33333, 0.66667].map((phase) => Math.round(255 * (0.5 + 0.5 * Math.cos(2 * Math.PI * (t + phase)))))
      return `rgb(${c.join(',')}) ${(t * 100).toFixed(1)}%`
    })
    return `linear-gradient(to top, ${colors.join(', ')})`
  }
  const map = COLORMAPS.find((m) => m.id === colormap) ?? COLORMAPS[0]!
  const n = map.stops.length - 1
  const colors = map.stops.map((c, i) => `rgb(${c.map(Math.round).join(',')}) ${((i / n) * 100).toFixed(1)}%`)
  return `linear-gradient(to top, ${colors.join(', ')})`
}
