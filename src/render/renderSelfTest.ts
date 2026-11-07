import { requestGpu } from '../gpu/device'
import { FieldRenderer } from './fieldRenderer'

const N = 64

async function renderPixels(
  device: GPUDevice,
  data: Float32Array<ArrayBuffer>,
  kind: 're' | 'im' | 'abs' | 'arg',
  colormap: string,
  real = false,
  limits: { lo: number; hi: number } | null = null,
): Promise<Uint8Array> {
  const field = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(field, 0, data)
  const texture = device.createTexture({
    size: [N, N],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const renderer = new FieldRenderer(device, texture, field, N, N, real)
  const staging = device.createBuffer({ size: N * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  renderer.render(enc, 0, kind, colormap, limits)
  enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: N * 4 }, [N, N])
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const pixels = new Uint8Array(staging.getMappedRange().slice(0))
  staging.unmap()
  for (const o of [field, staging]) o.destroy()
  texture.destroy()
  renderer.destroy()
  return pixels
}

/** The automatic color range the renderer reports for `data` after drawing it once. */
async function readRange(device: GPUDevice, data: Float32Array<ArrayBuffer>): Promise<{ lo: number; hi: number } | null> {
  const field = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(field, 0, data)
  const texture = device.createTexture({ size: [N, N], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT })
  const renderer = new FieldRenderer(device, texture, field, N, N)
  const enc = device.createCommandEncoder()
  renderer.render(enc, 0, 're', 'viridis')
  device.queue.submit([enc.finish()])
  const range = await renderer.readRange()
  field.destroy()
  texture.destroy()
  renderer.destroy()
  return range
}

/** Interleaved complex field whose real part follows `f(x)` (constant in y). */
const ramp = (f: (x: number) => number) => {
  const data = new Float32Array(2 * N * N)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) data[2 * (y * N + x)] = f(x)
  return data
}

export async function runRenderSelfTest(log: (line: string) => void): Promise<boolean> {
  const { device, adapter } = await requestGpu()
  log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture}`)
  let ok = true
  const expect = (name: string, pixels: Uint8Array, x: number, want: [number, number, number], tol = 3) => {
    const got = [0, 1, 2].map((c) => pixels[(32 * N + x) * 4 + c]!)
    const pass = got.every((v, c) => Math.abs(v - want[c]!) <= tol)
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} ${name} x=${x}: got rgb(${got.join(', ')}), want rgb(${want.join(', ')})`)
  }

  const up = ramp((x) => -1 + (2 * x) / (N - 1))
  let px = await renderPixels(device, up, 're', 'viridis')
  expect('viridis low', px, 0, [68, 1, 84], 6)
  expect('viridis high', px, N - 1, [253, 231, 37], 6)

  // the same ramp stored as plain f32 (real fields) must look identical
  const upReal = new Float32Array(N * N)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) upReal[y * N + x] = up[2 * (y * N + x)]!
  const pxReal = await renderPixels(device, upReal, 're', 'viridis', true)
  expect('real f32 field, viridis low', pxReal, 0, [68, 1, 84], 6)
  expect('real f32 field, viridis high', pxReal, N - 1, [253, 231, 37], 6)

  px = await renderPixels(device, up, 're', 'rdbu')
  expect('blue-white-red low', px, 0, [5, 48, 97])
  expect('blue-white-red middle', px, 31, [247, 247, 247], 14)
  expect('blue-white-red high', px, N - 1, [103, 0, 31])

  // all-positive data on rdbu: limits are independent [0, 1], so value 0 is at the low end (blue)
  px = await renderPixels(device, ramp((x) => x / (N - 1)), 're', 'rdbu')
  expect('rdbu independent limits (value 0 is low)', px, 0, [5, 48, 97])
  expect('rdbu independent limits (value 1 is high)', px, N - 1, [103, 0, 31])

  // fixed limits: the ramp 0…1 on the range [0, 2] only reaches the middle of the colormap
  px = await renderPixels(device, ramp((x) => x / (N - 1)), 're', 'viridis', false, { lo: 0, hi: 2 })
  expect('manual limits [0, 2]: value 0', px, 0, [68, 1, 84], 6)
  expect('manual limits [0, 2]: value 1 is mid-map', px, N - 1, [33, 145, 140], 8)
  // and the automatic range can be read back for the color bar
  const range = await readRange(device, up)
  const rangeOk = range !== null && Math.abs(range.lo + 1) < 1e-5 && Math.abs(range.hi - 1) < 1e-5
  ok &&= rangeOk
  log(`${rangeOk ? 'PASS' : 'FAIL'} automatic range read back: ${range ? `[${range.lo}, ${range.hi}]` : 'null'}, want [-1, 1]`)

  px = await renderPixels(device, up, 're', 'gray')
  expect('gray low', px, 0, [0, 0, 0])
  expect('gray high', px, N - 1, [255, 255, 255])

  // |A| of a complex field and the phase colormap
  const phase = new Float32Array(2 * N * N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const th = -Math.PI + (2 * Math.PI * (x + 0.5)) / N
      phase[2 * (y * N + x)] = 3 * Math.cos(th)
      phase[2 * (y * N + x) + 1] = 3 * Math.sin(th)
    }
  }
  px = await renderPixels(device, phase, 'abs', 'gray')
  expect('abs of constant-modulus field (flat)', px, 10, [px[(32 * N + 40) * 4]!, px[(32 * N + 40) * 4 + 1]!, px[(32 * N + 40) * 4 + 2]!])
  px = await renderPixels(device, phase, 'arg', 'viridis')
  const a = [0, 1, 2].map((c) => px[(32 * N + 0) * 4 + c]!)
  const b = [0, 1, 2].map((c) => px[(32 * N + 32) * 4 + c]!)
  const differs = a.some((v, c) => Math.abs(v - b[c]!) > 40)
  ok &&= differs
  log(`${differs ? 'PASS' : 'FAIL'} phase view uses a cyclic palette: left rgb(${a.join(', ')}), middle rgb(${b.join(', ')})`)
  return ok
}
