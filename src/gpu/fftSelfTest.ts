import { fft2d } from '../math/fft'
import { Fft2D } from './fft'
import { RealFft2D } from './realFft'
import { requestGpu } from './device'

export interface FftTestResult {
  nx: number
  ny: number
  batch: number
  forwardError: number
  roundTripError: number
  ms?: number
}

async function readback(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(buffer, 0, staging, 0, bytes)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(staging.getMappedRange().slice(0))
  staging.unmap()
  staging.destroy()
  return out
}

/** Compares the GPU FFT against the f64 CPU reference. Errors are relative to the max magnitude. */
export async function testFft(device: GPUDevice, nx: number, ny: number, batch = 1): Promise<FftTestResult> {
  const m = nx * ny
  const n = m * batch
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  const input = new Float32Array(2 * n)
  for (let i = 0; i < n; i++) {
    re[i] = Math.fround(Math.random() * 2 - 1)
    im[i] = Math.fround(Math.random() * 2 - 1)
    input[2 * i] = re[i]!
    input[2 * i + 1] = im[i]!
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  const a = device.createBuffer({ size: n * 8, usage })
  const b = device.createBuffer({ size: n * 8, usage })
  const c = device.createBuffer({ size: n * 8, usage })
  device.queue.writeBuffer(a, 0, input)

  const fft = new Fft2D(device, nx, ny, batch)
  const enc = device.createCommandEncoder()
  fft.forward(enc, a, b)
  fft.inverse(enc, b, c)
  device.queue.submit([enc.finish()])
  const spectrum = await readback(device, b, n * 8)
  const back = await readback(device, c, n * 8)

  // CPU reference per batched field, spectrum transposed to the GPU layout: index kx * ny + ky
  let maxMag = 0
  let fwdErr = 0
  for (let f = 0; f < batch; f++) {
    const fre = re.subarray(f * m, (f + 1) * m)
    const fim = im.subarray(f * m, (f + 1) * m)
    fft2d(fre, fim, nx, ny)
    for (let ky = 0; ky < ny; ky++) {
      for (let kx = 0; kx < nx; kx++) {
        const cpu = ky * nx + kx
        const gpu = f * m + kx * ny + ky
        maxMag = Math.max(maxMag, Math.hypot(fre[cpu]!, fim[cpu]!))
        fwdErr = Math.max(fwdErr, Math.hypot(spectrum[2 * gpu]! - fre[cpu]!, spectrum[2 * gpu + 1]! - fim[cpu]!))
      }
    }
  }
  let rtErr = 0
  for (let i = 0; i < 2 * n; i++) rtErr = Math.max(rtErr, Math.abs(back[i]! - input[i]!))

  for (const buf of [a, b, c]) buf.destroy()
  fft.destroy()
  return { nx, ny, batch, forwardError: fwdErr / maxMag, roundTripError: rtErr }
}

/** Real-to-complex FFT against the f64 CPU reference (half spectrum, transposed layout). */
export async function testRealFft(device: GPUDevice, nx: number, ny: number, batch = 1): Promise<FftTestResult> {
  const m = nx * ny
  const h = nx / 2 + 1
  const n = m * batch
  const input = new Float32Array(n)
  for (let i = 0; i < n; i++) input[i] = Math.fround(Math.random() * 2 - 1)

  const a = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const spectrumBytes = batch * h * ny * 8
  const b = device.createBuffer({ size: spectrumBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const c = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  device.queue.writeBuffer(a, 0, input)

  const fft = new RealFft2D(device, nx, ny, batch)
  const enc = device.createCommandEncoder()
  fft.forward(enc, a, b)
  fft.inverse(enc, b, c)
  device.queue.submit([enc.finish()])
  const spectrum = await readback(device, b, spectrumBytes)
  const back = await readback(device, c, n * 4)

  let maxMag = 0
  let fwdErr = 0
  for (let f = 0; f < batch; f++) {
    const re = new Float64Array(m)
    const im = new Float64Array(m)
    for (let i = 0; i < m; i++) re[i] = input[f * m + i]!
    fft2d(re, im, nx, ny)
    for (let ky = 0; ky < ny; ky++) {
      for (let kx = 0; kx < h; kx++) {
        const cpu = ky * nx + kx
        const gpu = f * h * ny + kx * ny + ky
        maxMag = Math.max(maxMag, Math.hypot(re[cpu]!, im[cpu]!))
        fwdErr = Math.max(fwdErr, Math.hypot(spectrum[2 * gpu]! - re[cpu]!, spectrum[2 * gpu + 1]! - im[cpu]!))
      }
    }
  }
  let rtErr = 0
  for (let i = 0; i < n; i++) rtErr = Math.max(rtErr, Math.abs(back[i]! - input[i]!))

  for (const buf of [a, b, c]) buf.destroy()
  fft.destroy()
  return { nx, ny, batch, forwardError: fwdErr / maxMag, roundTripError: rtErr }
}

export async function timeFft(device: GPUDevice, nx: number, ny: number, real = false, iterations = 20): Promise<number> {
  const n = nx * ny
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  const fft = real ? new RealFft2D(device, nx, ny) : new Fft2D(device, nx, ny)
  const a = device.createBuffer({ size: real ? n * 4 : n * 8, usage })
  const b = device.createBuffer({ size: real ? fft.modes * 8 : n * 8, usage })
  const run = async (count: number) => {
    const enc = device.createCommandEncoder()
    for (let i = 0; i < count; i++) {
      fft.forward(enc, a, b)
      fft.inverse(enc, b, a)
    }
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }
  await run(1) // warm-up
  const t0 = performance.now()
  await run(iterations)
  const ms = (performance.now() - t0) / iterations
  a.destroy()
  b.destroy()
  fft.destroy()
  return ms
}

export async function runSelfTest(log: (line: string) => void): Promise<boolean> {
  const { device, adapter } = await requestGpu()
  log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture}`)
  let ok = true
  for (const [nx, ny, batch] of [[2, 2, 1], [16, 16, 1], [64, 64, 1], [256, 128, 1], [128, 512, 1], [2048, 16, 1], [16, 2048, 1], [512, 512, 1], [64, 32, 3]] as const) {
    const r = await testFft(device, nx, ny, batch)
    const pass = r.forwardError < 2e-5 && r.roundTripError < 1e-4
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} ${nx}x${ny}${batch > 1 ? ` ×${batch}` : ''}  forward rel.err=${r.forwardError.toExponential(2)}  round-trip abs.err=${r.roundTripError.toExponential(2)}`)
  }
  for (const [nx, ny, batch] of [[4, 4, 1], [8, 4, 1], [16, 16, 1], [64, 64, 1], [256, 128, 1], [128, 512, 1], [4096, 4, 1], [16, 2048, 1], [512, 512, 1], [64, 32, 3]] as const) {
    const r = await testRealFft(device, nx, ny, batch)
    const pass = r.forwardError < 2e-5 && r.roundTripError < 1e-4
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} real ${nx}x${ny}${batch > 1 ? ` ×${batch}` : ''}  forward rel.err=${r.forwardError.toExponential(2)}  round-trip abs.err=${r.roundTripError.toExponential(2)}`)
  }
  for (const size of [512, 1024]) {
    const complex = await timeFft(device, size, size)
    const real = await timeFft(device, size, size, true)
    log(`time ${size}²: complex ${complex.toFixed(2)} ms, real-to-complex ${real.toFixed(2)} ms per forward+inverse`)
  }
  return ok
}
