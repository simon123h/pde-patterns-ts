import { fft2d } from '../math/fft'
import { Fft2D } from './fft'
import { requestGpu } from './device'

export interface FftTestResult {
  nx: number
  ny: number
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
export async function testFft(device: GPUDevice, nx: number, ny: number): Promise<FftTestResult> {
  const n = nx * ny
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

  const fft = new Fft2D(device, nx, ny)
  const enc = device.createCommandEncoder()
  fft.forward(enc, a, b)
  fft.inverse(enc, b, c)
  device.queue.submit([enc.finish()])
  const spectrum = await readback(device, b, n * 8)
  const back = await readback(device, c, n * 8)

  // CPU reference, spectrum transposed to the GPU layout: index kx * ny + ky
  fft2d(re, im, nx, ny)
  let maxMag = 0
  let fwdErr = 0
  for (let ky = 0; ky < ny; ky++) {
    for (let kx = 0; kx < nx; kx++) {
      const cpu = ky * nx + kx
      const gpu = kx * ny + ky
      maxMag = Math.max(maxMag, Math.hypot(re[cpu]!, im[cpu]!))
      fwdErr = Math.max(fwdErr, Math.hypot(spectrum[2 * gpu]! - re[cpu]!, spectrum[2 * gpu + 1]! - im[cpu]!))
    }
  }
  let rtErr = 0
  for (let i = 0; i < 2 * n; i++) rtErr = Math.max(rtErr, Math.abs(back[i]! - input[i]!))

  for (const buf of [a, b, c]) buf.destroy()
  fft.destroy()
  return { nx, ny, forwardError: fwdErr / maxMag, roundTripError: rtErr }
}

export async function timeFft(device: GPUDevice, nx: number, ny: number, iterations = 20): Promise<number> {
  const n = nx * ny
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  const a = device.createBuffer({ size: n * 8, usage })
  const b = device.createBuffer({ size: n * 8, usage })
  const fft = new Fft2D(device, nx, ny)
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
  for (const [nx, ny] of [[2, 2], [16, 16], [64, 64], [256, 128], [128, 512], [2048, 16], [16, 2048], [512, 512]] as const) {
    const r = await testFft(device, nx, ny)
    const pass = r.forwardError < 2e-5 && r.roundTripError < 1e-4
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} ${nx}x${ny}  forward rel.err=${r.forwardError.toExponential(2)}  round-trip abs.err=${r.roundTripError.toExponential(2)}`)
  }
  for (const size of [512, 1024]) log(`time ${size}²: ${(await timeFft(device, size, size)).toFixed(2)} ms per forward+inverse`)
  return ok
}
