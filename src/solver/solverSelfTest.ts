import { requestGpu } from '../gpu/device'
import { swiftHohenberg } from '../models/swiftHohenberg'
import { SpectralSolver } from './spectralSolver'

const N = 32
// 2 wavelengths of cos(k x) with k = 1 fit the domain, so mode (mx = 2, my = 0) has k² = 1
const GRID = { nx: N, ny: N, dx: (2 * Math.PI * 2) / N }

async function readPhysical(device: GPUDevice, solver: SpectralSolver): Promise<Float32Array> {
  const bytes = solver.cells * 8
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  solver.encodeView(enc)
  enc.copyBufferToBuffer(solver.physical, 0, staging, 0, bytes)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const data = new Float32Array(staging.getMappedRange().slice(0))
  staging.unmap()
  staging.destroy()
  return data
}

/** Amplitude of cos(m·2π x / L) in the field (projection), mean-free stripes along x. */
function stripeAmplitude(data: Float32Array, mode: number): number {
  let sum = 0
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) sum += data[2 * (y * N + x)]! * Math.cos((2 * Math.PI * mode * x) / N)
  }
  return (2 * sum) / (N * N)
}

function stripes(mode: number, amplitude: number): Float32Array {
  const init = new Float32Array(N * N)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) init[y * N + x] = amplitude * Math.cos((2 * Math.PI * mode * x) / N)
  return init
}

async function run(device: GPUDevice, init: Float32Array, params: Record<string, number>, h: number, steps: number) {
  const solver = new SpectralSolver(device, swiftHohenberg, GRID)
  solver.setParams(params)
  solver.setDt(h)
  solver.reset(init)
  // Submit in small batches so slow (software) adapters don't trip the GPU watchdog.
  for (let done = 0; done < steps; done += 25) {
    const enc = device.createCommandEncoder()
    solver.step(enc, Math.min(25, steps - done))
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }
  const data = await readPhysical(device, solver)
  const time = solver.time
  solver.destroy()
  return { data, time }
}

export async function runSolverSelfTest(log: (line: string) => void): Promise<boolean> {
  const { device, adapter } = await requestGpu()
  log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture}`)
  let ok = true
  const check = (name: string, got: number, want: number, tol: number) => {
    const rel = Math.abs(got - want) / Math.abs(want)
    const pass = rel < tol
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} ${name}: got ${got.toPrecision(6)}, want ${want.toPrecision(6)} (rel.err ${rel.toExponential(2)})`)
  }

  // Linear growth of the critical mode (k = 1): u ∝ e^{r t}
  {
    const { data, time } = await run(device, stripes(2, 1e-4), { r: 0.2, g: 0 }, 0.5, 40)
    check('critical mode growth e^{rt}', stripeAmplitude(data, 2) / 1e-4, Math.exp(0.2 * time), 2e-4)
  }
  // Linear decay of a damped mode (k = 2, L = r − 9)
  {
    const { data, time } = await run(device, stripes(4, 1e-3), { r: 0.2, g: 0 }, 0.02, 5)
    check('damped mode decay e^{Lt}', stripeAmplitude(data, 4) / 1e-3, Math.exp((0.2 - 9) * time), 2e-4)
  }
  // Nonlinear saturation of stripes: amplitude → sqrt(4r/3) (weakly nonlinear theory, small r)
  {
    const r = 0.05
    const { data } = await run(device, stripes(2, 0.05), { r, g: 0 }, 2, 150)
    check('stripe saturation amplitude sqrt(4r/3)', stripeAmplitude(data, 2), Math.sqrt((4 * r) / 3), 0.03)
  }
  return ok
}
