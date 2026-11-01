import { requestGpu } from '../gpu/device'
import { brusselator } from '../models/brusselator'
import { cahnHilliard } from '../models/cahnHilliard'
import { ginzburgLandau } from '../models/ginzburgLandau'
import { kuramotoSivashinsky } from '../models/kuramotoSivashinsky'
import { swiftHohenberg } from '../models/swiftHohenberg'
import type { SpectralModel } from '../models/types'
import { SpectralSolver } from './spectralSolver'

const N = 32
// 2 wavelengths of cos(k x) with k = 1 fit the domain, so mode m has k = m / 2
const LENGTH = 4 * Math.PI
const GRID = { nx: N, ny: N, dx: LENGTH / N }

/** Interleaved (re, im) of all fields in physical space after running `steps` steps of size h. */
async function run(
  device: GPUDevice,
  model: SpectralModel,
  init: Float32Array,
  params: Record<string, number>,
  h: number,
  steps: number,
): Promise<{ data: Float32Array; time: number }> {
  const solver = new SpectralSolver(device, model, GRID)
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
  const bytes = solver.total * 8
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  solver.encodeView(enc)
  enc.copyBufferToBuffer(solver.physical, 0, staging, 0, bytes)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const data = new Float32Array(staging.getMappedRange().slice(0))
  staging.unmap()
  staging.destroy()
  const time = solver.time
  solver.destroy()
  return { data, time }
}

/** Real stripes a·cos(2π m x / L), constant in y. */
function stripes(mode: number, amplitude: number): Float32Array {
  const init = new Float32Array(N * N)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) init[y * N + x] = amplitude * Math.cos((2 * Math.PI * mode * x) / N)
  return init
}

/** Complex amplitude c of the e^{+i 2π m x/L} component: (1/M) Σ A e^{-i…}, for the first field. */
function modeAmplitude(data: Float32Array, mode: number): [number, number] {
  let re = 0
  let im = 0
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const a = data[2 * (y * N + x)]!
      const b = data[2 * (y * N + x) + 1]!
      const c = Math.cos((2 * Math.PI * mode * x) / N)
      const s = Math.sin((2 * Math.PI * mode * x) / N)
      re += a * c + b * s
      im += b * c - a * s
    }
  }
  return [re / (N * N), im / (N * N)]
}

const stripeAmplitude = (data: Float32Array, mode: number) => 2 * modeAmplitude(data, mode)[0]
const mean = (data: Float32Array, offset = 0) => {
  let sum = 0
  for (let i = 0; i < N * N; i++) sum += data[2 * (offset + i)]!
  return sum / (N * N)
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

  // Swift–Hohenberg: linear growth of the critical mode (k = 1): u ∝ e^{r t}
  {
    const { data, time } = await run(device, swiftHohenberg, stripes(2, 1e-4), { r: 0.2, g: 0 }, 0.5, 40)
    check('SH critical mode growth e^{rt}', stripeAmplitude(data, 2) / 1e-4, Math.exp(0.2 * time), 2e-4)
  }
  // Swift–Hohenberg: damped mode (k = 2, L = r − 9)
  {
    const { data, time } = await run(device, swiftHohenberg, stripes(4, 1e-3), { r: 0.2, g: 0 }, 0.02, 5)
    check('SH damped mode decay e^{Lt}', stripeAmplitude(data, 4) / 1e-3, Math.exp((0.2 - 9) * time), 2e-4)
  }
  // Cahn–Hilliard: linearization L = k² − γk⁴ (checks the stabilizer split and the −k² multiplier)
  {
    const k = 0.5
    const { data, time } = await run(device, cahnHilliard, stripes(1, 1e-4), { gamma: 1, mean: 0 }, 0.5, 10)
    check('CH linear growth e^{(k²−γk⁴)t}', stripeAmplitude(data, 1) / 1e-4, Math.exp((k * k - k ** 4) * time), 2e-4)
  }
  // Cahn–Hilliard: ∫u is conserved by the nonlinear dynamics
  {
    const init = new Float32Array(N * N)
    for (let i = 0; i < init.length; i++) init[i] = 0.3 + 0.4 * Math.sin(i * 0.37) * Math.cos(i * 0.011)
    let m0 = 0
    for (const v of init) m0 += v
    const { data } = await run(device, cahnHilliard, init, { gamma: 1, mean: 0 }, 0.2, 30)
    check('CH mass conservation', mean(data), m0 / init.length, 1e-5)
  }
  // Kuramoto–Sivashinsky: the gradient term shifts the mean by −¼ a² k² h per unit time
  {
    const a = 0.5
    const k = 1
    const h = 0.01
    const { data } = await run(device, kuramotoSivashinsky, stripes(2, a), { nu: 1 }, h, 1)
    check('KS gradient nonlinearity (mean drift)', mean(data), -0.25 * a * a * k * k * h, 2e-2)
  }
  // Complex Ginzburg–Landau: a small plane wave rotates, A ∝ e^{−iαt} for k = 1 (L = −iα)
  {
    const alpha = 1.5
    const init = new Float32Array(2 * N * N)
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        init[2 * (y * N + x)] = 1e-4 * Math.cos((2 * Math.PI * 2 * x) / N)
        init[2 * (y * N + x) + 1] = 1e-4 * Math.sin((2 * Math.PI * 2 * x) / N)
      }
    }
    const { data, time } = await run(device, ginzburgLandau, init, { alpha, beta: -1.5 }, 0.1, 20)
    const [re, im] = modeAmplitude(data, 2)
    check('CGLE plane-wave amplitude', Math.hypot(re, im) / 1e-4, 1, 2e-4)
    check('CGLE plane-wave phase −αt', Math.atan2(im, re), -alpha * time, 1e-3)
  }
  // Brusselator (2 fields): the homogeneous state (A, B/A) is a fixed point
  {
    const A = 2
    const B = 3.6
    const init = new Float32Array(2 * N * N)
    init.fill(A, 0, N * N)
    init.fill(B / A, N * N)
    const { data } = await run(device, brusselator, init, { A, B, Dv: 8 }, 0.1, 20)
    check('Brusselator fixed point u', mean(data), A, 1e-5)
    check('Brusselator fixed point v', mean(data, N * N), B / A, 1e-5)
  }
  // Swift–Hohenberg: nonlinear saturation of stripes, amplitude → sqrt(4r/3) for small r
  {
    const r = 0.05
    const { data } = await run(device, swiftHohenberg, stripes(2, 0.05), { r, g: 0 }, 2, 150)
    check('SH stripe saturation amplitude sqrt(4r/3)', stripeAmplitude(data, 2), Math.sqrt((4 * r) / 3), 0.03)
  }
  return ok
}
