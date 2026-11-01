import { requestGpu } from '../gpu/device'
import { advectionDiffusion } from '../models/advectionDiffusion'
import { brusselator } from '../models/brusselator'
import { cahnHilliard } from '../models/cahnHilliard'
import { ginzburgLandau } from '../models/ginzburgLandau'
import { models } from '../models'
import { kuramotoSivashinsky } from '../models/kuramotoSivashinsky'
import { nikolaevskiy } from '../models/nikolaevskiy'
import { swiftHohenberg } from '../models/swiftHohenberg'
import { thinFilm } from '../models/thinFilm'
import type { SpectralModel } from '../models/types'
import { buildCoefficients } from './coefficients'
import { SpectralSolver, type AdaptiveOptions, type StepStats } from './spectralSolver'

const N = 32
// 2 wavelengths of cos(k x) with k = 1 fit the domain, so mode m has k = m / 2
const LENGTH = 4 * Math.PI
const GRID = { nx: N, ny: N, dx: LENGTH / N }

// Building a solver compiles its shaders, which takes minutes on a software adapter, so each model's
// solver is built once and reused (reset, parameters, dt and domain are set again for every test).
const solvers = new Map<string, SpectralSolver>()
function solverFor(device: GPUDevice, model: SpectralModel, domain: number): SpectralSolver {
  let solver = solvers.get(model.id)
  if (!solver) {
    solver = new SpectralSolver(device, model, { nx: N, ny: N, dx: domain / N })
    solvers.set(model.id, solver)
  }
  solver.setDomain(domain)
  return solver
}

/** Interleaved (re, im) of all fields in physical space after running `steps` steps of size h. */
async function run(
  device: GPUDevice,
  model: SpectralModel,
  init: Float32Array,
  params: Record<string, number>,
  h: number,
  steps: number,
  grid = GRID,
): Promise<{ data: Float32Array; time: number }> {
  const solver = solverFor(device, model, grid.nx * grid.dx)
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
  const data = await solver.readPhysical()
  return { data, time: solver.time }
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

/** Run `steps` adaptive steps and return the physical state and the controller statistics. */
async function runAdaptive(
  device: GPUDevice,
  model: SpectralModel,
  init: Float32Array,
  params: Record<string, number>,
  h0: number,
  options: AdaptiveOptions,
  steps: number,
): Promise<{ data: Float32Array; stats: StepStats }> {
  const solver = new SpectralSolver(device, model, GRID)
  solver.setParams(params)
  solver.setDt(h0)
  solver.setAdaptive(options)
  solver.reset(init)
  for (let done = 0; done < steps; done += 20) {
    const enc = device.createCommandEncoder()
    solver.step(enc, Math.min(20, steps - done))
    solver.encodeStatsCopy(enc)
    device.queue.submit([enc.finish()])
    await solver.afterSubmit()
    await device.queue.onSubmittedWorkDone()
  }
  // one more round trip so the statistics reflect the last step
  const enc = device.createCommandEncoder()
  solver.encodeStatsCopy(enc)
  device.queue.submit([enc.finish()])
  await solver.afterSubmit()
  const data = await solver.readPhysical()
  const stats = { ...solver.stats }
  solver.destroy()
  return { data, stats }
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

  // `?only=<group>` runs one group of checks: a model id, `models` for every model's smoke test, `coef`,
  // `thin` (the thin-film checks), `adaptive`, `brush`; every section logs how long it took
  const only = new URLSearchParams(location.search).get('only')
  const groups = new Set<string>()
  const section = async (names: string[], title: string, body: () => Promise<void>) => {
    names.forEach((n) => groups.add(n))
    if (only && !names.includes(only)) return
    const started = performance.now()
    await body()
    log(`  [${title}: ${((performance.now() - started) / 1000).toFixed(1)} s]`)
  }

  // GPU-evaluated (f32) coefficients agree with the CPU (f64) ones, for real and complex symbols
  const coefficientTests = async () => {
    for (const [model, params] of [
      [swiftHohenberg, { r: 0.2, g: 0 }],
      [ginzburgLandau, { alpha: 1.5, beta: -1.5 }],
      [cahnHilliard, { gamma: 1, mean: 0 }],
    ] as const) {
      const h = 0.17
      const solver = new SpectralSolver(device, model, GRID)
      solver.setParams(params)
      solver.setAdaptive({ enabled: true, tol: 1e9, hmax: h, hmin: h }) // pins h = 0.17
      solver.setDt(h)
      solver.reset(new Float32Array(2 * solver.total))
      const enc = device.createCommandEncoder()
      solver.step(enc, 1)
      device.queue.submit([enc.finish()])
      const gpu = await solver.readCoefficients()
      const cpu = buildCoefficients(model, params, { ...GRID }, h)
      solver.destroy()
      let worst = 0
      let worstPlane = 0
      const perPlane = cpu.length / 9
      for (let plane = 0; plane < 9; plane++) {
        let scale = 1e-3
        let diff = 0
        for (let i = 0; i < perPlane; i++) {
          const o = plane * perPlane + i
          scale = Math.max(scale, Math.abs(cpu[o]!))
          diff = Math.max(diff, Math.abs(gpu[o]! - cpu[o]!))
        }
        if (diff / scale > worst) {
          worst = diff / scale
          worstPlane = plane
        }
      }
      const pass = worst < 2e-4
      ok &&= pass
      log(`${pass ? 'PASS' : 'FAIL'} GPU vs CPU coefficients (${model.name}): max relative difference ${worst.toExponential(2)} (plane ${worstPlane})`)
    }
  }
  const thinFilmTests = async () => {
    // Thin film: dispersion relation λ = −h₀³k²(k² + f''(h₀)) about the flat film, f'' = −3h⁻⁴ + 6h⁻⁷;
    // mode 1 of a domain of length 8π has k = 1/4 (unstable), mode 2 has k = 1/2 (stable)
    {
      const h0 = 2
      const grid = { nx: N, ny: N, dx: (8 * Math.PI) / N }
      const fpp = -3 / h0 ** 4 + 6 / h0 ** 7
      const lambda = (k: number) => -(h0 ** 3) * k * k * (k * k + fpp)
      const film = (mode: number, amplitude: number) => stripes(mode, amplitude).map((v) => v + h0)
      const mode = (data: Float32Array, m: number, amplitude: number) => stripeAmplitude(data, m) / amplitude
      const params = { h0, noise: 0 }
      const growth = await run(device, thinFilm, film(1, 1e-4), params, 1, 60, grid)
      check('thin film unstable mode growth e^{λt}', mode(growth.data, 1, 1e-4), Math.exp(lambda(0.25) * growth.time), 2e-3)
      // a small step: the stabilizing part of N is large at k = 1/2, which limits ETDRK4's accuracy
      const decay = await run(device, thinFilm, film(2, 1e-3), params, 0.2, 50, grid)
      check('thin film stable mode decay e^{λt}', mode(decay.data, 2, 1e-3), Math.exp(lambda(0.5) * decay.time), 2e-3)
      check('thin film mass conservation', mean(decay.data), h0, 1e-5)
    }
    // Thin film, finite amplitude: ∂t h = (h³ p_x)_x for h = 2 + 0.5 cos(kx), compared with the analytic
    // right-hand side after one short step (checks the nonlinear terms, which linearization does not)
    {
      const grid = { nx: N, ny: N, dx: (8 * Math.PI) / N }
      const k = 0.5
      const a = 0.5
      const dt = 5e-4 // short enough that (h₁ − h₀)/dt approximates ∂t h also for the stiffly damped harmonics
      const fp2 = (h: number) => -3 / h ** 4 + 6 / h ** 7
      const fp3 = (h: number) => 12 / h ** 5 - 42 / h ** 8
      const init = new Float32Array(N * N)
      const rhs = new Float32Array(N)
      for (let x = 0; x < N; x++) {
        const c = Math.cos(k * x * grid.dx)
        const sn = Math.sin(k * x * grid.dx)
        const h = 2 + a * c
        const hx = -a * k * sn
        const hxx = -a * k * k * c
        const hxxx = a * k ** 3 * sn
        const hxxxx = a * k ** 4 * c
        rhs[x] = h ** 3 * (-hxxxx + fp2(h) * hxx + fp3(h) * hx * hx) + 3 * h * h * hx * (-hxxx + fp2(h) * hx)
        for (let y = 0; y < N; y++) init[y * N + x] = h
      }
      const { data } = await run(device, thinFilm, init, { h0: 2, noise: 0 }, dt, 1, grid)
      let err = 0
      let norm = 0
      for (let x = 0; x < N; x++) {
        const rate = (data[2 * x]! - init[x]!) / dt
        err += (rate - rhs[x]!) ** 2
        norm += rhs[x]! ** 2
      }
      const rel = Math.sqrt(err / norm)
      ok &&= rel < 0.02
      log(`${rel < 0.02 ? 'PASS' : 'FAIL'} thin film nonlinear right-hand side: relative L2 error ${rel.toExponential(2)} (rms rhs ${Math.sqrt(norm / N).toExponential(2)})`)
    }
  }
  // Brush: a stamp of amplitude 0.5 (SH paint amount) at a grid node must give a Gaussian of that peak
  await section(['swift-hohenberg'], 'brush', async () => {
      const solver = solverFor(device, swiftHohenberg, LENGTH)
      solver.reset(new Float32Array(N * N))
      const sigma = 0.06 * LENGTH
      const enc = device.createCommandEncoder()
      solver.paint(enc, new Float32Array([LENGTH / 2, LENGTH / 2, 2, 0]), sigma) // scale 2 × amount 0.5
      device.queue.submit([enc.finish()])
      const data = await solver.readPhysical()
      let peak = -Infinity
      let at = 0
      let sum = 0
      for (let i = 0; i < N * N; i++) {
        sum += data[2 * i]!
        if (data[2 * i]! > peak) [peak, at] = [data[2 * i]!, i]
      }
      check('brush peak amplitude', peak, 1, 2e-3)
      ok &&= at === (N / 2) * N + N / 2
      log(`${at === (N / 2) * N + N / 2 ? 'PASS' : 'FAIL'} brush peak located at the stamp position (index ${at})`)
      const dx = LENGTH / N
      check('brush integral 2πσ²/dx²', sum, (2 * Math.PI * sigma * sigma) / (dx * dx), 2e-3)
  })
  // Swift–Hohenberg: linear growth of the critical mode (k = 1): u ∝ e^{r t}
  await section(['swift-hohenberg'], 'SH critical mode', async () => {
      const { data, time } = await run(device, swiftHohenberg, stripes(2, 1e-4), { r: 0.2, g: 0 }, 0.5, 40)
      check('SH critical mode growth e^{rt}', stripeAmplitude(data, 2) / 1e-4, Math.exp(0.2 * time), 2e-4)
  })
  // Swift–Hohenberg: damped mode (k = 2, L = r − 9)
  await section(['swift-hohenberg'], 'SH damped mode', async () => {
      const { data, time } = await run(device, swiftHohenberg, stripes(4, 1e-3), { r: 0.2, g: 0 }, 0.02, 5)
      check('SH damped mode decay e^{Lt}', stripeAmplitude(data, 4) / 1e-3, Math.exp((0.2 - 9) * time), 2e-4)
  })
  await section(['thin-film', 'thin'], 'thin film', thinFilmTests)
  // Cahn–Hilliard: linearization L = k² − γk⁴ (checks the stabilizer split and the −k² multiplier)
  await section(['cahn-hilliard'], 'CH linearization', async () => {
      const k = 0.5
      const { data, time } = await run(device, cahnHilliard, stripes(1, 1e-4), { gamma: 1, mean: 0 }, 0.5, 10)
      check('CH linear growth e^{(k²−γk⁴)t}', stripeAmplitude(data, 1) / 1e-4, Math.exp((k * k - k ** 4) * time), 2e-4)
  })
  // Cahn–Hilliard: ∫u is conserved by the nonlinear dynamics
  await section(['cahn-hilliard'], 'CH mass', async () => {
      const init = new Float32Array(N * N)
      for (let i = 0; i < init.length; i++) init[i] = 0.3 + 0.4 * Math.sin(i * 0.37) * Math.cos(i * 0.011)
      let m0 = 0
      for (const v of init) m0 += v
      const { data } = await run(device, cahnHilliard, init, { gamma: 1, mean: 0 }, 0.2, 30)
      check('CH mass conservation', mean(data), m0 / init.length, 1e-5)
  })
  // Kuramoto–Sivashinsky: the gradient term shifts the mean by −¼ a² k² h per unit time
  await section(['kuramoto-sivashinsky'], 'KS gradient', async () => {
      const a = 0.5
      const k = 1
      const h = 0.01
      const { data } = await run(device, kuramotoSivashinsky, stripes(2, a), { nu: 1 }, h, 1)
      check('KS gradient nonlinearity (mean drift)', mean(data), -0.25 * a * a * k * k * h, 2e-2)
  })
  // Convection–diffusion, uniform drift only: cos(kx) translates by c t and decays as e^{−D k² t}
  await section(['advection-diffusion'], 'advection drift', async () => {
      const D = 0.1
      const cx = 1
      const init = new Float32Array(2 * N * N)
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) init[2 * (y * N + x)] = 1e-3 * Math.cos((2 * Math.PI * 2 * x) / N)
      const { data, time } = await run(device, advectionDiffusion, init, { D, cx, cy: 0, A: 0, n: 2 }, 0.1, 20)
      const [re, im] = modeAmplitude(data, 2)
      check('advection-diffusion decay e^{−Dk²t}', (2 * Math.hypot(re, im)) / 1e-3, Math.exp(-D * time), 2e-4)
      check('advection-diffusion drift phase −c k t', Math.atan2(im, re), -cx * time, 1e-3)
  })
  // Convection–diffusion with the vortex flow: divergence-free, so ∫u is conserved
  await section(['advection-diffusion'], 'advection vortex flow', async () => {
      const init = new Float32Array(N * N)
      for (let i = 0; i < init.length; i++) init[i] = 0.3 + 0.4 * Math.sin(i * 0.37) * Math.cos(i * 0.011)
      let m0 = 0
      for (const v of init) m0 += v
      const { data } = await run(device, advectionDiffusion, init, { D: 0.05, cx: 0, cy: 0, A: 1, n: 2 }, 0.01, 20)
      check('advection-diffusion mass conservation (vortex flow)', mean(data), m0 / init.length, 1e-4)
  })
  // Nikolaevskiy: L = k²[ε − (1 − k²)²], so the k = 1 mode grows at rate ε
  await section(['nikolaevskiy'], 'Nikolaevskiy', async () => {
      const { data, time } = await run(device, nikolaevskiy, stripes(2, 1e-4), { eps: 0.3 }, 0.5, 10)
      check('Nikolaevskiy critical mode growth e^{εt}', stripeAmplitude(data, 2) / 1e-4, Math.exp(0.3 * time), 2e-4)
  })
  // Complex Ginzburg–Landau: a small plane wave rotates, A ∝ e^{−iαt} for k = 1 (L = −iα)
  await section(['cgle'], 'CGLE plane wave', async () => {
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
  })
  // Brusselator (2 fields): the homogeneous state (A, B/A) is a fixed point
  await section(['brusselator'], 'Brusselator', async () => {
      const A = 2
      const B = 3.6
      const init = new Float32Array(2 * N * N)
      init.fill(A, 0, N * N)
      init.fill(B / A, N * N)
      const { data } = await run(device, brusselator, init, { A, B, Dv: 8 }, 0.1, 20)
      check('Brusselator fixed point u', mean(data), A, 1e-5)
      check('Brusselator fixed point v', mean(data, N * N), B / A, 1e-5)
  })
  // Adaptive stepping: the critical mode grows as e^{rt} with t accumulated by the GPU controller
  await section(['adaptive'], 'adaptive growth', async () => {
      const opts = { enabled: true, tol: 1e-4, hmax: 0.5, hmin: 1e-4 }
      const { data, stats } = await runAdaptive(device, swiftHohenberg, stripes(2, 1e-5), { r: 0.2, g: 0 }, 0.01, opts, 60)
      check('adaptive: growth e^{r t_gpu}', stripeAmplitude(data, 2) / 1e-5, Math.exp(0.2 * stats.t), 1e-3)
      const grew = stats.h > 0.4
      ok &&= grew
      log(`${grew ? 'PASS' : 'FAIL'} adaptive: step size grew from 0.01 to ${stats.h.toFixed(3)} (accepted ${stats.accepted}, rejected ${stats.rejected}, t = ${stats.t.toFixed(2)})`)
  })
  // Adaptive stepping: an oversized first step on a nonlinear state gets rejected and the run stays bounded
  await section(['adaptive'], 'adaptive rejection', async () => {
      const opts = { enabled: true, tol: 1e-5, hmax: 2, hmin: 1e-4 }
      const { data, stats } = await runAdaptive(device, swiftHohenberg, stripes(2, 0.8), { r: 0.2, g: 0.5 }, 2, opts, 40)
      let peak = 0
      for (let i = 0; i < N * N; i++) peak = Math.max(peak, Math.abs(data[2 * i]!))
      const pass = stats.rejected > 0 && peak < 3 && Number.isFinite(peak)
      ok &&= pass
      log(`${pass ? 'PASS' : 'FAIL'} adaptive: oversized step rejected (rejected ${stats.rejected}, accepted ${stats.accepted}, h = ${stats.h.toFixed(4)}, max|u| = ${peak.toFixed(3)})`)
  })
  await section(['coef'], 'GPU coefficients', coefficientTests)
  // Swift–Hohenberg: nonlinear saturation of stripes, amplitude → sqrt(4r/3) for small r
  await section(['swift-hohenberg'], 'SH saturation', async () => {
      const r = 0.05
      const { data } = await run(device, swiftHohenberg, stripes(2, 0.05), { r, g: 0 }, 2, 150)
      check('SH stripe saturation amplitude sqrt(4r/3)', stripeAmplitude(data, 2), Math.sqrt((4 * r) / 3), 0.03)
  })
  // Blow-up recovery: a step size far too large for Swift–Hohenberg diverges; the guard must roll back and
  // halve dt until the run is stable again
  await section(['recovery', 'swift-hohenberg'], 'blow-up recovery', async () => {
    const solver = solverFor(device, swiftHohenberg, LENGTH)
    const before = solver.recoveries
    solver.setParams({ r: 0.2, g: 0 })
    solver.setDt(60)
    solver.reset(stripes(2, 0.1))
    for (let i = 0; i < 40; i++) {
      const enc = device.createCommandEncoder()
      solver.step(enc, 4)
      solver.encodeHealthCheck(enc)
      device.queue.submit([enc.finish()])
      await solver.afterSubmit()
      await device.queue.onSubmittedWorkDone()
    }
    const data = await solver.readPhysical()
    const peak = data.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
    const recoveries = solver.recoveries - before
    const pass = recoveries >= 1 && !solver.unstable && Number.isFinite(peak) && peak < 5 && solver.dt < 60
    ok &&= pass
    log(`${pass ? 'PASS' : 'FAIL'} blow-up recovery: ${recoveries} rollback(s), dt 60 → ${solver.dt.toPrecision(3)}, max |u| = ${peak.toPrecision(3)}${solver.unstable ? ' (gave up)' : ''}`)
  })
  // Every registered model runs from its own initial state with its defaults and presets
  for (const model of models) {
    await section([model.id, 'models'], `${model.name} smoke test`, async () => {
      const settings = [{ name: 'defaults', preset: {} as { params?: Record<string, number>; domain?: number; dt?: number } }]
      for (const preset of model.presets ?? []) settings.push({ name: `preset "${preset.name}"`, preset })
      const solver = solverFor(device, model, model.domain.default)
      for (const { name, preset } of settings) {
        const domain = preset.domain ?? model.domain.default
        solver.setParams({ ...Object.fromEntries(model.params.map((p) => [p.key, p.default])), ...preset.params })
        solver.setDomain(domain)
        solver.setDt(preset.dt ?? model.dt.default)
        solver.reset()
        const enc = device.createCommandEncoder()
        solver.step(enc, 10)
        device.queue.submit([enc.finish()])
        await device.queue.onSubmittedWorkDone()
        const data = await solver.readPhysical()
        const peak = data.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
        const pass = Number.isFinite(peak) && peak < 1e3
        ok &&= pass
        log(`${pass ? 'PASS' : 'FAIL'} ${model.name}, ${name}: 10 steps stay finite (max |u| = ${peak.toPrecision(3)})`)
      }
    })
  }
  if (only && !groups.has(only)) {
    log(`FAIL unknown group "${only}"; known groups: ${[...groups].join(', ')}`)
    return false
  }
  return ok
}
