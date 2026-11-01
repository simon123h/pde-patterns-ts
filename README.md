# PDE Patterns

Real-time simulation of pattern-forming PDEs in two space dimensions, running entirely in the
browser on the GPU (WebGPU). Pseudospectral discretization on a periodic domain, exponential
time integration (ETDRK4), written in TypeScript with a Vue 3 front end.

**Try it: <https://simon123h.github.io/pde-patterns-ts/>** (needs a WebGPU browser, see Requirements). The
self-test pages are deployed with it, e.g. `/fft-test.html`, to check your own GPU.

## Equations

| Model | Equation | Notes |
|---|---|---|
| Swift–Hohenberg | ∂ₜu = r u − (1+∇²)²u + g u² − u³ | stripes, hexagons, labyrinths |
| Cahn–Hilliard | ∂ₜu = ∇²(u³ − u) − γ∇⁴u | spinodal decomposition, coarsening, conserves ∫u |
| Kuramoto–Sivashinsky | ∂ₜu = −∇²u − ν∇⁴u − ½\|∇u\|² | spatiotemporal chaos |
| Nikolaevskiy | ∂ₜu = −∇²[ε − (1+∇²)²]u − ½\|∇u\|² | soft-mode turbulence |
| Complex Ginzburg–Landau | ∂ₜA = A + (1+iα)∇²A − (1+iβ)\|A\|²A | spirals, defect/phase turbulence; views \|A\|, Re, arg |
| Gray–Scott | u,v reaction–diffusion | spots, worms, mitosis, chaos |
| Brusselator | u,v reaction–diffusion | Turing and Hopf instabilities |
| Convection–diffusion | ∂ₜu + v·∇u = D∇²u | uniform drift + n×n vortex flow |
| Thin film | ∂ₜh = ∇·(h³∇p), p = −∇²h + h⁻³ − h⁻⁶ | spinodal dewetting into droplets on a precursor film, conserves ∫h |

All models run on a periodic square domain whose side length `L` is a parameter (the grid
spacing is `dx = L/n`).

## Requirements

A browser with WebGPU. Recent Chrome/Edge and Safari work; on Linux the feature may need enabling:

- **Chrome**: start with `google-chrome --enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native --ignore-gpu-blocklist`
  (without the Vulkan flags you may only get the SwiftShader software adapter).
- **Firefox**: set `dom.webgpu.enabled` and `gfx.webgpu.ignore-blocklist` to `true` in `about:config`.

The page must be served from a secure context (`http://localhost` is fine).

## Getting started

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build (BASE_PATH=/sub/path/ for a sub-directory)
npm test           # CPU unit tests (Vitest)
```

### Controls

Pick an equation, adjust its parameters, the domain size `L`, the time step and the number of
steps per frame, and the grid resolution (64²–1024²). **Reset** restarts from the model's initial
condition. **Painting:** drag on the field with the left mouse button to add a Gaussian bump,
with the right button to subtract one (brush size and strength are sliders; what is added to
which field is defined per model, e.g. Gray–Scott adds `v` and removes `u`).

**Adaptive dt:** tick *adaptive dt* to let the solver choose the step size from an error
tolerance (10⁻⁶ … 10⁻¹) up to a maximum dt; the dt slider then sets the starting value, and the
panel shows the current dt and the accepted/rejected step counts. This pays off for Cahn–Hilliard
coarsening, where dt grows by orders of magnitude. Complex fields offer several views (|A|, Re A, arg A); several ColorBrewer and
perceptual colormaps are available, and the diverging ones are centered on zero.

### In-browser self-tests

Some tests need a real WebGPU device, so they are pages of the dev server. Each prints PASS/FAIL
lines and ends with `ALL PASSED`:

| Page | Checks |
|---|---|
| `/fft-test.html` | complex and real-to-complex GPU FFTs against the f64 CPU reference (incl. batching) plus timings |
| `/solver-test.html` | linear growth/decay rates, conservation laws, saturation amplitude, the brush, adaptive stepping and GPU-vs-CPU coefficients (`?only=<group>` runs one group: a model id, `coef`, `adaptive`, `models`; each section logs its time) |
| `/render-test.html` | colormaps, diverging centering and phase view rendered offscreen |

## Numerics

- **Space**: Fourier pseudospectral on a periodic grid, 2/3-rule dealiasing. Pointwise
  nonlinearities are evaluated in physical space; derivatives inside nonlinearities (`∇u`) are
  taken spectrally.
- **Time**: Krogstad's variant of ETDRK4. The (diagonal) linear part is integrated exactly, the
  nonlinearity with a 4-stage exponential Runge–Kutta scheme (4th order, better stiff order than
  Cox–Matthews). Each step costs 4 nonlinear evaluations = 4 inverse + 4 forward FFTs per field.
- **Coefficients**: the φ-functions are evaluated in f64 on the CPU (Taylor series for |z| < 1,
  closed forms beyond; complex symbols are supported) and uploaded as f32, because the φ-functions
  suffer cancellation in f32.
- **Model hooks**: a model supplies the linear symbol `L(k)` (complex), an optional Fourier-space
  multiplier on the nonlinear term (e.g. `−k²` for Cahn–Hilliard's `∇²(u³−3u)` after
  stabilization) and a WGSL snippet for the pointwise nonlinearity.
- **FFT**: custom WGSL implementation. A radix-2 FFT of a whole row lives in workgroup memory,
  twiddles come from an f64 table, and 2D transforms are row-FFT → tiled transpose → row-FFT. The
  spectrum is kept **transposed** (`index = kx·ny + ky`), which saves two transposes per step.
  Real fields (all models except the CGLE) use **real-to-complex** transforms: a real row of n
  samples is transformed as a complex row of n/2 samples and unpacked, so only the half spectrum
  `kx ≤ nx/2` is stored and transformed — about half the memory and work, and the physical
  fields are plain `f32` buffers. Complex fields use full complex transforms.
- **Adaptive time stepping**: the 4th-order ETDRK4 result is compared with the (2nd-order) third
  stage; their difference estimates the local error (∝ h³). A one-workgroup GPU kernel reduces the
  per-workgroup error sums, accepts or rejects the step, picks the next h with a PI controller
  (factor = 0.9·e^(−0.7/3)·e₋^(0.4/3) with e = error/tolerance and e₋ the ratio of the previous
  accepted step, growth limited to 2×; a rejected step is retried with the plain exponent −1/3) and
  a commit kernel applies the new state only if accepted. The ETD coefficients
  for the new h are then recomputed on the GPU (f32, own `sincos` for accurate phases) through an
  indirect dispatch that is skipped while h is unchanged. No CPU round trip is involved in the
  loop; only the statistics are read back asynchronously for display.
- **Mouse brush**: stamps are added directly in Fourier space (the transform of a Gaussian is a
  Gaussian times a phase), so painting costs a single kernel and no extra FFT.
- **Rendering**: fields are drawn straight from the GPU buffer with bilinear periodic sampling;
  the colour range is found by a small compute kernel — nothing is read back to the CPU.

## Project layout

```
src/
  gpu/          device setup, complex and real-to-complex 2D FFTs (WGSL) and their self-test
  math/         CPU reference FFT, φ-functions and ETDRK4 coefficients (f64)
  models/       one file per PDE + the model interface (types.ts)
  solver/       SpectralSolver, coefficient builder, WGSL kernels (combine, spectral derivatives, brush,
                nonlinearity, GPU coefficients, step-size controller)
  render/       FieldRenderer, colormaps and shaders
  components/   Vue UI
tests/          CPU unit tests
*-test.html     in-browser GPU self-tests
```

### Adding a model

Create `src/models/<name>.ts` that calls `defineModel(spec, impl)` and add the result to
`src/models/index.ts`; `src/models/cahnHilliard.ts` is a short example.

- The **spec** is plain data (`ModelSpec` in `src/models/types.ts`): name, description, parameters with
  ranges and defaults, default domain size and dt, steps per frame, views, brush and optional
  `presets`. The UI is built entirely from it.
- The **implementation** (`ModelImpl`) holds the pseudospectral part: the linear symbol, an optional
  Fourier multiplier for the nonlinearity, a WGSL nonlinearity returning a `vec2<f32>` (re, im),
  and the initial condition.

`defineModel` rejects inconsistent models at start-up (defaults out of range, a view of a missing
field, a WGSL body that uses an undeclared parameter, ...). Every registered model, and each of its
presets, is also covered automatically by `tests/models.test.ts` and by the solver self-test.

## Ideas for later

- Non-periodic boundaries through a finite-difference backend
- Better error estimators (embedded ETD pairs) and a PI step-size controller
- Packing two real fields into one complex transform, GPU-resident analysis (spectra, defect counts)
