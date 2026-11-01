# PDE Patterns

Real-time simulation of pattern-forming PDEs in two space dimensions, running entirely in the
browser on the GPU (WebGPU). Pseudospectral discretization on a periodic domain, exponential
time integration (ETDRK4), written in TypeScript with a Vue 3 front end.

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
npm run build      # typecheck + production build
npm test           # CPU unit tests (Vitest)
```

### Controls

Pick an equation, adjust its parameters, the domain size `L`, the time step and the number of
steps per frame, and the grid resolution (64²–1024²). **Reset** restarts from the model's initial
condition. Complex fields offer several views (|A|, Re A, arg A); several ColorBrewer and
perceptual colormaps are available, and the diverging ones are centered on zero.

### In-browser self-tests

Some tests need a real WebGPU device, so they are pages of the dev server. Each prints PASS/FAIL
lines and ends with `ALL PASSED`:

| Page | Checks |
|---|---|
| `/fft-test.html` | GPU FFT against the f64 CPU reference (incl. batching) plus timings |
| `/solver-test.html` | linear growth/decay rates, conservation laws, saturation amplitude for the models |
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
- **FFT**: custom WGSL implementation. A radix-2 FFT of a whole row lives in workgroup memory
  (rows up to 2048), twiddles come from an f64 table, and 2D transforms are
  row-FFT → tiled transpose → row-FFT. The spectrum is kept **transposed**
  (`index = kx·ny + ky`), which saves two transposes per step.
- **Rendering**: fields are drawn straight from the GPU buffer with bilinear periodic sampling;
  the colour range is found by a small compute kernel — nothing is read back to the CPU.

## Project layout

```
src/
  gpu/          device setup, 2D FFT (WGSL) and its self-test
  math/         CPU reference FFT, φ-functions and ETDRK4 coefficients (f64)
  models/       one file per PDE + the model interface (types.ts)
  solver/       SpectralSolver, coefficient builder, WGSL kernels (combine, gradient, nonlinearity)
  render/       FieldRenderer, colormaps and shaders
  components/   Vue UI
tests/          CPU unit tests
*-test.html     in-browser GPU self-tests
```

### Adding a model

Create `src/models/<name>.ts` exporting a `SpectralModel` (see `src/models/types.ts` for the
documented fields) and add it to `src/models/index.ts`. A model with a real field needs, at a
minimum: the parameters, a linear symbol, a WGSL nonlinearity returning a `vec2<f32>` (re, im),
a view list and an initial condition. Everything else — UI, stepping, rendering — is generic.

## Roadmap

- mouse painting (perturb the field interactively)
- real-to-complex FFTs (half the memory and transform work for real fields)
- adaptive time stepping
