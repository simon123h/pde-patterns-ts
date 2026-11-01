# Notes for AI agents

Browser PDE pattern-formation simulator: TypeScript, Vue 3, WebGPU, Fourier pseudospectral solver
(ETDRK4). `README.md` describes the numerics and the layout.

## Testing: run the cheap checks, leave the rest to CI

| Command | Cost | When |
|---|---|---|
| `npm run check` (typecheck + unit tests) | < 5 s, CPU only | after every change |
| `node scripts/browser-tests.mjs fft` / `render` / `solver` | minutes, uses all CPU cores | only for the matching code, see below |
| `npm run test:browser` (all three pages) | 10+ min | leave to CI |

The browser tests run the WebGPU code on SwiftShader (a software GPU), which saturates the CPU and
slows the whole machine down. CI runs all of them (the solver page as one job per equation, in parallel) on pull requests that touch code they cover,
so do not run the full browser suite locally by default.

Run a single browser page locally only when you changed the code it covers:
- `fft` for `src/gpu/` (FFT, transpose, real-to-complex)
- `render` for `src/render/`
- `solver` for `src/solver/` (time stepping, coefficients, adaptive dt, shaders). The full page takes
  over 10 minutes, so run one group instead: `node scripts/browser-tests.mjs solver:<group>`, where
  the group is a model id (`cgle`, `thin-film`, `swift-hohenberg`, ...: that model's checks and its
  smoke test, about a minute), `coef` (GPU vs CPU coefficients), `adaptive`, `thin` (thin-film checks
  only) or `models` (the smoke test of every model). The log shows how long each section took.

A new or changed model needs `npm run check` and, if you changed its nonlinearity or symbols,
`solver:<its id>`. Changes to parameters, UI or docs only need `npm run check`; the model smoke tests in
`tests/models.test.ts` cover every registered model automatically. Wrong numbers in a GPU kernel
often still give a plausible picture, so do run the matching page after touching a kernel.

SwiftShader timings are meaningless. Real-GPU checks (speed, the live app on a canvas) cannot be
done headlessly: say so instead of claiming them verified. On Linux, real WebGPU in Chrome needs
`--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native --ignore-gpu-blocklist`.

## Workflow

- Commit in with conventional commit messages (`feat(solver): ...`, `fix:`, `test:`,
  `docs:`, `refactor:`, `ci:`).
- Larger work should be done on a feature branch and have a pull request; smaller things can go directly to `main`.
- Do not leave dev servers or headless Chrome processes running. `scripts/browser-tests.mjs`
  cleans up after itself; stop anything else you start, by port or pid file, not with `pkill -f`.
- Adding an equation: see "Adding a model" in the README. `defineModel` validates the model and
  the generic tests cover it, so a new model needs no test code of its own.
