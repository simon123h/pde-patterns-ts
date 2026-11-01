<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { requestGpu } from '../gpu/device'
import { models } from '../models'
import type { SpectralModel } from '../models/types'
import { COLORMAPS } from '../render/colormaps'
import { FieldRenderer } from '../render/fieldRenderer'
import { MAX_STAMPS, SpectralSolver } from '../solver/spectralSolver'

const GRID_SIZES = [64, 128, 256, 512, 1024]

const canvas = ref<HTMLCanvasElement | null>(null)
const error = ref<string | null>(null)
const adapterLabel = ref('')

const model = shallowRef<SpectralModel>(models[0]!)
// `?grid=64` selects the initial grid size (handy on slow software adapters)
const requestedGrid = Number(new URLSearchParams(location.search).get('grid'))
const gridSize = ref(GRID_SIZES.includes(requestedGrid) || requestedGrid === 64 ? requestedGrid : 256)
const dt = ref(model.value.dt.default)
const domain = ref(model.value.domain.default)
const stepsPerFrame = ref(model.value.stepsPerFrame)
const viewIndex = ref(0)
const adaptive = ref(false)
const toleranceExp = ref(-3) // error tolerance = 10^toleranceExp
const maxDt = ref(model.value.dt.max)
const colormap = ref('viridis')
const brushSize = ref(3) // percent of the domain length
const brushStrength = ref(1)
const running = ref(true)
const params = reactive<Record<string, number>>({})
const stats = reactive({ fps: 0, time: 0, dt: 0, accepted: 0, rejected: 0 })
// updated every frame, copied into `stats` a few times per second: re-rendering the whole panel
// at 60 Hz made native controls (the equation dropdown in particular) flicker and close
let liveFps = 0
let liveTime = 0
let liveDt = 0
let liveAccepted = 0
let liveRejected = 0
let statsTimer = 0

let device: GPUDevice | null = null
let solver: SpectralSolver | null = null
let renderer: FieldRenderer | null = null
let raf = 0
let lastFrame = 0
// brush stamps queued by pointer events, applied at the start of the next frame
let pendingStamps: number[] = []
let lastPaint: { x: number; y: number } | null = null
let inFlight = 0 // submitted frames the GPU hasn't finished; limits queue build-up on slow adapters

function loadModelDefaults() {
  const m = model.value
  for (const key of Object.keys(params)) delete params[key]
  for (const p of m.params) params[p.key] = p.default
  dt.value = m.dt.default
  maxDt.value = m.dt.max
  domain.value = m.domain.default
  stepsPerFrame.value = m.stepsPerFrame
  viewIndex.value = 0
  brushSize.value = (m.paint?.sigma ?? 0.03) * 100
}

/** Applies a preset of the current model; settings it leaves out keep their value. */
function applyPreset(index: number) {
  const preset = model.value.presets?.[index]
  if (!preset) return
  Object.assign(params, preset.params)
  if (preset.dt !== undefined) dt.value = preset.dt
  if (preset.domain !== undefined) domain.value = preset.domain
}

function selectModel(id: string) {
  model.value = models.find((m) => m.id === id)!
  loadModelDefaults()
  build()
}

function build() {
  if (!device || !canvas.value) return
  solver?.destroy()
  renderer?.destroy()
  const grid = { nx: gridSize.value, ny: gridSize.value, dx: domain.value / gridSize.value }
  solver = new SpectralSolver(device, model.value, grid)
  solver.setParams({ ...params })
  solver.setDt(dt.value)
  solver.setDomain(domain.value)
  applyAdaptive()
  solver.reset()
  renderer = new FieldRenderer(device, canvas.value, solver.physical, grid.nx, grid.ny, solver.real)
}

function applyAdaptive() {
  solver?.setAdaptive({
    enabled: adaptive.value,
    tol: 10 ** toleranceExp.value,
    hmax: maxDt.value,
    hmin: model.value.dt.min * 0.01,
  })
}

function resetSimulation() {
  solver?.reset()
  renderer?.resetRange()
}

/** Domain coordinates of a pointer event (y grows downward, like the rendered field). */
function domainPoint(e: PointerEvent) {
  const rect = canvas.value!.getBoundingClientRect()
  const length = solver!.domain
  return { x: ((e.clientX - rect.left) / rect.width) * length, y: ((e.clientY - rect.top) / rect.height) * length }
}

function queueStamps(e: PointerEvent) {
  if (!solver) return
  const to = domainPoint(e)
  const sign = (e.buttons & 2 ? -1 : 1) * brushStrength.value
  const radius = (brushSize.value / 100) * solver.domain
  const from = lastPaint ?? to
  // stamp along the stroke at a fraction of the brush radius so fast strokes stay continuous
  const distance = Math.hypot(to.x - from.x, to.y - from.y)
  const n = Math.max(1, Math.ceil(distance / (0.4 * radius)))
  for (let i = 1; i <= n; i++) {
    const t = i / n
    pendingStamps.push(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, sign, 0)
  }
  lastPaint = to
}

function onPointerDown(e: PointerEvent) {
  if (e.button > 2) return
  canvas.value?.setPointerCapture(e.pointerId)
  lastPaint = null
  queueStamps(e)
}

function onPointerMove(e: PointerEvent) {
  if (e.buttons & 3) queueStamps(e)
}

function onPointerUp() {
  lastPaint = null
}

function resizeCanvas() {
  const el = canvas.value
  if (!el) return
  const dpr = window.devicePixelRatio || 1
  const size = Math.max(64, Math.floor(el.clientWidth * dpr))
  if (el.width !== size || el.height !== size) {
    el.width = size
    el.height = size
  }
}

function frame(now: number) {
  raf = requestAnimationFrame(frame)
  if (!device || !solver || !renderer || inFlight >= 2) return
  resizeCanvas()
  const enc = device.createCommandEncoder()
  if (pendingStamps.length) {
    solver.paint(enc, new Float32Array(pendingStamps.slice(0, MAX_STAMPS * 4)), (brushSize.value / 100) * solver.domain)
    pendingStamps = pendingStamps.slice(MAX_STAMPS * 4)
  }
  if (running.value) solver.step(enc, stepsPerFrame.value)
  solver.encodeView(enc)
  const view = model.value.views[viewIndex.value] ?? model.value.views[0]!
  renderer.render(enc, view.field, view.kind, colormap.value)
  solver.encodeStatsCopy(enc)
  device.queue.submit([enc.finish()])
  void solver.afterSubmit()
  inFlight++
  void device.queue.onSubmittedWorkDone().then(() => inFlight--)

  if (lastFrame) liveFps = liveFps * 0.9 + (1000 / (now - lastFrame)) * 0.1
  lastFrame = now
  liveTime = solver.time
  liveDt = solver.dt
  liveAccepted = solver.stats.accepted
  liveRejected = solver.stats.rejected
}

onMounted(async () => {
  loadModelDefaults()
  try {
    const gpu = await requestGpu()
    device = gpu.device
    const info = gpu.adapter.info
    adapterLabel.value = `${info.vendor} ${info.architecture}`.trim()
    device.lost.then((info) => (error.value = `GPU device lost: ${info.message}`))
    build()
    raf = requestAnimationFrame(frame)
    statsTimer = window.setInterval(() => {
      stats.fps = liveFps
      stats.time = liveTime
      stats.dt = liveDt
      stats.accepted = liveAccepted
      stats.rejected = liveRejected
    }, 250)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  clearInterval(statsTimer)
  solver?.destroy()
  renderer?.destroy()
})

watch(gridSize, build)
watch(dt, (v) => solver?.setDt(v))
watch(domain, (v) => solver?.setDomain(v))
watch([adaptive, toleranceExp, maxDt], applyAdaptive)
watch(params, () => solver?.setParams({ ...params }), { deep: true })
// parameters that only shape the initial condition restart the run
watch(
  () => model.value.params.filter((p) => p.resetOnChange).map((p) => params[p.key]),
  () => resetSimulation(),
)
</script>

<template>
  <div class="layout">
    <div class="stage"><canvas
        ref="canvas"
        class="view"
        @pointerdown.prevent="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointercancel="onPointerUp"
        @contextmenu.prevent
      /></div>
    <aside class="panel">
      <h1>PDE Patterns</h1>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-else class="muted">{{ adapterLabel || 'WebGPU' }} · {{ stats.fps.toFixed(0) }} fps · t = {{ stats.time.toFixed(1) }}</p>

      <label>
        <span>equation</span>
        <select
          :value="model.id"
          @change="(e) => { selectModel((e.target as HTMLSelectElement).value); (e.target as HTMLSelectElement).blur() }"
        >
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name }}</option>
        </select>
      </label>
      <p class="muted desc">{{ model.description }}</p>
      <label v-if="model.presets?.length">
        <span>preset</span>
        <select
          :key="model.id"
          @change="(e) => { const t = e.target as HTMLSelectElement; applyPreset(t.selectedIndex - 1); t.selectedIndex = 0; t.blur() }"
        >
          <option disabled selected>choose…</option>
          <option v-for="(preset, i) in model.presets" :key="i" :title="preset.description">{{ preset.name }}</option>
        </select>
      </label>

      <h2>Parameters</h2>
      <label v-for="p in model.params" :key="p.key">
        <span>{{ p.label }} <output>{{ params[p.key]?.toFixed(2) }}</output></span>
        <input v-model.number="params[p.key]" type="range" :min="p.min" :max="p.max" :step="p.step" />
      </label>

      <h2>Domain &amp; numerics</h2>
      <label>
        <span>domain size L <output>{{ domain.toFixed(model.domain.step < 1 ? 1 : 0) }}</output></span>
        <input v-model.number="domain" type="range" :min="model.domain.min" :max="model.domain.max" :step="model.domain.step" />
      </label>
      <label>
        <span>dt <output>{{ dt.toFixed(2) }}</output></span>
        <input v-model.number="dt" type="range" :min="model.dt.min" :max="model.dt.max" :step="model.dt.step" />
      </label>
      <label class="check">
        <input v-model="adaptive" type="checkbox" />
        <span>adaptive dt (dt above is the starting value)</span>
      </label>
      <template v-if="adaptive">
        <label>
          <span>error tolerance <output>1e{{ toleranceExp.toFixed(1) }}</output></span>
          <input v-model.number="toleranceExp" type="range" min="-6" max="-1" step="0.1" />
        </label>
        <label>
          <span>max dt <output>{{ maxDt.toFixed(2) }}</output></span>
          <input v-model.number="maxDt" type="range" :min="model.dt.min" :max="model.dt.max * 20" :step="model.dt.step" />
        </label>
        <p class="muted desc">
          current dt = {{ stats.dt.toPrecision(3) }} · accepted {{ stats.accepted }} · rejected {{ stats.rejected }}
        </p>
      </template>
      <label>
        <span>steps / frame <output>{{ stepsPerFrame }}</output></span>
        <input v-model.number="stepsPerFrame" type="range" min="1" max="20" step="1" />
      </label>
      <label>
        <span>grid</span>
        <select v-model.number="gridSize">
          <option v-for="n in GRID_SIZES" :key="n" :value="n">{{ n }} × {{ n }}</option>
        </select>
        <small class="muted">dx = {{ (domain / gridSize).toPrecision(3) }}</small>
      </label>
      <label v-if="model.views.length > 1">
        <span>view</span>
        <select v-model.number="viewIndex">
          <option v-for="(v, i) in model.views" :key="i" :value="i">{{ v.label }}</option>
        </select>
      </label>
      <label>
        <span>colormap</span>
        <select v-model="colormap">
          <option v-for="c in COLORMAPS" :key="c.id" :value="c.id">{{ c.label }}</option>
        </select>
      </label>

      <h2>Brush</h2>
      <label>
        <span>size <output>{{ brushSize.toFixed(1) }} % of L</output></span>
        <input v-model.number="brushSize" type="range" min="0.5" max="15" step="0.5" />
      </label>
      <label>
        <span>strength <output>{{ brushStrength.toFixed(1) }}×</output></span>
        <input v-model.number="brushStrength" type="range" min="0.1" max="5" step="0.1" />
      </label>
      <p class="muted desc">Drag on the field: left button adds, right button subtracts.</p>

      <div class="buttons">
        <button @click="running = !running">{{ running ? 'Pause' : 'Run' }}</button>
        <button @click="resetSimulation">Reset</button>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.layout { display: flex; height: 100vh; }
.stage { flex: 1; min-width: 0; display: grid; place-items: center; background: #000; }
.view { width: min(100%, 100vh); aspect-ratio: 1; cursor: crosshair; touch-action: none; }
.panel { width: 280px; padding: 16px; background: var(--panel); overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 13px; margin: 12px 0 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
label { display: flex; flex-direction: column; gap: 4px; }
label span { display: flex; justify-content: space-between; }
label.check { flex-direction: row; align-items: center; gap: 8px; }
output { color: var(--muted); font-variant-numeric: tabular-nums; }
select, button { background: #232734; color: var(--fg); border: 1px solid #333a4d; border-radius: 6px; padding: 6px 10px; }
button { cursor: pointer; flex: 1; }
button:hover { border-color: var(--accent); }
.buttons { display: flex; gap: 8px; margin-top: 8px; }
.muted { color: var(--muted); margin: 0; }
.desc { font-size: 12px; }
.error { color: #ff7b7b; margin: 0; }
</style>
