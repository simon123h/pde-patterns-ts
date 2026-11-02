<script setup lang="ts">
import { computed } from 'vue'
import type { ViewKind } from '../models/types'
import { COLORMAPS, gradientCss } from '../render/colormaps'
import { displayRange, formatTick, type Range } from '../render/range'

// Only this component reads the (frequently updated) automatic range, so that the panels around it
// are not re-rendered with it.
const props = defineProps<{
  colormap: string
  kind: ViewKind
  /** The automatic range as read back from the GPU. */
  auto: Range
  /** Fixed limits, or null for the automatic range. */
  manual: Range | null
}>()

const cyclic = computed(() => props.kind === 'arg')
const limits = computed(() => {
  const diverging = COLORMAPS.find((c) => c.id === props.colormap)?.diverging ?? false
  return displayRange(props.manual ?? props.auto, props.kind, diverging)
})
const labels = computed(() => {
  if (cyclic.value) return ['π', '0', '−π']
  const { lo, hi } = limits.value
  return [formatTick(hi), formatTick((lo + hi) / 2), formatTick(lo)]
})
</script>

<template>
  <div class="colorbar" aria-label="color scale">
    <div class="bar" :style="{ background: gradientCss(colormap, cyclic) }" />
    <div class="ticks">
      <span v-for="(label, i) in labels" :key="i">{{ label }}</span>
    </div>
  </div>
</template>

<style scoped>
.colorbar { flex: none; display: flex; gap: 6px; height: min(60vh, 420px); font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted); }
.bar { flex: none; width: 14px; border-radius: 3px; border: 1px solid #333a4d; }
/* a fixed width, so that labels with more or fewer digits do not move the bar and the field next to it */
.ticks { flex: none; display: flex; flex-direction: column; justify-content: space-between; width: 8ch; overflow: hidden; white-space: nowrap; }
</style>
