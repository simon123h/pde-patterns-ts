<script setup lang="ts">
export interface Stats {
  fps: number
  time: number
  dt: number
  accepted: number
  rejected: number
  recoveries: number
}

// Reads the fast-changing numbers in its own component, so that updating them re-renders only
// this text and not the surrounding panel with its native <select> controls.
defineProps<{ stats: Stats; kind: 'summary' | 'adaptive'; adapter?: string }>()
</script>

<template>
  <template v-if="kind === 'summary'">{{ adapter || 'WebGPU' }} · {{ stats.fps.toFixed(0) }} fps · t = {{ stats.time.toFixed(1) }}<template v-if="stats.recoveries"> · dt halved {{ stats.recoveries }}× after blow-up</template></template>
  <template v-else>current dt = {{ stats.dt.toPrecision(3) }} · accepted {{ stats.accepted }} · rejected {{ stats.rejected }}</template>
</template>
