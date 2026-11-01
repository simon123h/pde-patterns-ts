import { describe, expect, it } from 'vitest'
import workflow from '../.github/workflows/ci.yml?raw'
import { models } from '../src/models'

// The browser self-tests of the solver run as one CI job per group, so a new equation must be listed
// in the workflow or its checks would silently not run in CI.
describe('CI workflow', () => {
  it.each(models.map((m) => m.id))('runs the browser tests of %s', (id) => {
    expect(workflow).toContain(`- solver:${id}`)
  })

  it('runs the adaptive and coefficient groups', () => {
    expect(workflow).toContain('- solver:adaptive')
    expect(workflow).toContain('- solver:coef')
  })
})
