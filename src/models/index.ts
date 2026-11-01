import { advectionDiffusion } from './advectionDiffusion'
import { brusselator } from './brusselator'
import { cahnHilliard } from './cahnHilliard'
import { ginzburgLandau } from './ginzburgLandau'
import { grayScott } from './grayScott'
import { kuramotoSivashinsky } from './kuramotoSivashinsky'
import { nikolaevskiy } from './nikolaevskiy'
import { swiftHohenberg } from './swiftHohenberg'
import type { SpectralModel } from './types'

export const models: SpectralModel[] = [
  swiftHohenberg,
  cahnHilliard,
  kuramotoSivashinsky,
  nikolaevskiy,
  ginzburgLandau,
  grayScott,
  brusselator,
  advectionDiffusion,
]
