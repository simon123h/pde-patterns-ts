import { brusselator } from './brusselator'
import { cahnHilliard } from './cahnHilliard'
import { ginzburgLandau } from './ginzburgLandau'
import { grayScott } from './grayScott'
import { kuramotoSivashinsky } from './kuramotoSivashinsky'
import { swiftHohenberg } from './swiftHohenberg'
import type { SpectralModel } from './types'

export const models: SpectralModel[] = [
  swiftHohenberg,
  cahnHilliard,
  kuramotoSivashinsky,
  ginzburgLandau,
  grayScott,
  brusselator,
]
