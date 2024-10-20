from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"] | Float[Tensor, "batch view gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"] | Float[Tensor, "batch view gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] | Float[Tensor, "batch view gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"] | Float[Tensor, "batch view gaussian"]
