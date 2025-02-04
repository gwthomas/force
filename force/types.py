from typing import Callable

from torch import Tensor


type Observation = Tensor
type Action = Tensor


# The most general form of policy: a function that maps observation -> action,
# possibly stochastically.
type PolicyFunction = Callable[[Observation], Action]