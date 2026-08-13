# Observation wrappers used to inspect BabyAI environments

import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper

def use_rgb_partial_observation(
        env: gym.Env,
        tile_size: int = 8,

):
    """
        Replace MiniGrid's symbolic image with a rendered RGB agent view.
        MiniGrid's original ``obs["image"]`` is a symbolic partial observation.
        Each visible grid cell is represented by three integer values:

            (object_type, color, state)

        For example, ``state`` describes whether a door is open, closed, or
        locked. These numbers are semantic codes, not RGB pixel intensities.

        ``RGBImgPartialObsWrapper`` renders the same agent-centered, partially
        observable field of view as RGB pixels. It changes ``obs["image"]`` while
        preserving the observation dictionary, including ``obs["mission"]`` and
        ``obs["direction"]``.

        Partial observability is important in vision-language navigation because
        the agent cannot see the whole map at once. It must connect the language
        instruction to its current visual input and gather more information by
        moving through the environment.

        No resizing, normalization, flattening, or language tokenization is
        performed here.
    """
    return RGBImgPartialObsWrapper(env, tile_size=tile_size)
