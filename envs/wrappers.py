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

class NavigationOnlyActionWrapper(gym.ActionWrapper):
    """Restrict MiniGrid to the three actions needed for basic VLN.

    Exposed action space:

        0 -> left
        1 -> right
        2 -> forward

    MiniGrid uses these same IDs internally, so the mapping is currently
    identity. Keeping an explicit map makes the restriction clear and avoids
    accidentally allowing pickup, drop, toggle, or done.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._action_map = (
            0,  # left
            1,  # right
            2,  # forward
        )

        self.action_space = gym.spaces.Discrete(
            len(self._action_map)
        )

    def action(self, action):
        """Convert the restricted action into a MiniGrid action ID."""
        restricted_action = int(action)

        if not self.action_space.contains(restricted_action):
            raise ValueError(
                f"Invalid navigation action {restricted_action}. "
                "Expected 0=left, 1=right, or 2=forward."
            )

        return self._action_map[restricted_action]

    def reverse_action(self, action):
        """Convert a MiniGrid action ID back into the restricted ID."""
        action_id = int(action)

        try:
            return self._action_map.index(action_id)
        except ValueError as error:
            raise ValueError(
                f"MiniGrid action {action_id} is not part of the "
                "navigation-only action space."
            ) from error