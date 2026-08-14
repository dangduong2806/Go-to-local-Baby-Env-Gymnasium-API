# BabyAI environment
from typing import Optional
import gymnasium as gym

from .wrappers import (
    NavigationOnlyActionWrapper,
    use_rgb_partial_observation,
)

ENV_ID = "BabyAI-GoToLocal-v0"

def make_env(
        render_mode: Optional[str] = None,
        rgb_partial_obs: bool = True,
        navigation_actions_only: bool = False,
        seed: Optional[int] = None,
):
    # Create the BabyAI GoToLocal environment
    """
    Args:
        render_mode:
            ``None`` for no rendering, ``"human"`` for an interactive window,
            or ``"rgb_array"`` to make ``env.render()`` return an RGB frame.

        rgb_partial_obs:
            If True, convert the symbolic partial image into RGB pixels while
            preserving the complete observation dictionary and mission text.

        seed:
            Optional seed used for the environment's first reset. The action
            space is seeded as well so random action sampling is reproducible.
        
        navigation_actions_only:
            If True, expose only left, right, and forward as Discrete(3).
            The default False preserves MiniGrid's original Discrete(7)
            action space for environment inspection.

    Returns:
        A standard Gymnasium environment.

        ``reset()`` returns:
            observation, info

        ``step(action)`` returns:
            observation, reward, terminated, truncated, info
    """
    valid_render_modes = {None, "human", "rgb_array"}
    if render_mode not in valid_render_modes:
        raise ValueError(
            f"Unsupported render_mode={render_mode!r}. "
            f"Expected one of {valid_render_modes}."
        )

    # gymnasium.make adds Gymnasium's normal API-management wrappers, such as
    # order enforcement and environment checking. We leave those enabled so
    # the returned environment follows the standard Gymnasium API.
    env = gym.make(ENV_ID, render_mode = render_mode)

    if rgb_partial_obs:
        # This built-in MiniGrid wrapper changes only the symbolic image into
        # an RGB rendering of the agent's partial field of view. It keeps the
        # dictionary structure, mission instruction, and direction value.
        env = use_rgb_partial_observation(env=env)

    if navigation_actions_only:
        # Basic GoToLocal navigation requires only turning left, turning right,
        # and moving forward. Restricting the action space prevents PPO from
        # wasting probability on pickup, drop, toggle, and done.
        env = NavigationOnlyActionWrapper(env)

    if seed is not None:
        # Gymnasium seeds the environment through reset(seed=...), not through
        # the deprecated env.seed(...) method.
        #
        # make_env returns only the environment, so this initial observation is
        # intentionally discarded. A caller should still call env.reset() before
        # beginning its episode.
        env.reset(seed=seed)

        # Environment seeding and action-space seeding are separate. This makes
        # env.action_space.sample() reproducible as well.
        env.action_space.seed(seed)

    return env


