from dm_control.suite.reacher import Physics, get_model_and_assets, _BIG_TARGET, _DEFAULT_TIME_LIMIT
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards
import collections


class ReacherWORandom(base.Task):
    """A reacher `Task` to reach the target."""

    def __init__(self, init_fn, target_size, random=None):
        """Initialize an instance of `Reacher`.
        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target_size = target_size
        self._init_fn = init_fn
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.named.model.geom_size['target', 0] = self._target_size
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        # Randomize target position
        x, y = self._init_fn()
        physics.named.model.geom_pos['target', 'x'] = x
        physics.named.model.geom_pos['target', 'y'] = y

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['to_target'] = physics.finger_to_target()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))


def fixed_random(init_fn, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = ReacherWORandom(init_fn, target_size=_BIG_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )