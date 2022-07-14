from typing import Any, Optional, Dict
from collections import deque, OrderedDict

import dm_env
import numpy as np
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums


class Wrapper(dm_env.Environment):
    """This allows to modify attributes which agent observes and to pack it back."""
    def __init__(self, env: dm_env.Environment):
        self.env = env

    @staticmethod
    def observation(timestep: dm_env.TimeStep) -> Any:
        return timestep.observation

    @staticmethod
    def reward(timestep: dm_env.TimeStep) -> float:
        return timestep.reward

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        return timestep._replace(
            reward=self.reward(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self) -> dm_env.specs.Array:
        return self.env.action_spec()

    def observation_spec(self) -> dm_env.specs.Array:
        return self.env.observation_spec()


class ActionRepeat(Wrapper):
    """Repeat the same action multiple times."""
    def __init__(self, env, frames_number: int):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        rew_sum = 0.
        for _ in range(self.fn):
            timestep = self.env.step(action)
            rew_sum += timestep.reward
            if timestep.last():
                break
        return timestep._replace(reward=rew_sum)


class FrameStack(Wrapper):
    """Stack previous observations to form a richer state."""
    def __init__(self, env, frames_number: int = 1):
        super().__init__(env)
        self.fn = frames_number
        self._state = None

    def reset(self):
        self._state = None
        return super().reset()

    def observation(self, timestep):
        if self._state is None:
            self._state = deque(self.fn * [timestep.observation], maxlen=self.fn)
        else:
            self._state.append(timestep.observation)
        state = OrderedDict()
        for key in self._state[0].keys():
            state[key] = np.stack(list(map(lambda obs: obs.get(key), self._state)))
        return state

    def observation_spec(self):
        new_spec = OrderedDict()
        for name, spec in self.env.observation_spec().items():
            new_spec[name] = spec.replace(shape=(self.fn, *spec.shape),
                                          name=f'{self.fn}_stacked_{spec.name}')
        return new_spec


#TODO: redo and make understandable
class PointCloudWrapper(Wrapper):
    """Creates point cloud from depth map using MuJoCo engine."""
    def __init__(self,
                 env: dm_env.Environment,
                 pn_number: int = 1000,
                 render_kwargs: Optional[Dict] = None,
                 downsample: int = 1
                 ):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=240, width=320)
        assert all(map(lambda k: k in self.render_kwargs, ('camera_id', 'height', 'width')))
        self.pn_number = pn_number

        self.scene_option = wrapper.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0

        self.downsample = downsample

    def observation(self, timestamp):
        depth_map = self.env.physics.render(depth=True,
                                            scene_option=self.scene_option,
                                            **self.render_kwargs)
        point_cloud = self._get_point_cloud(depth_map)

        point_cloud = np.reshape(point_cloud, (-1, 3))
        segmentation_mask = self._segmentation_mask()
        mask = self._mask(point_cloud)  # additional mask if needed
        selected_points = point_cloud[segmentation_mask & mask][::self.downsample]
        return self._to_fixed_number(selected_points).astype(np.float32)

    def inverse_matrix(self):
        # one could reuse the matrix if a camera remains static
        cam_id, height, width = map(self.render_kwargs.get, ('camera_id', 'height', 'width'))
        fov = self.env.physics.model.cam_fovy[cam_id]
        rotation = self.env.physics.data.cam_xmat[cam_id].reshape(3, 3)
        cx = (width - 1)/2.
        cy = (height - 1)/2.
        f_inv = 2.*np.tan(np.deg2rad(fov)/2.)/height
        inv_mat = np.array([
            [f_inv, 0, -cx*f_inv],
            [0, f_inv, -f_inv*cy],
            [0, 0, 1.]
        ])
        return rotation.T@inv_mat

    def _segmentation_mask(self):
        seg = self.env.physics.render(segmentation=True,
                                      scene_option=self.scene_option,
                                      **self.render_kwargs)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        n = len(pc)
        if n == 0:
            pc = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pc = np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            pc = np.random.permutation(pc)[:self.pn_number]
        return pc

    def _get_point_cloud(self, depth_map):
        inv_mat = self.inverse_matrix()
        width = self.render_kwargs['width']
        height = self.render_kwargs['height']
        grid = 1. + np.mgrid[:height, :width]
        grid = np.concatenate((grid, depth_map[np.newaxis]), axis=0)
        return np.einsum('ij, jhw -> hwi', inv_mat, grid)

    def _mask(self, point_cloud):
        return point_cloud[..., 2] < 10.

    def observation_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')


#TODO: redo awful code
class PointCloudWrapperV2(Wrapper):
    def __init__(self,
                 env,
                 pn_number: int = 1000,
                 render_kwargs=None,
                 append_rgb=False,
                 stride: int = -1):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        assert all(map(lambda k: k in self.render_kwargs, ('camera_id', 'height', 'width')))

        self._grid = 1. + np.mgrid[:self.render_kwargs['height'], :self.render_kwargs['width']]

        self.stride = stride
        self.pn_number = pn_number
        self.append_rgb = append_rgb
        self._selected_geoms = np.array(self._segment_by_name(
            env.physics, ('ground', 'wall', 'floor'), **self.render_kwargs
        ))

    def observation(self, timestep):
        depth = self.env.physics.render(depth=True, **self.render_kwargs)
        pcd = self._point_cloud_from_depth(depth)
        mask = self._mask(pcd)

        if self.append_rgb:
            rgb = self._get_colours()
            pcd = np.concatenate((pcd, rgb), axis=1)

        pcd = self._downsampling(pcd[mask])
        return self._to_fixed_number(pcd).astype(np.float32)

    def _point_cloud_from_depth(self, depth):
        f_inv, cx, cy = self._inverse_intrinsic_matrix_params()
        x, y = (depth * self._grid)
        x = (x - cx) * f_inv
        y = (y - cy) * f_inv

        pc = np.stack((x, y, depth), axis=-1)
        return pc.reshape(-1, 3)
        # rot_mat = self.env.physics.data.cam_xmat[self.render_kwargs['camera_id']].reshape(3, 3)
        # return np.einsum('ij, hwi->hwj', rot_mat, pc).reshape(-1, 3)

    def _to_fixed_number(self, pc):
        n = pc.shape[0]
        if n == 0:
            pc = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pc = np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            pc = np.random.permutation(pc)[:self.pn_number]
        return pc

    def _inverse_intrinsic_matrix_params(self):
        height = self.render_kwargs['height']
        cx = (height - 1) / 2.
        cy = (self.render_kwargs['width'] - 1) / 2.
        fov = self.env.physics.model.cam_fovy[self.render_kwargs['camera_id']]
        f_inv = 2 * np.tan(np.deg2rad(fov) / 2.) / height
        return f_inv, cx, cy

    def _mask(self, point_cloud):
        seg = self.env.physics.render(segmentation=True, **self.render_kwargs)
        segmentation = np.isin(seg[..., 0].flatten(), self._selected_geoms)
        truncate = point_cloud[..., 2] < 10.
        return np.logical_and(segmentation, truncate)

    def observation_spec(self):
        return dm_env.specs.Array(shape=(self.pn_number, 3 + 3*self.append_rgb),
                                  dtype=np.float32,
                                  name='point_cloud' + '+rgb'*self.append_rgb)

    @staticmethod
    def _segment_by_name(physics, bad_geoms_names, **render_kwargs):
        geom_ids = physics.render(segmentation=True, **render_kwargs)[..., 0]

        def _predicate(geom_id):
            if geom_id == -1:  # infinity
                return False
            return all(
                map(
                    lambda name: name not in physics.model.id2name(geom_id, 'geom'),
                    bad_geoms_names
                )
            )

        return list(filter(_predicate, np.unique(geom_ids).tolist()))

    def _downsampling(self, pcd):
        if self.stride < 0:
            adaptive_stride = pcd.shape[0] // self.pn_number
            return pcd[::max(adaptive_stride, 1)]
        else:
            return pcd[::self.stride]

    def _get_colours(self):
        rgb = self.env.physics.render(**self.render_kwargs).reshape(3, -1).astype(np.float32)
        rgb /= 255.
        return rgb.T


class CheetahWrapper(PointCloudWrapperV2):
    def observation(self, timestep):
        pcd = super().observation(timestep)
        velocity = self.env.physics.velocity().astype(np.float32)
        speed = np.array(self.env.physics.speed())[None].astype(np.float32)
        return OrderedDict(point_cloud=pcd, velocity=velocity, speed=speed)

    def observation_spec(self):
        pc_spec = super().observation_spec()
        velocity_spec = dm_env.specs.Array(
            shape=self.env.physics.velocity().shape,
            dtype=np.float32,
            name='velocity'
        )
        sped_spec = dm_env.specs.Array(
            shape=(1,),
            dtype=np.float32,
            name='speed'
        )
        return OrderedDict(point_cloud=pc_spec, velocity=velocity_spec, speed=sped_spec)


