import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object

import math

## not good for scaling the reward
# def normpdf(x, mean, sd):
#     var = float(sd)**2
#     denom = (2*math.pi*var)**.5
#     num = math.exp(-(float(x)-float(mean))**2/(2*var))
#     return num/denom

class RopeFlattenEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            config = self.get_default_config()
        default_config = config            
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        # set reward range
        # self.reward_max = 0
        # self.reward_min = -self.rope_length
        # self.reward_range = self.reward_max - self.reward_min

        self.reward_max = self.horizon*self.rope_length
        self.reward_min = 0
        self.reward_range = self.reward_max - self.reward_min

        # save intial key point positions

        self.last_key_point_pos = self.get_key_point_vec()

        return self._get_obs()
    
    def get_key_point_vec(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        return keypoint_pos # [10,3]

    def _step(self, action):

        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""

        # original reward function
        # curr_endpoint_dist = self._get_endpoint_distance()
        # curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        # r = curr_distance_diff
        # return r

        # This original may be problematic, as it does not encourage the right action to be taken nor exploration;
        # Once the rope is at a reasonably large length, the agent will leave it there and not touch it,
        # as in such way the reward is already pretty good.

        # In the new reward function, any action will be rewarded as long as the rope moves. The reward is the current distance
        # between the endpoints of the rope.

        current_key_point_pos = self.get_key_point_vec()
        keypoint_displacement = np.sum(np.linalg.norm(current_key_point_pos - self.last_key_point_pos, axis=1))
        self.last_key_point_pos = current_key_point_pos

        d = self._get_endpoint_distance()

        if keypoint_displacement < 0.01: # the rope is not moving
            if d > 0.95*self.rope_length: # if the rope is almost fully stretched, give a reward
                r = d
            else: # if the rope is not stretched, give no reward
                r = 0
        else:
            r = d

        com_pos = np.linalg.norm(np.mean(current_key_point_pos, axis=0))
        coeff = 1 - (np.clip(com_pos, 0.5, 1.0)-0.5)/0.5
            
        return coeff*r

    def _get_info(self):
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)

        performance = curr_distance_diff
        normalized_performance = (performance - self.reward_min) / self.reward_range

        return {
            'performance': performance,
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist
        }
