import sys

import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

from matplotlib import pyplot as plt

# sys.path.append('/home/jiale/softgym/PyFlex/bindings/build')
import pyflex

SAVEPATH = '/home/jiale/softgym/data/figs/'

def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    fig.savefig(SAVEPATH+f'{1:04d}.png')

def show_point_cloud(pc_obs):
    pc = pc_obs[:20064].reshape(-1, 3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1000, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=128, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=1, help='If to test the depth rendering by showing it')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = True
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['observation_mode'] = 'point_cloud'

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(env.horizon):
        action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        # for debug
        if env_kwargs['observation_mode'] == 'point_cloud':
            show_point_cloud(obs)
        frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth and i==0:
            show_depth()

    # if args.save_video_dir is not None:
    #     save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
    #     save_numpy_as_gif(np.array(frames), save_name)
    #     print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
