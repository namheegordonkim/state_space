from argparse import ArgumentParser

import gym
import numpy as np
import torch
from colorhash import ColorHash
from scipy.interpolate import griddata
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style()
np.set_printoptions(formatter={'float': "{:0.3f}".format})


class EnvFactory:

    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name, render=True)


def main(args):
    policy_path = args.policy_path
    expert = PPO.load(policy_path)

    # Initialize environment for input standardization
    factory = EnvFactory(args.env)
    env = DummyVecEnv([factory.make_env])
    env = VecNormalize.load(args.stats_path, env)
    env.training = False

    states = []
    for i in np.arange(-10, 110):
        for j in np.arange(-3, 3, 0.05):
            states.append([i, j])
    states = np.stack(states)
    states_scaled = env.normalize_obs(states)
    states_tensor = torch.as_tensor(states_scaled).float()

    policy: ActorCriticPolicy = expert.policy.cpu()
    true_actions_tensor, _, _ = policy.forward(states_tensor, deterministic=True)
    features_tensor = policy.features_extractor.forward(states_tensor)
    shared_latents_tensor = policy.mlp_extractor.shared_net.forward(features_tensor)
    policy_latents_tensor_layer1 = policy.mlp_extractor.policy_net[0].forward(shared_latents_tensor)
    policy_latents_tensor_layer1_activated = policy.mlp_extractor.policy_net[1].forward(policy_latents_tensor_layer1)
    policy_latents_tensor_layer2 = policy.mlp_extractor.policy_net[2].forward(policy_latents_tensor_layer1_activated)
    policy_latents_tensor_layer2_activated = policy.mlp_extractor.policy_net[3].forward(policy_latents_tensor_layer2)
    actions_tensor = policy.action_net.forward(policy_latents_tensor_layer2_activated)

    assert actions_tensor.equal(true_actions_tensor)

    binary_embeddings_layer1 = policy_latents_tensor_layer1_activated > 0
    binary_embeddings_layer1 = binary_embeddings_layer1.cpu().detach().numpy()
    binary_embeddings_layer2 = policy_latents_tensor_layer2_activated > 0
    binary_embeddings_layer2 = binary_embeddings_layer2.cpu().detach().numpy()

    binary_embeddings = np.concatenate([binary_embeddings_layer1, binary_embeddings_layer2], axis=1).astype(np.int)
    integer_embeddings = np.packbits(binary_embeddings, axis=1, bitorder="little").reshape(-1)

    # convert raw integer embeddings to 0, 1, 2, 3...
    # fast rendering of state cells via grid interpolation
    grid_x, grid_y = np.mgrid[-10:110:1000j, -3:3:1000j]
    z = griddata((states[:, 0], states[:, 1]), integer_embeddings, (grid_x, grid_y), method='nearest')

    # convert raw integer
    convert_raw_integer_to_colorhash = np.vectorize(lambda x: ColorHash(x).rgb)
    grid_z = np.array(convert_raw_integer_to_colorhash(z)).swapaxes(0, 1).swapaxes(1, 2)

    plt.figure()
    plt.imshow(grid_z, extent=[-10, 110, -3, 3], aspect='auto')
    plt.title("State Space Visualized")
    plt.xlabel("$x$")
    plt.ylabel("$\\dot x$")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", help="Name of the environment as defined in __init__.py somewhere", type=str,
                        required=True)
    parser.add_argument("--policy_path", help="Path to policy zip file, if any. Otherwise compute null actions.",
                        type=str, required=True)
    parser.add_argument("--stats_path", help="Path to policy normalization stats.", type=str, required=True)
    args = parser.parse_args()
    main(args)
