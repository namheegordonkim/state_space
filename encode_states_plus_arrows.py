from argparse import ArgumentParser

import sys
import gym
import numpy as np
import torch
from colorhash import ColorHash
from scipy.interpolate import griddata
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from random import sample, uniform
import seaborn as sns
import math

sns.set_style()
np.set_printoptions(formatter={'float': "{:0.3f}".format})


class EnvFactory:

    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name, render=True)

def main(args):
    policy_path = args.policy_path
    env = args.env
    stats_path = args.stats_path
    (z, prev_states, next_states) = calculate_integer_encoding(env, policy_path, stats_path)

    # convert raw integer
    convert_raw_integer_to_colorhash = np.vectorize(lambda x: ColorHash(x).rgb)
    grid_z = np.array(convert_raw_integer_to_colorhash(z)).swapaxes(0, 1).swapaxes(1, 2)
    fig, ax = plt.subplots()
    #plt.figure()
    plt.imshow(grid_z, extent=[-10, 110, -3, 3], aspect='auto')
    plt.title("State Space Visualized")

    sample_ind = range(0, len(prev_states), round(len(prev_states)/1000)) # ~1000 indexes
    n_arrows = len(sample_ind)

    X = np.zeros(n_arrows)
    Y = np.zeros(n_arrows)
    U = np.zeros(n_arrows)
    V = np.zeros(n_arrows)

    ind = 0
    for i in sample_ind:
    	x0 = prev_states[i, 0]
    	y0 = prev_states[i, 1]
    	x1 = next_states[i, 0]
    	y1 = next_states[i, 1]
    	distance = [x1 - x0, y1 - y0]
    	#norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    	dx = distance[0] #/ norm
    	dy = distance[1] #/ norm
    	#plt.arrow(x=x0, y=y0, dx=dx, dy=dy, width=.08)
    	X[ind] = x0
    	Y[ind] = y0
    	U[ind] = dx
    	V[ind] = dy
    	ind = ind + 1

    U = U / np.sqrt(U**2 + V**2);
    V = V / np.sqrt(U**2 + V**2);
    ax.quiver(X,Y,U,V)

    plt.xlabel("$x$")
    plt.ylabel("$\\dot x$")
    plt.show()


def calculate_integer_encoding(env, policy_path, stats_path):
    expert = PPO.load(policy_path)

    # Initialize environment for input standardization
    factory = EnvFactory(env)
    env = DummyVecEnv([factory.make_env])
    env = VecNormalize.load(stats_path, env)
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

    binary_embeddings = np.concatenate([binary_embeddings_layer1, binary_embeddings_layer2], axis=1).astype(int)
    integer_embeddings = np.packbits(binary_embeddings, axis=1, bitorder="little")
    integer_embeddings = integer_embeddings @ (256 ** np.arange(integer_embeddings.shape[1]))  # to allow arbitrary number of bits

    # convert raw integer embeddings to 0, 1, 2, 3...
    # fast rendering of state cells via grid interpolation
    grid_x, grid_y = np.mgrid[-10:110:1000j, -3:3:1000j]
    z = griddata((states[:, 0], states[:, 1]), integer_embeddings, (grid_x, grid_y), method='nearest')

    actions = actions_tensor.detach().numpy()

    combined = np.concatenate((states, actions), axis =1)

    x_pos = combined[:, 0]
    x_vel = combined[:, 1]
    x_acc = combined[:, 2]

    print(x_pos)
    print(x_vel)
    print(x_acc)
    (prev_states, next_states) = calculate_next_states(x_pos, x_vel, x_acc)

    return (z, prev_states, next_states)

def calculate_next_states(x_pos, x_vel, x_acc):
    sim_step = 1 #simulation timestep
    new_x_pos = x_pos + sim_step * x_vel + 0.5 * sim_step ** 2 * x_acc
    new_x_vel = np.clip(new_x_pos, -10, 110)
    new_x_vel = x_vel + sim_step * x_acc
    new_x_vel = np.clip(new_x_vel, -3, 3)
    return (np.stack((x_pos, x_vel), axis=1), np.stack((new_x_pos, new_x_vel), axis=1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", help="Name of the environment as defined in __init__.py somewhere", type=str,
                        required=True)
    parser.add_argument("--policy_path", help="Path to policy zip file, if any. Otherwise compute null actions.",
                        type=str, required=True)
    parser.add_argument("--stats_path", help="Path to policy normalization stats.", type=str, required=True)
    parser.add_argument("--with_sim", help="Simulate also and draw path lines.", type=str, required=False, default='False')
    args = parser.parse_args()
    main(args)
