import os
from argparse import ArgumentParser
from shutil import copyfile

import gym
import numpy as np
import torch
from tqdm import tqdm

import wandb
from gym import Env
from stable_baselines3 import PPO
from scipy.interpolate import griddata
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from torch import nn
import math

from colorhash import ColorHash
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': "{:0.3f}".format})
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class EnvFactory:
    """
    Factory pattern helps use parallel processing support more elegantly
    """

    def __init__(self, env_name):
        self.env_name = env_name

    def make_env(self):
        return gym.make(self.env_name, render=False)


def get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows):
    """
    rewards and dones are 2d numpy arrays.
    each row corresponds to a process running the environment.
    each column corresponds to a timestep.
    for each row, we accumulate up to the first case where done=True
    """
    cumulative_rewards = []
    for reward_row, done_row in zip(reward_rows, done_rows):
        cumulative_reward = 0
        for reward, done in zip(reward_row, done_row):
            cumulative_reward += reward
            if done:
                break
        cumulative_rewards.append(cumulative_reward)
    return np.array(cumulative_rewards)


class UCRCallback(BaseCallback):
    """
    a callback for calculating encoding and Used Cell Ratio

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, envs: VecNormalize,  eval_every: int, env_name: str, verbose=0):
        #super(UCRCallback, self).__init__(verbose)
        self._called = 0
        self.envs = envs
        self.eval_every = eval_every
        self.env_name = env_name
        self.means = {}
        self.variances = {}
        super().__init__(verbose)

    def _on_step(self):
        pass

    def _on_rollout_start(self) -> None:
        if self.n_calls % self.eval_every == 0:
            print("starting")
            expert = self.model


            factory = EnvFactory(self.env_name)

            #policy_path = "checkpoints/car1d_scratch/best.zip"
            stats_path = "checkpoints/car1d_scratch/best_stats.pth"

            env = DummyVecEnv([factory.make_env]) # for rendering
            #env = VecNormalize(stats_path, env)
            #env = DummyVecEnv([factory.make_env])
            env = VecNormalize.load(stats_path, env)

            env.training = False

            states = []
            for i in np.arange(-10, 110):
                for j in np.arange(-3, 3, 0.05):
                    states.append([i, j])
            states = np.stack(states)
            #print(states.shape)
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


            grid_x, grid_y = np.mgrid[-10:110:1000j, -3:3:1000j]
            z = griddata((states[:, 0], states[:, 1]), integer_embeddings, (grid_x, grid_y), method='nearest')

            # convert raw integer
            convert_raw_integer_to_colorhash = np.vectorize(lambda x: ColorHash(x).rgb)
            grid_z = np.array(convert_raw_integer_to_colorhash(z)).swapaxes(0, 1).swapaxes(1, 2)

            actions = actions_tensor.detach().numpy()
            #print(actions)
            #print(actions.shape)
            ie = np.reshape(integer_embeddings, (-1, 1))
            #print(integer_embeddings)
            #print(ie)
            #print(ie.shape)
            combined = np.concatenate((ie, actions), axis =1)
            #print(combined)
            #print(states.shape)
            #print(states)

            obs = env.reset()
            done = False
            visited_states = []
            while not done:
                action, state = expert.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                #print(obs)
                x_obs = round(obs[0][0])
                y_obs = round(obs[0][1])
                x_obs = np.clip(x_obs, -10, 110)
                y_obs = np.clip(y_obs, -100, 100)
                for i in range(len(states)):
                    #print(states[i])
                    if(states[i][0] == x_obs and states[i][1] == y_obs):
                        #print(x_obs)
                        #print(states[i][0])
                        #print(y_obs)
                        #print(states[i][1])
                        #print(integer_embeddings[i])
                        visited_states.append(integer_embeddings[i])
                        break    # break here
                    elif(i == range(len(states) - 1)):
                        print("FAILED TO FIND CELL FOR STATE" + str(obs))
            visited_states = np.array(visited_states)
            count = len(np.unique(visited_states))
            cells, cell_visits = np.unique(visited_states, return_counts=True)
            countall = len(np.unique(integer_embeddings))
            print("UCR: " + str(count) + "/" + str(countall))
            #fig, ax = plt.subplots()
            plt.figure()
            plt.imshow(grid_z, extent=[-10, 110, -3, 3], aspect='auto')
            plt.title("State Space Visualized")
            plt.xlabel("$x$")
            plt.ylabel("$\\dot x$")


            patches = []
            for i in np.unique(combined[:,0]):
                #print('trying to find: ' + str(i))
                ind =  np.where(cells == i)
                #print(ind)
                if(len(ind[0]) > 0):
                    ind = ind[0][0]
                    visits = cell_visits[ind]
                else:
                    visits = 0

                indarr = np.where(combined[:,0] == i) #list of all found occurance indexes
                cellcount = len(indarr[0])
                #print(cellcount)
                #print(indarr[0])
                coords = states[indarr[0]] #here are the corresponding coordinates
                #print(coords)
                centroid = np.mean(coords, axis = 0) # calculate centroid as the mean of coordinates
                #print(centroid)
                #print(ColorHash(i).rgb)
                #ax.annotate(str(i), (centroid[0], centroid[1]))
                color = tuple([z / 255 for z in ColorHash(i).rgb])
                #print(color)

                txt = str(i) + ' (' + str(round(centroid[0], 2)) + ',' + str(round(centroid[1], 2)) + ')' + ', ' + str(cellcount)
                patches.append( mpatches.Patch(label=txt, color = color))

                tmp = combined[indarr[0]]
                #print(tmp)
                if i in self.means:
                    arr = self.means[i].copy()
                    arr.append((self.n_calls, centroid[0], centroid[1], cellcount, np.mean(tmp[:,1]), np.var(tmp[:,1]), visits)) #calculate mean  and variance of actions
                    #print(np.mean(tmp[:,1]))
                    self.means[i] = arr
                else:
                    self.means[i] = [(self.n_calls, centroid[0], centroid[1], cellcount, np.mean(tmp[:,1]), np.var(tmp[:,1]), visits)] #n_calls, cell_id, x, y, cell_count, mean, variance
                    #print(np.mean(tmp[:,1]))

            plt.legend(handles=patches, loc="lower center")
            #plt.show()
            figname = 'images/' + str(self.n_calls) + '.png'
            plt.savefig(figname)

            with open("ucrresults.csv", "a") as ucrfile:
                ucrfile.write(str(self.n_calls) + "," + str(countall) + "," + str(count) + "\n")

            with open("statsresults.csv", "w") as statsfile:
                statsfile.write("n_calls" + "," + "cell_id" + "," + "x" + "," + "y" + "," + "cell_count" + "," + "mean" + "," + "variance" + "," + "visits" + "\n")
                for key in self.means:
                    #print('test')
                    #print(key)
                    for res in self.means[key]:
                        input = str(key)
                        for item in res:
                            input = input + "," + str(item)
                        input = input + "\n"
                        statsfile.write(input)




class WAndBEvalCallback(BaseCallback):

    def __init__(self, render_env: Env, eval_every: int, envs: VecNormalize, verbose=0):
        self.render_env = render_env  # if render with rgb_array is implemented, use this to collect images
        self.eval_every = eval_every
        self.best_cumulative_rewards_mean = -np.inf
        self.envs = envs
        super().__init__(verbose)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        run_dir = os.path.join("checkpoints", "{:s}_{:s}".format(args.project_name, args.run_name))
        os.makedirs(run_dir, exist_ok=True)

        # save policy weights
        self.model.save(os.path.join(wandb.run.dir, "latest.zip"))
        self.model.save(os.path.join(run_dir, "latest.zip".format(args.project_name, args.run_name)))
        copyfile(os.path.join(run_dir, "latest.zip".format(args.project_name, args.run_name)),
                 os.path.join(run_dir, "second_latest.zip".format(args.project_name, args.run_name)))

        # save stats for normalization
        self.envs.save(os.path.join(wandb.run.dir, "latest_stats.pth"))
        stats_path = os.path.join(run_dir, "latest_stats.pth")
        self.envs.save(stats_path)
        copyfile(os.path.join(run_dir, "latest_stats.pth".format(args.project_name, args.run_name)),
                 os.path.join(run_dir, "second_latest_stats.pth".format(args.project_name, args.run_name)))

        metrics = {"n_calls": self.n_calls}
        if self.n_calls % self.eval_every == 0:
            obs_column = self.envs.reset()
            reward_columns = []
            done_columns = []
            actions = []
            # We can optionally gather images and render as video
            # images = []
            self.envs.training = False
            for i in range(1000):
                action_column, states = self.model.predict(obs_column, deterministic=True)
                next_obs_column, old_reward_column, done_column, info = self.envs.step(action_column)
                for a in action_column:
                    actions.append(a)
                reward_column = self.envs.get_original_reward()
                reward_columns.append(reward_column)
                done_columns.append(done_column)
                obs_column = next_obs_column

            self.envs.training = True
            reward_rows = np.stack(reward_columns).transpose()
            done_rows = np.stack(done_columns).transpose()
            cumulative_rewards = get_cumulative_rewards_from_vecenv_results(reward_rows, done_rows)
            cumulative_rewards_mean = np.mean(cumulative_rewards)
            # Also can compute standard deviation of rewards across different inits
            # cumulative_rewards_std = np.std(cumulative_rewards)
            # Uploads images as video
            # images = np.stack(images)
            # metrics.update({"video": wandb.Video(images, fps=24, format="mp4")})

            # Can also do other things like upload plots, etc.

            if cumulative_rewards_mean > self.best_cumulative_rewards_mean:
                self.best_cumulative_rewards_mean = cumulative_rewards_mean
                self.model.save(os.path.join(wandb.run.dir, "best.zip"))
                self.model.save(os.path.join(run_dir, "best.zip"))

                self.envs.save(os.path.join(wandb.run.dir, "best_stats.pth"))
                self.envs.save(os.path.join(run_dir, "best_stats.pth"))

            metrics.update({"cumulative_rewards_mean": cumulative_rewards_mean})

        wandb.log(metrics)


def main(args):
    wandb.init(project=args.project_name, name=args.run_name)
    n_envs = len(os.sched_getaffinity(0))
    factory = EnvFactory(args.env)

    # Wrap the
    render_env = factory.make_env()  # for rendering

    callback = CallbackList([])

    # Wrap the environment around parallel processing friendly wrapper, unless debug is on
    if args.debug:
        envs = DummyVecEnv([factory.make_env for _ in range(n_envs)])
    else:
        envs = SubprocVecEnv([factory.make_env for _ in range(n_envs)])

    if args.stats_path is None:
        envs = VecNormalize(envs, norm_obs=True, clip_obs=np.inf, norm_reward=False, clip_reward=np.inf)
    else:
        envs = VecNormalize.load(args.stats_path, envs)
    eval_callback = WAndBEvalCallback(render_env, args.eval_every, envs)
    callback.callbacks.append(eval_callback)


    # Calculate UCR callback
    ucr_callback = UCRCallback(envs, args.eval_every, args.env)
    #ucr_callbackNTimesteps = EveryNTimesteps(n_steps=10, callback=ucr_callback)
    callback.callbacks.append(ucr_callback)
    #make csv out of results
    with open("ucrresults.csv", "w") as ucrfile:
        ucrfile.write("n calls" + "," + "cells total" + "," + "used cells" + "," + "mean" + "," + "variance" + "\n")

    with open("statsresults.csv", "w") as statfile:
        statfile.write("n_calls" + "," + "cell_id" + "," + "x" + "," + "y" + "," + "cell_count" + "," + "mean" + "," + "variance" + "\n")

    print("Do random explorations to build running averages")
    envs.reset()
    for _ in tqdm(range(1000)):
        random_action = np.stack([envs.action_space.sample() for _ in range(n_envs)])
        envs.step(random_action)
    envs.training = False  # freeze the running averages (what a terrible variable name...)

    # We use PPO by default, but it should be easy to swap out for other algorithms.
    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path
        learner = PPO.load(pretrained_path, envs, device=args.device)
        learner.learn(total_timesteps=args.total_timesteps, callback=callback)
    else:
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(
                vf=args.value_dims,
                pi=args.policy_dims
            )
            ],
            log_std_init=args.log_std_init,
            squash_output=False

        )

        learner = PPO(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, policy_kwargs=policy_kwargs, device=args.device, target_kl=2e-2)
        if args.device == 'cpu':
            torch.cuda.empty_cache()
        learner.learn(total_timesteps=args.total_timesteps, callback=callback)

    render_env.close()
    envs.close()

    #plot ucr results
    ucrresults = np.genfromtxt("ucrresults.csv", delimiter=',')
    #print(ucrresults)
    ucrresults = np.delete(ucrresults, (0), axis=0) #drop first row
    #print(ucrresults[:,0])
    #plt.plot(ucrresults[:,0], ucrresults[:,1], label = "total cells")
    #plt.plot(ucrresults[:,0], ucrresults[:,2], label = "used cells")
    plt.plot(ucrresults[:,0], ucrresults[:,3], label = "mean")
    plt.plot(ucrresults[:,0], ucrresults[:,4], label = "variance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--run_name", help="Weights & Biases run name", required=True, type=str)
    parser.add_argument("--env", help="Name of the environment as registered in __init__.py somewhere", required=True,
                        type=str)
    parser.add_argument("--n_steps", help="Number of timesteps in each rollouts when training with PPO", required=True,
                        type=int)
    parser.add_argument("--total_timesteps", help="Total timesteps to train with PPO", required=True,
                        type=int)
    parser.add_argument("--policy_dims", help="Hidden layers for policy network", nargs='+', type=int, required=True)
    parser.add_argument("--value_dims", help="Hidden layers for value predictor network", nargs='+', type=int, required=True)
    parser.add_argument("--eval_every", help="Evaluate current policy every eval_every episodes", required=True,
                        type=int)
    parser.add_argument("--pretrained_path", help="Path to the pretrained policy zip file, if any", type=str)
    parser.add_argument("--stats_path", help="Path to the pretrained policy normalizer stats file, if any", type=str)
    parser.add_argument("--log_std_init", help="Initial Gaussian policy exploration level", type=float, default=-2.0)
    parser.add_argument("--debug", help="Set true to disable parallel processing and run debugging programs",
                        action="store_true")
    parser.add_argument("--device", help="Device option for stable baselines algorithms", default="auto")
    args = parser.parse_args()
    main(args)
