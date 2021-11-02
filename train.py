import os
from argparse import ArgumentParser
from shutil import copyfile

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import wandb
from gym import Env
from colorhash import ColorHash
from scipy.interpolate import griddata
from stable_baselines3 import SAC,A2C,PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from torch import nn

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

class CellDivisionCallback(BaseCallback):

    def __init__(self, envs, policy_dims, model_name, env_name, verbose=1):
        self.envs = envs
        self.policy_dims = policy_dims
        self.policy_dims_str = '_'.join([str(x) for x in self.policy_dims])
        self.model_name = model_name
        self.env_name = env_name
        self.i = 0
        super().__init__(verbose)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_start(self) -> None:
        states = []
        for i in np.arange(-10, 110):
            for j in np.arange(-3, 3, 0.05):
                states.append([i, j])
        states = np.stack(states)
        states_scaled = self.envs.normalize_obs(states)
        states_tensor = torch.as_tensor(states_scaled).float()

        policy: ActorCriticPolicy = self.model.policy.cpu()
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

        # convert raw integer
        convert_raw_integer_to_colorhash = np.vectorize(lambda x: ColorHash(x).rgb)
        grid_z = np.array(convert_raw_integer_to_colorhash(z)).swapaxes(0, 1).swapaxes(1, 2)

        plt.figure()
        plt.imshow(grid_z, extent=[-10, 110, -3, 3], aspect='auto')
        plt.title("State Space Visualized")
        plt.xlabel("$x$")
        plt.ylabel("$\\dot x$")

        # Create a directory for the current policy dimensions if it does not exist yet
        path_for_figures = f'policies/{self.env_name}/{self.model_name}/{self.policy_dims_str}'
        if not os.path.exists(path_for_figures):
            os.mkdir(path_for_figures)

        plt.savefig(f'{path_for_figures}/fig{self.i}')
        self.i += 1

def main(args):

    # wandb.init(project=args.project_name, name=args.run_name)
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


    # eval_callback = WAndBEvalCallback(render_env, args.eval_every, envs)
    # callback.callbacks.append(eval_callback)

    celldivision_callback = CellDivisionCallback(envs=envs,policy_dims=args.policy_dims,model_name=args.model_name,env_name=args.env)
    callback.callbacks.append(celldivision_callback)
    
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

        # Loop through different policy dims, train model for each
        for policy_dims in [args.policy_dims]:
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                net_arch=[dict(
                    vf=args.value_dims,
                    pi=policy_dims # args.policy_dims
                )
                ],
                log_std_init=args.log_std_init,
                squash_output=False
            )

            # Could loop though different algorithms as well
            if args.model_name.lower() == 'ppo':
                learner = PPO(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, policy_kwargs=policy_kwargs, device=args.device, target_kl=2e-2)
            elif args.model_name.lower() == 'a2c':
                learner = A2C(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, policy_kwargs=policy_kwargs, device=args.device)
            elif args.model_name.lower() == 'sac':
                learner = SAC(MlpPolicy, envs, n_steps=args.n_steps, verbose=1, policy_kwargs=policy_kwargs, device=args.device)

            if args.device == 'cpu':
                torch.cuda.empty_cache()
            learner.learn(total_timesteps=args.total_timesteps, callback=callback)

    render_env.close()
    envs.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_name", help="Weights & Biases project name", required=True, type=str)
    parser.add_argument("--run_name", help="Weights & Biases run name", required=True, type=str)
    parser.add_argument("--env", help="Name of the environment as registered in __init__.py somewhere", required=True,
                        type=str)
    parser.add_argument("--n_steps", help="Number of timesteps in each rollouts when training with model", required=True,
                        type=int)
    parser.add_argument("--total_timesteps", help="Total timesteps to train with model", required=True,
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
    parser.add_argument("--model_name", help="Model name", default="ppo")
    args = parser.parse_args()
    main(args)
