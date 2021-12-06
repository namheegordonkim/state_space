import sys

import gym
import numpy as np
import pygame
from gym.spaces import Box


class Car1DEnv0(gym.Env):
    initial_x_pos = 0.
    target_x_pos = 100.
    simulation_timestep = 0.1
    current_x_pos = 0.
    current_x_vel = 0.
    current_x_acc = 0.
    max_x_vel = 3.
    rewards = {"distance_reward": 0., "action_penalty": 0.}
    screen_size = (640, 480)

    def __init__(self, render=False):
        # Set up action space and observation space
        self.render_yes = render

        action_high = np.array([1])
        action_low = -action_high
        self.action_space = Box(action_low, action_high)

        obs_high = np.array([np.inf, np.inf])
        obs_low = -obs_high
        self.observation_space = Box(obs_low, obs_high)

        self.seed()  # seed by default; seed can be set from outside too.

        # Display setup
        self.screen = None
        self.screen_initialized = False

    def _get_obs(self):
        return np.array([self.current_x_pos, self.current_x_vel])

    def step(self, action):
        """
        Action is the acceleration of the car.
        Apply the dynamics equations, weighted by simulation timestep:
        velocity = velocity + acceleration
        position = position + velocity
        """
        # Compute next state
        self.current_x_acc = float(action)
        # Very simple example simulation
        self.current_x_vel += self.current_x_acc
        self.current_x_vel = np.clip(self.current_x_vel, -self.max_x_vel, self.max_x_vel)
        next_state = self._get_obs()

        # Compute reward
        distance_to_target = self.target_x_pos - self.current_x_pos
        distance_penalty = np.abs(distance_to_target)
        distance_reward = 1 - 1e-2 * distance_penalty  # 1 when distance is 0
        action_penalty = np.abs(self.current_x_acc)
        max_x_vel_penalty = 0
        if np.abs(np.abs(self.current_x_vel) - self.max_x_vel) < 1e-3:
            max_x_vel_penalty = 1
        self.rewards = {
            "distance_reward": distance_reward,
            "action_penalty": action_penalty
        }
        reward = distance_reward - 1e-3 * action_penalty - 1e-2 * max_x_vel_penalty

        # Failure condition
        # None; Timeout limit is determined in __init__.py
        done = False
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.current_x_pos = self.initial_x_pos
        self.current_x_vel = 0.

        # Add random perturbations to initial conditions (adds robustness to physical perturbations)
        self.current_x_pos += self.np_random.uniform(-1, 1)
        self.current_x_vel += self.np_random.uniform(-0.1, 0.1)
        self.current_x_acc = 0.
        self.rewards = {"distance_reward": 0., "action_penalty": 0.}

        state = self._get_obs()
        return state

    def set_current_state(self, x_pos, x_vel):
        self.current_x_pos = x_pos
        self.current_x_vel = x_vel
        state = self._get_obs()
        return state

    def seed(self, seed=None):
        # PyBullet does seeding this way. Apparently it's more parallel processing friendly.
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if self.render_yes and not self.screen_initialized:
            pygame.init()
            self.screen = pygame.display.get_surface()
            self.screen = pygame.display.set_mode(self.screen_size)
            self.screen_initialized = True

        # Drawing loop
        black = 0, 0, 0
        white = 255, 255, 255
        red = 255, 0, 0
        green = 0, 255, 0
        self.screen.fill(black)

        # line
        screen_width, screen_height = self.screen_size
        y_pos = screen_height / 2
        line_start = (0, y_pos)
        line_end = (screen_width, y_pos)
        pygame.draw.line(self.screen, white, line_start, line_end)

        # objects, scaled to fit the screen
        # center is 0, 3/4 of the width is the target
        offset = screen_width / 2
        scaling_factor = 0.25 * screen_width / self.target_x_pos
        rendered_car_x_pos = int(self.current_x_pos * scaling_factor + offset)
        rendered_y_pos = int(y_pos)
        rendered_target_x_pos = int(self.target_x_pos * scaling_factor + offset)

        # car
        pygame.draw.circle(self.screen, white, (rendered_car_x_pos, rendered_y_pos), 30)

        # target
        pygame.draw.circle(self.screen, red, (rendered_target_x_pos, rendered_y_pos), 30)

        # texts
        font = pygame.font.SysFont("comicsansms", 32)
        car_x_pos_text = font.render("x={:.2f}".format(self.current_x_pos), False, white)
        self.screen.blit(car_x_pos_text, (rendered_car_x_pos, rendered_y_pos + 30))

        car_x_vel_text = font.render("v={:.2f}".format(self.current_x_vel), False, white)
        self.screen.blit(car_x_vel_text, (rendered_car_x_pos, rendered_y_pos + 60))

        car_x_acc_text = font.render("a={:.2f}".format(self.current_x_acc), False, white)
        self.screen.blit(car_x_acc_text, (rendered_car_x_pos, rendered_y_pos + 90))

        distance_reward_text = font.render("dist_reward={:.2f}".format(self.rewards["distance_reward"]), False, white)
        self.screen.blit(distance_reward_text, (rendered_car_x_pos, rendered_y_pos + 120))

        action_penalty_text = font.render("action_penalty={:.2f}".format(self.rewards["action_penalty"]), False, white)
        self.screen.blit(action_penalty_text, (rendered_car_x_pos, rendered_y_pos + 150))

        target_x_pos_text = font.render("x={:.2f}".format(self.target_x_pos), False, red)
        self.screen.blit(target_x_pos_text, (rendered_target_x_pos, rendered_y_pos - 60))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()


class Car1DEnv1(Car1DEnv0):
    initial_x_pos = 0.
    target_x_pos = 100.
    simulation_timestep = 1.0
    current_x_pos = 0.
    current_x_vel = 0.
    current_x_acc = 0.
    max_x_vel = 100.
    rewards = {"distance_reward": 0., "action_penalty": 0.}
    screen_size = (640, 480)

    def step(self, action):
        """
        Action is the acceleration of the car.
        Apply the dynamics equations, weighted by simulation timestep:
        position = position + (velocity * dt) + 1/2 * (acceleration * dt^2)
        velocity = velocity + (acceleration * dt)
        """
        # Compute next state
        self.current_x_acc = float(action)
        # Very simple example simulation
        self.current_x_pos += self.simulation_timestep * self.current_x_vel + 0.5 * self.simulation_timestep ** 2 * self.current_x_acc
        self.current_x_vel += self.simulation_timestep * self.current_x_acc
        self.current_x_vel = np.clip(self.current_x_vel, -self.max_x_vel, self.max_x_vel)
        next_state = self._get_obs()

        # Compute reward
        distance_to_target = self.target_x_pos - self.current_x_pos
        distance_penalty = np.abs(distance_to_target)
        distance_reward = 1 - 1e-2 * distance_penalty  # 1 when distance is 0
        action_penalty = np.abs(self.current_x_acc)
        max_x_vel_penalty = 0
        if np.abs(np.abs(self.current_x_vel) - self.max_x_vel) < 1e-3:
            max_x_vel_penalty = 1
        self.rewards = {
            "distance_reward": distance_reward,
            "action_penalty": action_penalty
        }
        reward = distance_reward - 1e-3 * action_penalty - 1e-2 * max_x_vel_penalty

        # Failure condition
        # None; Timeout limit is determined in __init__.py
        done = False
        info = {}
        return next_state, reward, done, info


class PendulumEnv0(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, render=False):
        self.render_yes = render
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.viewer = None

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u = action
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.render_yes:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
