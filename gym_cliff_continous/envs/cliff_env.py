import gym
import numpy as np
import warnings
import cv2

from gym import error, spaces, utils
from gym.utils import seeding


class CliffContinuous(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w=400, h=400, speed=20):

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        self.w = w
        self.h = h
        self.speed = speed
        self.state = np.array([0., 0.])
        self.trajectory = [np.copy(self.state)]
        self.ep_len = 0

    def step(self, action):
        prev_state = np.copy(self.state)
        self.move(self.state, self.speed, action)
        self.move(self.state, self.speed / 2, self.action_space.sample())
        self.trajectory.append(np.copy(self.state))
        self.ep_len += 1
        if self.ep_len >= 200:
            reward = -100
            terminal = True

        elif self.in_box():
            reward = -1
            terminal = False
        elif self.in_goal():
            reward = +10
            terminal = True
        elif self.in_cliff():
            if self.state[0] < self.w:
                reward = -100 * (1 - (self.state[0] / self.w))
            else:
                reward = -100
            terminal = True
        else:
            reward = -1
            terminal = False
            self.state = prev_state
            del self.trajectory[-1]
        
        
        return np.copy(self.state), reward, terminal, {}


    def reset(self):
        self.state = np.array([0., 0.])
        self.trajectory = [np.copy(self.state)]
        self.ep_len = 0
        return np.copy(self.state)

    def render(self, mode='rgb_array'):
        img = np.zeros((self.w * 2, self.h * 2, 3), np.uint8)
        cv2.rectangle(img, (self.w // 2, self.h // 2), (3 * self.w // 2, 3 * self.h // 2), (255, 255, 255), 1)
        cv2.rectangle(img, (self.w // 2, 0), (int(1.3 * self.w), self.h // 2), (200, 200, 255), -1)
        cv2.rectangle(img, (int(1.3 * self.w), 0), (int(1.5 * self.w), self.h // 2), (200, 255, 200), -1)

        for i, point in enumerate(self.trajectory[1:]):
            start = np.copy(self.trajectory[i])
            point = np.copy(point)
            start += np.array([self.w / 2, self.h / 2])
            point += np.array([self.w / 2, self.h / 2])
            
            cv2.line(img, tuple(start.astype(int)), tuple(point.astype(int)), (200, 255, 255), 2)


        img = np.flip(img, axis=0)
        return img

    def in_box(self):
        if (0 <= self.state[0] <= self.w) and (0 <= self.state[1] <= self.h):
            return True
        else:
            return False
    
    def in_cliff(self):
        if self.state[1] < 0:
            return True
        else:
            return False

    def in_goal(self):
        if (self.w * 0.8 <= self.state[0] <= self.w) and (self.state[1] < 0):
            return True
        else:
            return False

    @staticmethod
    def move(point, speed, theta):
        point[0] += speed * np.cos(theta)
        point[1] += speed * np.sin(theta)


if __name__ == "__main__":
    import numpy as np

    from sac import PessimisticSAC
    from stable_baselines3.sac import MlpPolicy
    from stable_baselines3.common.noise import NormalActionNoise

    from stable_baselines3.common.monitor import Monitor
    from gym.wrappers.time_limit import TimeLimit
    
    env = Monitor(CliffContinous())
    action_noise = None
    # action_noise = NormalActionNoise(mean=0, sigma=1)

    model = PessimisticSAC(MlpPolicy, env, action_noise=action_noise, verbose=1, beta=0.2, sample_action_n=64)
    model.learn(total_timesteps=100000)
    model.save("beta0.2")

    del model

    model = PessimisticSAC(MlpPolicy, env, action_noise=action_noise, verbose=1, beta=0.7, sample_action_n=64)
    model.learn(total_timesteps=100000)
    model.save("beta0.7")



