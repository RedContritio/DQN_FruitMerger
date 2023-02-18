import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils import ReplayMemory, logger
import parl
import copy
from GameInterface import GameInterface
import os

WEIGHT_DIR = "weights"
OUTPUT_DIR = "output"

FINAL_PARAM_PATH = "final.pdparams"

for d in [WEIGHT_DIR, OUTPUT_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

LEARN_FREQUENCY = 1
MEMORY_SIZE = 50000
MEMORY_WARMUP_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GAMMA = 0.99


class Model(parl.Model):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Model, self).__init__()
        self.hidden_size = [64, 64, 64]

        self.fc1 = (nn.Linear(obs_dim, self.hidden_size[0]))
        self.fc2 = (nn.Linear(self.hidden_size[0], self.hidden_size[1]))
        self.fc3 = (nn.Linear(self.hidden_size[1], self.hidden_size[2]))
        self.fc4 = (nn.Linear(self.hidden_size[2], act_dim))

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = self.fc4(h3)
        return out


class Agent(parl.Agent):
    def __init__(
        self,
        algorithm,
        obs_dim: int,
        act_dim: int,
        e_greed: float = 0.1,
        e_greed_decrement: float = 0,
    ):
        super(Agent, self).__init__(algorithm)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        
        self.global_step = 0
        self.update_target_steps = 200

    def predict(self, obs) -> int:
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        act = int(pred_q.argmax())
        
        return act

    def sample(self, obs) -> int:
        if np.random.rand() < self.e_greed:
            act = np.random.randint(0, self.act_dim)
        else:
            act = self.predict(obs)
        
        self.e_greed = max(1e-5, self.e_greed - self.e_greed_decrement)

        return act

    def learn(self, obs, act, reward, next_obs, alive):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        
        # act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        alive = np.expand_dims(alive, axis=-1)

        obs = paddle.to_tensor(obs, dtype="float32")
        act = paddle.to_tensor(act, dtype="int32")
        reward = paddle.to_tensor(reward, dtype="float32")
        next_obs = paddle.to_tensor(next_obs, dtype="float32")
        alive = paddle.to_tensor(alive, dtype="float32")

        loss = self.alg.learn(obs, act, reward, next_obs, alive)
        return loss


def run_train_episode(env: GameInterface, agent: Agent, rpm: ReplayMemory):
    if not hasattr(run_train_episode, "id"):
        run_train_episode.id = 0

    env.reset()

    step, rewards_sum = 0, 0
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    debug = False

    assert alive

    while alive:
        step += 1

        action = agent.sample(feature)
        if not isinstance(action, int):
            print(action)
        next_feature, reward, alive = env.next(action)

        reward = reward if alive else -1000

        rpm.append(feature, action, reward, next_feature, alive)

        if len(rpm) > MEMORY_WARMUP_SIZE and step % LEARN_FREQUENCY == 0:
            (
                feature_batch,
                action_batch,
                reward_batch,
                next_feature_batch,
                alive_batch,
            ) = rpm.sample_batch(BATCH_SIZE)

            print(feature_batch.shape, action_batch.shape, reward_batch.shape, next_feature_batch.shape, alive_batch.shape)

            _loss = agent.learn(
                feature_batch,
                action_batch,
                reward_batch,
                next_feature_batch,
                alive_batch,
            )

        reward_sum = np.sum(reward)
        rewards_sum += reward_sum

        feature = next_feature

        if debug and step % 20 == 0:
            logger.debug(
                f"Episode: {run_train_episode.id}, step: {step}, reward: {reward_sum}, e_greed: {agent.e_greed}"
            )
        if debug and step % 100 == 0:
            img_path = os.path.join(
                OUTPUT_DIR, f"episode_{run_train_episode.id}_step_{step}.png"
            )
            env.game.draw()

            env.game.save_screen(img_path)

    run_train_episode.id += 1

    return rewards_sum


if __name__ == "__main__":
    env = GameInterface()
    act_dim = env.ACT_DIM
    obs_dim = env.OBS_DIM
    logger.info("obs_dim {}, act_dim {}".format(obs_dim, act_dim))

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim=obs_dim, act_dim=1)
    act_dim = GameInterface.ACTION_NUM

    model = Model(obs_dim, act_dim)
    alg = parl.algorithms.DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim, act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    # warmup memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(env, agent, rpm)
