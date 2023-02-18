import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils import ReplayMemory, logger, summary
import parl
import copy
from GameInterface import GameInterface
import os
import typing
from multiprocessing import Pipe, Pool
from multiprocessing.connection import Connection

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

MAX_EPISODE = 20000

def start_game(conn: Connection):
    game = GameInterface()
    while True:
        action = conn.recv()
        feature, reward, alive = game.next(action)

        if not alive:
            game.reset()
        
        conn.send([feature, reward, alive])

class CombinedEnvs:
    def __init__(self, process_num: int = GameInterface.ACT_DIM) -> None:
        self.process_num = process_num
        
        self.process_pool = Pool(self.process_num)
        self.parent_conns, self.child_conns = zip(*[Pipe() for _ in range(self.process_num)])
        
        for i in range(self.process_num):
            self.process_pool.apply_async(start_game, args=(self.child_conns[i],))
    
    def next(self, actions: typing.List[int]):
        for i, conn in enumerate(self.parent_conns):
            conn.send(actions[i])
        
        features, rewards, alives = [], [], []
        
        for i, conn in enumerate(self.parent_conns):
            feature, reward, alive = conn.recv()
            features.append(feature)
            rewards.append(reward)
            alives.append(alive)
            
        return features, rewards, alives
        

class Model(parl.Model):
    def __init__(self, obs_dim: int, act_dim: int):
        super(Model, self).__init__()
        self.hidden_size = [256, 256, 256]

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


def run_evaluate_episode(env: GameInterface, agent: Agent, eval_episode_groups=1, render=False):
    rewards_sum = []
    
    for i in range(eval_episode_groups):
        for j in range(env.ACT_DIM):
            env.reset()

            reward_sum = 0
            action = j
            feature, _, alive = env.next(action)

            while alive:
                action = agent.predict(feature)
                next_feature, reward, alive = env.next(action)

                reward = reward if alive else -1000

                if alive:
                    reward_sum += np.sum(reward)

                feature = next_feature
            
            rewards_sum.append(reward_sum)

    return np.mean(rewards_sum)

def run_train_episode(env: GameInterface, agent: Agent, rpm: ReplayMemory, episode_id: int):
    env.reset()

    step, rewards_sum = 0, 0

    # action = np.random.randint(0, env.action_num)
    action = episode_id % env.action_num
        
    feature, _, alive = env.next(action)

    assert alive

    while alive:
        step += 1

        action = agent.sample(feature)
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

            _loss = agent.learn(
                feature_batch,
                action_batch,
                reward_batch,
                next_feature_batch,
                alive_batch,
            )

        if alive:
            reward_sum = np.sum(reward)
            rewards_sum += reward_sum

        feature = next_feature

    return rewards_sum


if __name__ == "__main__":
    parl.connect("localhost:6007", distributed_files=['resources/images/*'])
    
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
    warmup_id = 0
    while warmup_id % act_dim != 0 or len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(env, agent, rpm, warmup_id)
        warmup_id += 1

    episode = 0
    EVALUATE_EVERY_EPISODE = 100
    while episode < MAX_EPISODE:
        for i in range(EVALUATE_EVERY_EPISODE):
            total_reward = run_train_episode(env, agent, rpm, episode)
            episode += 1
            
        eval_reward = run_evaluate_episode(env, agent, eval_episode_groups=1)
        logger.info(f"Episode: {episode}, Reward: {total_reward}, Eval Reward: {eval_reward}")
        summary.add_scalar("log/train", total_reward, episode)
        summary.add_scalar("log/eval", eval_reward, episode)
        
    agent.save('./dqn_model.ckpt')