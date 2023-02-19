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

FEATURE_MAP_WIDTH, FEATURE_MAP_HEIGHT = 16, 20

ACT_DIM = 16
OBS_DIM = FEATURE_MAP_WIDTH * FEATURE_MAP_HEIGHT * 2

MAX_EPISODE = 20000

class CombinedEnvs:
    def __init__(self, process_num: int = ACT_DIM) -> None:
        self.process_num = process_num
        
        self.games = [GameInterface() for _ in range(self.process_num)]
    
    def next(self, actions: typing.List[int]):
        jobs = [game.next(actions[i]) for i, game in enumerate(self.games)]
        returns = [job.get() for job in jobs]
        
        features, rewards, alives = list(zip(*returns))
            
        return features, rewards, alives
    
    def reset(self):
        for game in self.games:
            game.reset()
        

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


def run_evaluate_episode(envs: CombinedEnvs, agent: Agent):
    reward_list = []
    act_dim = ACT_DIM
    
    for i in range(1):
        envs.reset()
        total_rewards = [0 for _ in range(act_dim)]
        dead = [False for _ in range(act_dim)]
        
        actions = list(range(act_dim))
        np.random.shuffle(actions)
        
        features, rewards, alives = envs.next(actions)
        
        while not np.all(dead):
            actions = [agent.sample(feature) for feature in features]
            next_features, rewards, alives = envs.next(actions)
            
            for i, alive in enumerate(alives):
                if not dead[i]:
                    total_rewards[i] += rewards[i]
                    if not alive:
                        dead[i] = True
                        
            features = next_features
        
        reward_list += total_rewards

    return np.mean(reward_list)

def run_train_episode(envs: CombinedEnvs, agent: Agent, rpm: ReplayMemory, episode_id: int):
    step = 0

    envs.reset()
    dead_count = 0
    
    actions = list(range(envs.process_num))
    np.random.shuffle(actions)
    
    features, rewards, alives = envs.next(actions)

    while dead_count < 2 * envs.process_num:
        step += 1

        actions = [agent.sample(feature) for feature in features]
        next_features, rewards, alives = envs.next(actions)

        rewards = [reward if alives[i] else -1000 for (i, reward) in enumerate(rewards)]

        for i, action in enumerate(actions):
            rpm.append(features[i], action, rewards[i], next_features[i], alives[i])
            dead_count += 0 if alives[i] else 1

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

        features = next_features

    return None


if __name__ == "__main__":
    parl.connect("localhost:6007", distributed_files=['resources/images/*.png'])
    
    envs = CombinedEnvs()
    act_dim = ACT_DIM
    obs_dim = OBS_DIM
    logger.info("obs_dim {}, act_dim {}".format(obs_dim, act_dim))

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim=obs_dim, act_dim=1)

    model = Model(obs_dim, act_dim)
    alg = parl.algorithms.DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim, act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    # warmup memory
    warmup_id = 0
    while warmup_id % act_dim != 0 or len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(envs, agent, rpm, warmup_id)
        warmup_id += 1

    episode = 0
    EVALUATE_EVERY_EPISODE = 50
    while episode < MAX_EPISODE:
        for i in range(EVALUATE_EVERY_EPISODE):
            run_train_episode(envs, agent, rpm, episode)
            episode += 1
            
        eval_reward = run_evaluate_episode(envs, agent)
        logger.info(f"Episode: {episode}, Eval Reward: {eval_reward}")
        summary.add_scalar("log/eval", eval_reward, episode)
        
    agent.save('./dqn_model.ckpt')