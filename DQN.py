import os
import random
import typing
import cv2
import collections
import numpy as np
from paddle import nn
import paddle

from GameInterface import GameInterface

WEIGHT_DIR = "weights"
OUTPUT_DIR = "output"

FINAL_PARAM_PATH = "final.pdparams"

for d in [WEIGHT_DIR, OUTPUT_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

LEARN_FREQUENCY = 5
MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99


class ReplayMemory(collections.deque):
    def __init__(self, max_size: int = MEMORY_SIZE) -> None:
        super().__init__(maxlen=max_size)

    def sample(
        self, batch_size: int
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mini_batch = random.sample(self, batch_size)

        # feature_batch, action_batch, reward_batch, next_feature_batch, alive_batch = experiences
        experiences = list(zip(*mini_batch))

        return tuple([np.array(exp) for exp in experiences])


def build_model(input_size: int, output_size: int) -> nn.Layer:
    model_prototype = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=output_size),
    )

    return model_prototype


class Agent:
    def __init__(
        self,
        build_model: typing.Callable,
        feature_dim: int,
        action_num: int,
        e_greed: float = 0.1,
        e_greed_decrement: float = 1e-6,
        learning_rate: float = LEARNING_RATE,
        loss_func: typing.Callable[
            [paddle.Tensor, paddle.Tensor], paddle.Tensor
        ] = nn.MSELoss("mean"),
    ) -> None:
        self.policy_net = build_model(feature_dim, action_num)
        self.target_net = build_model(feature_dim, action_num)
        self.feature_dim = feature_dim
        self.action_num = action_num
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

        self.loss_func = loss_func
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.policy_net.parameters(), learning_rate=learning_rate
        )

        self.global_step = 0
        self.update_target_steps = 200

    def sample(self, feature: np.ndarray) -> int:
        if np.random.uniform() < self.e_greed:
            action = np.random.randint(0, self.action_num, size=(1))
        else:
            action = self.predict(feature)

        self.e_greed = max(0, self.e_greed - self.e_greed_decrement)

        return action

    def predict(self, feature: np.ndarray) -> np.ndarray:
        with paddle.no_grad():
            action = self.policy_net(paddle.to_tensor(feature)).argmax()
        return action.numpy()

    def learn(
        self,
        feature: np.ndarray,
        action: int,
        reward: float,
        next_feature: np.ndarray,
        alive: bool,
    ):
        if self.global_step % self.update_target_steps == 0:
            self.target_net.load_dict(self.policy_net.state_dict())
            pass

        self.global_step += 1

        feature_batch = paddle.to_tensor(feature, dtype="float32")
        action_batch = paddle.to_tensor(action, dtype="int32")
        reward_batch = paddle.to_tensor(reward, dtype="float32")
        next_feature_batch = paddle.to_tensor(next_feature, dtype="float32")
        alive_batch = paddle.to_tensor(alive, dtype="float32")

        output_policy = paddle.squeeze(self.policy_net(feature_batch))
        action_batch = paddle.squeeze(action_batch)
        # print(action_batch, self.action_num)
        action_batch_onehot = nn.functional.one_hot(action_batch, self.action_num)

        # print(paddle.multiply(output_policy, action_batch_onehot).shape)
        policy_q_value = paddle.sum(
            paddle.multiply(output_policy, action_batch_onehot), axis=1
        )

        with paddle.no_grad():
            output_target_next = paddle.squeeze(self.target_net(next_feature_batch))
            target_next_q_value = paddle.max(output_target_next, axis=1)

        target_q_value = paddle.squeeze(reward_batch) + GAMMA * paddle.squeeze(
            target_next_q_value
        ) * paddle.squeeze(alive_batch)

        # print(policy_q_value.shape, target_q_value.shape)
        loss = self.loss_func(policy_q_value, target_q_value)

        self.optimizer.clear_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()


def run_episode(
    env: GameInterface, agent: Agent, memory: ReplayMemory, episode_id: int, debug=False
):
    env.reset()

    step, rewards_sum = 0, 0
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    assert alive

    while alive:
        step += 1

        action = agent.sample(feature)
        next_feature, reward, alive = env.next(action)

        memory.append((feature, action, reward, next_feature, alive))

        if (
            len(memory) >= MEMORY_WARMUP_SIZE
            and agent.global_step % LEARN_FREQUENCY == 0
        ):
            (
                feature_batch,
                action_batch,
                reward_batch,
                next_feature_batch,
                alive_batch,
            ) = memory.sample(BATCH_SIZE)

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
            print(
                f"Episode: {episode_id}, step: {step}, reward: {reward_sum}, e_greed: {e_greed}"
            )
        if debug and step % 100 == 0:
            img_path = os.path.join(OUTPUT_DIR, f"episode_{episode_id}_step_{step}.png")
            env.game.draw()

            env.game.save_screen(img_path)

    return rewards_sum


if __name__ == "__main__":
    class_count = 11
    memory_size = 15

    action_dim = GameInterface.ACTION_NUM
    feature_dim = (class_count + 2) * (memory_size + 1)
    e_greed = 0.2
    e_greed_decrement = 1e-6

    env = GameInterface()

    memory = ReplayMemory(MEMORY_SIZE)

    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    if os.path.exists(FINAL_PARAM_PATH):
        agent.policy_net.set_state_dict(paddle.load(FINAL_PARAM_PATH))
        print("Loaded final param.")

    print("Warm up.")
    while len(memory) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, memory, -1)

    max_episode = 20000
    episode_per_save = max_episode // 20
    print("Start training.")
    for episode_id in range(1, max_episode + 1):
        total_reward = run_episode(env, agent, memory, episode_id)

        if episode_id % episode_per_save == 0:
            save_path = os.path.join(
                WEIGHT_DIR, f"episode_{episode_id}_reward_{total_reward}.pdparams"
            )
            paddle.save(agent.policy_net.state_dict(), save_path)
            print(f"Saved model to {save_path}")
            print(f"Episode: {episode_id}, reward: {total_reward}, e_greed: {e_greed}")

    paddle.save(agent.policy_net.state_dict(), FINAL_PARAM_PATH)
