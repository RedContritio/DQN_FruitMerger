import os
import numpy as np

import paddle
from DQN import FINAL_PARAM_PATH, Agent, RandomAgent, build_model, evaluate
from GameInterface import GameInterface
from PRNG import PRNG

evaluate_random = PRNG()
evaluate_random.seed("RedContritio")

if __name__ == "__main__":
    EVALUATE_TIMES = 200

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2
    e_greed = 0.5
    e_greed_decrement = 1e-6

    env = GameInterface()

    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    if os.path.exists(FINAL_PARAM_PATH):
        agent.policy_net.set_state_dict(paddle.load(FINAL_PARAM_PATH))
        print("Loaded final param.")

    random_agent = RandomAgent(GameInterface.ACTION_NUM)

    scores1, rewards1 = [], []
    scores2, rewards2 = [], []

    for _ in range(EVALUATE_TIMES):
        seed = evaluate_random.random()

        score1, reward1 = evaluate(env, agent, seed)
        scores1.append(score1)
        rewards1.append(reward1)

        score2, reward2 = evaluate(env, random_agent, seed)
        scores2.append(score2)
        rewards2.append(reward2)

    print(
        f"""[DQN Agent]\t:\tmean_score: {np.mean(scores1)},\tmean_reward: {np.mean(rewards1)},
\t\t\tmax_score: {np.max(scores1)},\tmax_reward: {np.max(rewards1)},
\t\t\tmin_score: {np.min(scores1)},\tmin_reward: {np.min(rewards1)}"""
    )
    print(
        f"""[Random Agent]\t:\tmean_score: {np.mean(scores2)},\tmean_reward: {np.mean(rewards2)},
\t\t\tmax_score: {np.max(scores2)},\tmax_reward: {np.max(rewards2)},
\t\t\tmin_score: {np.min(scores2)},\tmin_reward: {np.min(rewards2)}"""
    )
