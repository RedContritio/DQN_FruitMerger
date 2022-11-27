import cv2
import numpy as np
from GameInterface import GameInterface
from DQN import Agent, build_model
import paddle

if __name__ == "__main__":
    class_count = 11
    memory_size = 15

    action_dim = GameInterface.ACTION_NUM
    feature_dim = (class_count + 2) * (memory_size + 1)
    e_greed = 0
    e_greed_decrement = 1e-6

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    model_path = "final.pdparams"

    agent.policy_net.set_state_dict(paddle.load(model_path))

    env.reset()

    step, rewards_sum = 0, 0
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    assert alive

    WINNAME = "fruit-merger AI"

    cv2.namedWindow(WINNAME)

    while alive:
        env.game.draw()
        cv2.imshow(WINNAME, env.game.screen)
        key = cv2.waitKey(0)

        if key == ord("q") or key == 27:
            break
        # close the window
        if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
            break

        step += 1

        action = agent.sample(feature)
        next_feature, reward, alive = env.next(action)

        reward_sum = np.sum(reward)
        rewards_sum += reward_sum

        feature = next_feature

        print(f"step: {step}, reward: {reward_sum}, rewards_sum: {rewards_sum}")

    env.game.draw()
    cv2.imshow(WINNAME, env.game.screen)
    key = cv2.waitKey(0)
