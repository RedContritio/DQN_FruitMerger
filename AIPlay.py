import typing
import cv2
import numpy as np
from Game import visualize_feature
from GameInterface import GameInterface
from DQN import Agent, build_model
import paddle

from render_utils import cover

if __name__ == "__main__":
    WINNAME = "fruit-merger AI"
    WINNAME2 = "feature map"

    cv2.namedWindow(WINNAME)
    cv2.namedWindow(WINNAME2)

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2
    e_greed = 0.4
    e_greed_decrement = 4e-6

    env = GameInterface()

    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    model_path = "final.pdparams"

    agent.policy_net.set_state_dict(paddle.load(model_path))

    env.reset()

    step, rewards_sum = 0, 0
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    assert alive

    while alive:

        reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
        feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(
            np.uint8
        )
        cv2.imshow(WINNAME2, feature_img)

        step += 1

        screen = env.game.draw()

        action = agent.sample(feature)

        unit_w = 1.0 * env.game.width / action_dim

        red_rect = np.zeros_like(screen, dtype=np.uint8)
        red_rect = cv2.rectangle(
            red_rect,
            (int(action * unit_w), 0),
            (int((action + 1) * unit_w), env.game.height),
            (0, 0, 255, 60),
            -1,
        )

        cover(screen, red_rect, 1)

        cv2.imshow(WINNAME, screen)

        key = cv2.waitKey(0)
        if key == ord("q") or key == 27:
            break
        # close the window
        if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
            break

        next_feature, reward, alive = env.next(action)

        reward_sum = np.sum(reward)
        rewards_sum += reward_sum

        feature = next_feature

        print(f"step: {step}, reward: {reward_sum}, rewards_sum: {rewards_sum}")

    env.game.draw()
    cv2.imshow(WINNAME, env.game.screen)
    print(f"score: {env.game.score}")
    key = cv2.waitKey(0)
