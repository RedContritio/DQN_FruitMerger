import random
import typing
import cv2

import numpy as np
from Game import GameCore


class GameInterface:
    ACTION_NUM = 16
    SIMULATE_FPS = 60

    def __init__(self) -> None:
        self.game = GameCore()
        self.action_num = GameInterface.ACTION_NUM
        self.action_segment_len = self.game.width / GameInterface.ACTION_NUM

    def reset(self) -> None:
        self.game.reset()

    def simulate_until_stable(self) -> None:
        self.game.update_until_stable(GameInterface.SIMULATE_FPS)

    def decode_action(self, action: int) -> typing.Tuple[int, int]:
        x = int((action + 0.5) * self.action_segment_len)

        return (x, 0)

    def next(self, action: int):
        self.simulate_until_stable()

        feature = self.game.get_features()
        current_fruit = self.game.current_fruit_type

        score_1 = self.game.score

        self.game.click(self.decode_action(action))
        self.simulate_until_stable()

        score_2 = self.game.score

        score, reward, alive = self.game.score, score_2 - score_1, self.game.alive

        reward = reward if reward > 0 else -current_fruit

        flatten_feature = np.expand_dims(feature.flatten(), axis=0).astype(np.float32)

        return flatten_feature, reward, alive

    def auto_play(self):
        WINNAME, VIDEO_FPS = "fruit-merger", 5
        cv2.namedWindow(WINNAME)

        while True:
            action = random.randint(0, self.action_num - 1)
            feature, reward, alive = self.next(action)

            self.game.draw()
            cv2.imshow(WINNAME, self.game.__screen)

            key = cv2.waitKey(int(1000 / VIDEO_FPS))

            print(feature.shape)

            if not alive:
                self.game.rclick((0, 0))

            # if key != -1:
            #     print(key)

            if key == ord("q") or key == 27:
                break
            # close the window
            if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    gi = GameInterface()
    gi.auto_play()
