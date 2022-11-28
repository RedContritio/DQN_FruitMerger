import typing
import numpy as np
import pymunk
from threading import Lock
from GameEvent import GameEventBase, EventType, MouseEvent
import cv2
from PRNG import PRNG

from render_utils import cover, intersectRect, putInverseColorText, putText2

GRAVITY = (0, 800)
GAME_RESOLUTION = GAME_WIDTH, GAME_HEIGHT = 300, 400

# list[0] is nonsense for type 0
FRUIT_RADIUS = [int(1.3 * r) for r in [-1, 10, 15, 21, 23, 29, 35, 37, 50, 59, 60, 78]]
FRUIT_RADIUS = [int(1.2 * r) for r in FRUIT_RADIUS]
FRUIT_SIZES = [(2 * r, 2 * r) for r in FRUIT_RADIUS]

FRUIT_IMAGE_PATHS = [f"res/{i:02d}.png" for i in range(11)]
FRUIT_RAW_IMAGES = [
    cv2.imread(FRUIT_IMAGE_PATHS[i], -1) if i > 0 else None for i in range(11)
]

FRUIT_IMAGES = [
    None if img is None else cv2.resize(img, FRUIT_SIZES[i])
    for i, img in enumerate(FRUIT_RAW_IMAGES)
]


class Fruit:
    def __init__(self, type: int, x: int, y: int) -> None:
        self.type = type
        self.r = FRUIT_RADIUS[self.type]
        self.size = FRUIT_SIZES[self.type]

        self.x, self.y = x, y

    def update_position(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def draw(self, screen: np.ndarray) -> None:
        Fruit.paint(screen, self.type, self.x, self.y)

    def paint(
        screen: np.ndarray, type: int, x: int, y: int, alpha: float = 1.0
    ) -> None:
        assert type > 0 and type <= 11
        l, t = (x - FRUIT_RADIUS[type], y - FRUIT_RADIUS[type])
        w, h = FRUIT_SIZES[type]

        l, t, w, h = [int(v) for v in (l, t, w, h)]

        il, it, iw, ih = [
            int(v) for v in intersectRect((l, t, w, h), (0, 0, *screen.shape[1::-1]))
        ]
        # print(il, it, iw, ih)
        # cv2.addWeighted(screen[it:it+ih, il:il+iw], 1 - alpha, FRUIT_IMAGES[type][it-t:it-t+ih, il-l:il-l+iw], alpha, 0, screen[it:it+ih, il:il+iw])
        cover(
            screen[it : it + ih, il : il + iw],
            FRUIT_IMAGES[type][it - t : it - t + ih, il - l : il - l + iw],
            alpha=alpha,
        )
        # cv2.circle(screen, (x, y), FRUIT_RADIUS[type], (255, 0, 0), 2)


class GameCore(GameEventBase):
    def __init__(self, gravity: typing.Tuple[int, int] = GRAVITY) -> None:
        self.resolution = self.width, self.height = GAME_WIDTH, GAME_HEIGHT
        self.init_x = int(self.width / 2)
        self.init_y = int(0.15 * self.height)

        self.score = 0
        self.recent_score_delta = 0

        self.fruits: typing.List[Fruit] = []
        self.balls: typing.List[pymunk.Shape] = []

        self.background_color = (0xE1, 0x69, 0x41, 0)
        self.preset_background = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.preset_background[:, :] = self.background_color
        self.preset_redline_screen = self.preset_background.copy()
        cv2.line(
            self.preset_redline_screen,
            (0, self.init_y),
            (self.width, self.init_y),
            (0, 0, 255),
            2,
        )
        self.__screen = self.preset_background.copy()

        self.lock = Lock()
        self.render_lock = Lock()

        self.prng = PRNG()

        self.stable_frame_threshold = 10
        self.current_frame_id = 0
        self.stable_frame_id = self.current_frame_id - self.stable_frame_threshold
        self.clickable = False

        self.largest_fruit_type = 1
        self.current_fruit_type = self.create_random_fruit_type()

        self.reset()

        self.space = pymunk.Space()
        self.space.gravity = gravity

        self.reset()

        self.init_segment()
        self.setup_collision_handler()

        super().__init__()

    def reset(self, seed: int = None) -> None:
        for ball in self.balls:
            self.space.remove(ball, ball.body)

        self.prev_score, self.score = 0, 0

        self.fruits.clear()
        self.balls.clear()

        self.current_frame_id = 0
        self.stable_frame_id = self.current_frame_id - self.stable_frame_threshold
        self.prev_stable_frame_id = self.stable_frame_id
        self.clickable = False

        self.largest_fruit_type = 1
        self.current_fruit_type = self.create_random_fruit_type()

        self.prng.seed(**({} if seed is None else {"seed": seed}))

        self.alive = True

    def init_segment(self, thinkness: float = 20, friction: float = 0.6):
        l, t = 0 - thinkness, 0 - thinkness - self.height // 2
        r, b = self.width + thinkness, self.height + thinkness

        def create_segment(
            p1: typing.Tuple[int, int], p2: typing.Tuple[int, int]
        ) -> pymunk.Segment:
            s = pymunk.Segment(self.space.static_body, p1, p2, thinkness)
            s.friction = friction
            return s

        self.space.add(create_segment((l, t), (l, b)))
        self.space.add(create_segment((r, t), (r, b)))
        # no top wall
        # self.space.add(create_segment((l, t), (r, t)))
        self.space.add(create_segment((l, b), (r, b)))

    def setup_collision_handler(self):
        def collision_post_solve(arbiter: pymunk.Arbiter, space: pymunk.Space, _data):
            with self.lock:
                s0, s1 = arbiter.shapes[:2]
                new_type = s0.collision_type + 1
                x1, y1 = s0.body.position
                x2, y2 = s1.body.position
                x, y = (x1, y1) if y1 > y2 else (x2, y2)

                if s0 in self.balls and s1 in self.balls:
                    self.remove_ball(space, s0)
                    self.remove_ball(space, s1)

                    fruit = Fruit(new_type, x, self.init_y)
                    self.fruits.append(fruit)

                    ball = self.create_ball(
                        self.space, x, y, fruit.r // 10, fruit.r - 1, new_type
                    )
                    self.balls.append(ball)

                    self.largest_fruit_type = max(self.largest_fruit_type, new_type)
                    self.recent_score_delta = new_type if new_type < 11 else 100
                    self.score += self.recent_score_delta

        for collision_type in range(1, 11):
            self.space.add_collision_handler(
                collision_type, collision_type
            ).post_solve = collision_post_solve

    def create_random_fruit_type(self) -> int:
        return self.prng.randint(1, min(self.largest_fruit_type, 5))

    def create_fruit(self, type: int, x: int) -> Fruit:
        return Fruit(type, x, self.init_y - FRUIT_RADIUS[type])

    def create_ball(
        self,
        space: pymunk.Space,
        x: int,
        y: int,
        mass: int = 1,
        radius: int = 7,
        type: int = 1,
    ) -> pymunk.Shape:
        ball_moment = pymunk.moment_for_circle(mass, 0, radius)
        ball_body = pymunk.Body(mass, ball_moment)
        ball_body.position = x, y
        ball_shape = pymunk.Circle(ball_body, radius)
        ball_shape.elasticity = 0.3
        ball_shape.friction = 0.6
        ball_shape.collision_type = type
        space.add(ball_body, ball_shape)
        return ball_shape

    def remove_ball(self, space: pymunk.Space, ball: pymunk.Circle):
        p = self.balls.index(ball)

        space.remove(ball, ball.body)

        self.balls.pop(p)
        self.fruits.pop(p)

    def save_screen(self, path: str = "screenshot.png") -> bool:
        rgb_img = cv2.cvtColor(self.screen, cv2.COLOR_BGRA2BGR)
        return cv2.imwrite(path, rgb_img)

    def draw(self, debug=False):
        backbuffer = self.preset_background.copy()

        # if self.clickable:
        if self.current_fruit_type > 0:
            y = self.init_y - FRUIT_RADIUS[self.current_fruit_type]
            Fruit.paint(
                backbuffer,
                self.current_fruit_type,
                self.init_x,
                y,
                1 if self.clickable else 0.5,
            )

        for i, f in enumerate(self.fruits):
            f.draw(backbuffer)
            if debug:
                cv2.circle(backbuffer, (int(f.x), int(f.y)), f.r // 2, (0, 0, 0), 1)
                putInverseColorText(
                    backbuffer,
                    f"{self.balls[i].body.velocity.y:.2f}",
                    (int(f.x), int(f.y + f.r)),
                    font_scale=0.5,
                    thickness=1,
                )

        cv2.addWeighted(backbuffer, 1, self.preset_redline_screen, 0.5, 0, backbuffer)

        putInverseColorText(
            backbuffer,
            f"Score: {self.score}",
            (0, 20),
            font_scale=0.7,
            thickness=1,
            putTextFunc=cv2.putText,
        )

        if not self.alive:
            putInverseColorText(
                backbuffer,
                f"Failed\nClick RButton to Restart",
                (int(self.width / 2), int(self.height / 2)),
                font_scale=0.7,
                thickness=2,
            )

        with self.render_lock:
            self.__screen[:, :, :] = backbuffer
            return self.__screen

    @property
    def screen(self) -> np.ndarray:
        with self.render_lock:
            return self.__screen

    def get_features(self, width: int, height: int) -> np.ndarray:
        """
        params:
            - width: width of the grid
            - height: height of the grid
        return:
            - features: (height, width, 2) np.ndarray
                - features[:, :, 0]: smaller than current fruit
                - features[:, :, 1]: larger than current fruit
        """
        uw, uh = self.width / width, self.height / height

        features = np.zeros((height, width, 2), dtype=np.float32)

        # type, dr
        auxilary = np.zeros((height, width, 2), dtype=np.float32)
        auxilary[:, :, 1] = np.inf

        threshold = ((uw**2) + (uh**2)) // 2

        for f in self.fruits:
            r2 = f.r * f.r
            for j in range(width):
                x = (0.5 + j) * uw
                for i in range(height):
                    y = (0.5 + i) * uh

                    dx, dy = f.x - x, f.y - y
                    # dr = np.sqrt(dx * dx + dy * dy) - f.r
                    dr = dx * dx + dy * dy - r2

                    if dr < threshold and dr < auxilary[i, j, 1]:
                        auxilary[i, j, 0] = f.type
                        auxilary[i, j, 1] = dr

        is_empty = auxilary[:, :, 0] == 0
        is_same = auxilary[:, :, 0] == self.current_fruit_type

        features[:, :, 0] = auxilary[:, :, 0] - self.current_fruit_type
        features[:, :, 0] = features[:, :, 0].clip(max=0)
        features[:, :, 0][is_same] = 1
        features[:, :, 0][is_empty] = 0

        features[:, :, 1] = self.current_fruit_type - auxilary[:, :, 0]
        features[:, :, 1] = features[:, :, 1].clip(max=0)
        features[:, :, 1][is_same] = 1
        features[:, :, 1][is_empty] = 0

        return features

    def update_until_stable(self, fps: float = 60, max_seconds: int = 5):
        self.set_unstable()

        max_steps = int(fps * max_seconds)
        step = 0

        while (
            self.current_frame_id <= self.stable_frame_id + self.stable_frame_threshold
            and step < max_steps
        ):
            self.update(1.0 / fps)
            step += 1

        if step == max_steps:
            # if not os.path.exists("screenshots"):
            #     os.mkdir("screenshots")

            # print("status: forever unstable")
            # for i in range(10):
            #     self.draw(debug=True)
            #     self.save_screen(
            #         os.path.join(
            #             "screenshots",
            #             f"step_{i}_velocity_{self.max_balls_velocity_y}.png",
            #         )
            #     )
            #     print(
            #         self.current_frame_id,
            #         self.stable_frame_id,
            #         self.alive,
            #         self.clickable,
            #     )
            #     self.update(1.0 / fps)

            self.clickable = True

    def update(self, time_delta: float):
        # print(self.current_frame_id, self.stable_frame_id, self.alive, self.clickable, file=f)
        self.current_frame_id += 1
        self.space.step(time_delta)

        stable = self.check_stable()
        if not stable:
            self.set_unstable()

        self.alive = self.alive and self.check_alive()
        if not self.alive:
            for event in self.events:
                if event.type == EventType.RBUTTONDOWN:
                    self.reset()
                    break
            return

        if (
            not self.clickable
            and self.current_frame_id
            > self.stable_frame_id + self.stable_frame_threshold
        ):
            self.prev_stable_frame_id = self.stable_frame_id
            self.clickable = True

        for event in self.events:
            if event.type == EventType.LBUTTONDOWN and self.clickable:
                x, _y = event.pos

                fruit = self.create_fruit(self.current_fruit_type, x)
                self.fruits.append(fruit)

                y = self.init_y - fruit.r
                ball = self.create_ball(
                    self.space,
                    x,
                    y,
                    (fruit.r // 10) ** 2,
                    fruit.r - 1,
                    self.current_fruit_type,
                )
                self.balls.append(ball)

                self.current_fruit_type = self.create_random_fruit_type()
                self.set_unstable()
                self.clickable = False

            elif event.type == EventType.MOUSEMOVE:
                self.init_x, _y = event.pos
                self.init_x = max(
                    self.init_x, 0 + FRUIT_RADIUS[self.current_fruit_type]
                )
                self.init_x = min(
                    self.init_x, self.width - FRUIT_RADIUS[self.current_fruit_type]
                )

        assert not self.lock.locked()

        with self.lock:
            for i, ball in enumerate(self.balls):
                x, y = ball.body.position
                angle = ball.body.angle

                # xi, yi = int(x), int(y)

                self.fruits[i].update_position(x, y)

    def set_unstable(self) -> None:
        self.stable_frame_id = self.current_frame_id + 1

    def check_stable(self) -> bool:
        return self.max_balls_velocity_y < 20

    @property
    def max_balls_velocity_y(self) -> float:
        return (
            max([abs(ball.body.velocity.y) for ball in self.balls])
            if len(self.balls) > 0
            else 0
        )

    def check_alive(self) -> bool:
        if self.current_frame_id > self.stable_frame_id + self.stable_frame_threshold:
            for f in self.fruits:
                if f.y < self.init_y:
                    return False
        return True

    def click(self, pos: tuple[int, int]):
        self.add_event(MouseEvent(EventType.LBUTTONDOWN, pos))

    def move(self, pos: tuple[int, int]):
        self.add_event(MouseEvent(EventType.MOUSEMOVE, pos))

    def rclick(self, pos: tuple[int, int]):
        self.add_event(MouseEvent(EventType.RBUTTONDOWN, pos))


def visualize_feature(
    feature: np.ndarray, game_resolution: typing.Tuple[int, int]
) -> np.ndarray:
    game_w, game_h = game_resolution
    feature_img = np.zeros((game_h, game_w * 2, 3), dtype=np.uint8)

    uw, uh = game_w / feature.shape[1], game_h / feature.shape[0]

    print(feature[:, :, 0].max(), feature[:, :, 0].min())
    print(feature[:, :, 1].max(), feature[:, :, 1].min())

    _v2c = lambda v: 255 if v > 0 else (0 if v == 0 else int(-v / 13.0 * 255.0))
    value2color = (
        lambda v: (_v2c(v), _v2c(v), _v2c(v)) if v >= 0 else (127, _v2c(v), _v2c(v))
    )

    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature_img[
                int(i * uh) : int((i + 1) * uh), int(j * uw) : int((j + 1) * uw)
            ] = value2color(feature[i, j, 0])
            feature_img[
                int(i * uh) : int((i + 1) * uh),
                int(j * uw + game_w) : int((j + 1) * uw + game_w),
            ] = value2color(feature[i, j, 1])

            putText2(
                feature_img,
                f"{int(feature[i, j, 0])}",
                (int((j + 0.5) * uw), int((i + 0.5) * uh)),
                font_scale=0.3,
                color=(0, 0, 255),
            )
            putText2(
                feature_img,
                f"{int(feature[i, j, 1])}",
                (int((j + 0.5) * uw + game_w), int((i + 0.5) * uh)),
                font_scale=0.3,
                color=(0, 0, 255),
            )

    for i in range(feature.shape[0]):
        cv2.line(
            feature_img, (0, int(i * uh)), (game_w * 2, int(i * uh)), (255, 0, 0), 1
        )
    cv2.line(feature_img, (0, game_h - 1), (game_w * 2, game_h - 1), (255, 0, 0), 1)

    for j in range(feature.shape[1]):
        cv2.line(feature_img, (int(j * uw), 0), (int(j * uw), game_h), (0, 255, 0), 1)
    cv2.line(feature_img, (game_w - 1, 0), (game_w - 1, game_h), (0, 255, 0), 1)

    for j in range(feature.shape[1]):
        cv2.line(
            feature_img,
            (int(j * uw + game_w), 0),
            (int(j * uw + game_w), game_h),
            (0, 255, 0),
            1,
        )
    cv2.line(
        feature_img,
        (game_w - 1 + game_w, 0),
        (game_w - 1 + game_w, game_h),
        (0, 255, 0),
        1,
    )

    cv2.line(feature_img, (game_w, 0), (game_w, game_h), (255, 255, 0), 1)

    return feature_img
