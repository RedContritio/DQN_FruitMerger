from Game import GameCore
import cv2

WINNAME = "fruit-merger"

FPS = 60
FAST_MODE = True
# FAST_MODE = False

gc = GameCore()


def onTick():
    gc.update(1.0 / FPS)
    gc.draw(debug=True)
    # gc.draw()
    cv2.imshow(WINNAME, gc.screen)


def onMouse(event, x, y, flags, param=None):
    if event == cv2.EVENT_LBUTTONDOWN:
        gc.click((x, y))
        if FAST_MODE:
            gc.update_until_stable(FPS)
    elif event == cv2.EVENT_MOUSEMOVE:
        gc.move((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        gc.rclick((x, y))
        gc.save_screen()


if __name__ == "__main__":
    cv2.namedWindow(WINNAME)
    cv2.setMouseCallback(WINNAME, onMouse)

    while True:
        onTick()

        key = cv2.waitKey(int(1000 / FPS))

        if key != -1:
            print(key)

        if key == ord("q") or key == 27:
            break
        # close the window
        if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
            break

    cv2.destroyAllWindows()
