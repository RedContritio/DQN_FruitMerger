import enum
import typing


class EventType(enum.Enum):
    LBUTTONDOWN = (1,)
    MOUSEMOVE = (2,)
    RBUTTONDOWN = (3,)


class Event:
    def __init__(self, type: EventType):
        self.type = type

    def __str__(self) -> str:
        keys = [k for k in dir(self) if not k.startswith("__")]
        assert "type" in keys
        kvs = dict([(k, getattr(self, k)) for k in keys if k != "type"])
        return (
            f'Event {self.type.name} {", ".join([f"{k}={v}" for k, v in kvs.items()])}'
        )


class MouseEvent(Event):
    def __init__(self, type: EventType, pos: tuple[int, int]):
        self.pos = pos
        super().__init__(type)


class GameEventBase:
    def __init__(self):
        self.__events = []

    def add_event(self, event: Event):
        self.__events.append(event)

    def get_events(self) -> typing.List[Event]:
        events = self.__events[:]
        self.__events.clear()
        return events


if __name__ == "__main__":
    base = GameEventBase()

    base.add_event(MouseEvent(EventType.LBUTTONDOWN, (1, 2)))

    for e in base.get_events():
        print(e)
