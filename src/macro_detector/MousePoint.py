import dataclasses

@dataclasses.dataclass
class MousePoint():
    timestamp:str
    x:int
    y:int
    deltatime:float