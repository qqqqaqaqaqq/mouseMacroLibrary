import dataclasses
from datetime import datetime

@dataclasses.dataclass
class MousePoint():
    timestamp:datetime
    x:int
    y:int
    deltatime:float