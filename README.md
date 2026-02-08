## Credits
This library is based on the [Original Project Name](https://github.com/qqqqaqaqaqq/mouseMacroDetector.git) source code.
- Original Model: Transformer based Autoencoder
- Changes: Refactored for library use, added preprocessing scripts, etc.

pip install git+https://github.com/qqqqaqaqaqq/mouseMacroLibrary.git

---

# python
```
import macro_detector

# 임포트 경로가 잘 잡혔는지 확인
print(macro_detector.__file__) 

sample_data = {
    'x': 100, # int
    'y': 200, # int
    'timestamp': 2026-02-03T19:26:54.887758, # str
    'deltatime': 0.01 # float
}

# 실행 테스트
result = macro_detector.get_macro_result(sample_data)

# 결과 출력 (초반 SEQ_LEN개까지는 데이터 쌓는 중이라 None이 나옵니다)
print(f"결과: {result}")
```

--- 

# fastapi
```
from fastapi import APIRouter
from typing import List
from macro_detector import get_macro_result, MousePoint

router = APIRouter()

# class MousePoint(BaseModel):
#     timestamp: datetime
#     x: int
#     y: int
#     deltatime: float

@router.post("/get_points")
async def get_mouse_pointer(data: List[MousePoint]):

    print(len(data))
    result = get_macro_result(data)

    if result:
        print(result)
    
    return {"status": "collecting", "buffer_count": "데이터 축적 중..."}
```
```
100
{
    'status': '0', 
    'data': [
        {'raw_error': 0.01729, 'threshold': 0.054254673421382904, 'is_macro': np.False_}, 
        {'raw_error': 0.01732, 'threshold': 0.054254673421382904, 'is_macro': np.False_}, 
        {'raw_error': 0.01729, 'threshold': 0.054254673421382904, 'is_macro': np.False_},
        ...
        ]
}

```

# Return Code
```
# error
return {
    "status": "1",
    "message": f"데이터가 부족합니다. 현재 {len(receive_data_list)}개 보냈습니다. 최소 51개 이상 넣어주세요.",
    "hint": {}
}

return {
    "status": "1",
    "message": f"데이터 형식 오류입니다. 해당 데이터 형식으로 전달 해주세요.",
    "hint": {
        "example": [
            {
                "timestamp": "2026-02-08T20:48:29",
                "x": 100,
                "y": 200,
                "deltatime": 0.016
            }
        ],
        "description": "위와 같은 형식의 객체를 리스트에 담아 최소 51개 이상 POST 요청으로 보내야 분석이 시작됩니다."
    }
}   

# success
result = {
    "status": "0",
    "data" : all_data
}
```