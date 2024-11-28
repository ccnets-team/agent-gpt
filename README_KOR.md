## 환경 게이트웨이(EnvGateway) - OpenAI Gym 및 Unity 환경을 위한 간단한 Flask API 래퍼
현재 버전: 1.0.0 <br>
최신 업데이트: 2024-11-25

---

### 소개
EnvGateway는 OpenAI Gym과 Unity ML-Agents 환경을 RESTful API로 감싸는 간단한 Flask 애플리케이션입니다. <br>
이를 통해 강화학습 환경을 HTTP 요청을 통해 원격으로 제어하고 상호 작용할 수 있습니다.

---

### 주요 기능
#### ㆍ환경 생성 및 관리
원하는 환경을 생성하고 고유한 키로 여러 환경을 관리할 수 있습니다.

#### ㆍ벡터화된 환경 지원
다중 환경 인스턴스를 동시에 생성하여 병렬 학습을 지원합니다.

#### ㆍ환경 상호 작용
환경의 재설정, 단계 진행, 종료 등을 API를 통해 수행할 수 있습니다.

#### ㆍ스페이스 정보 제공
액션 스페이스와 관측 스페이스 정보를 API로 제공합니다.

---
### 코드 상세 설명
**1. environment_gateway.py**<br>
`EnvGateway` 클래스 역할: Flask 앱을 생성하고 환경 관리 및 API 엔드포인트를 제공합니다.

<br>

**속성**
- `app`: Flask 애플리케이션 인스턴스.
- `environments`: 생성된 환경을 저장하는 딕셔너리.
- `_backend`: 환경 생성에 사용되는 백엔드 클래스 (GymEnv, UnityEnv 등).

<br>

**주요 메서드**<br>
`__init__()`: 앱 초기화 및 라우트 정의. <br>
`_define_routes()`: API 엔드포인트와 함수 연결.<br>
`make()`: 새로운 환경 생성 (/make 엔드포인트).<br>
`make_vec()`: 벡터화된 환경 생성 (/make_vec 엔드포인트).<br>
`reset()`: 환경 재설정 (/reset 엔드포인트).<br>
`step()`: 환경에서 한 단계 진행 (/step 엔드포인트).<br>
`action_space()`: 액션 스페이스 반환 (/action_space 엔드포인트).<br>
`observation_space()`: 관측 스페이스 반환 (/observation_space 엔드포인트).<br>
`close()`: 환경 종료 (/close 엔드포인트).<br>
`run()`: 서버 실행 및 백엔드 등록.

<br>

**상수 및 유틸리티**<br>
- HTTP 상태 코드 상수: HTTP_OK, HTTP_BAD_REQUEST 등.<br>
- 유틸리티 모듈: 데이터 변환 및 공간 직렬화를 위한 모듈 (convert_ndarrays_to_nested_lists, serialize_space 등).<br>

<br>

**2. GymEnv 클래스**<br>
**설명:** OpenAI Gym 환경을 위한 백엔드 구현체입니다.

<br>

**주요 메서드**<br>
`__init__(self, env, **kwargs)`: 환경 인스턴스 초기화.<br>
`make(env_id, **kwargs)`: 단일 환경 생성.<br>
`make_vec(env_id, num_envs, **kwargs)`: 벡터화된 환경 생성.<br>
`reset(**kwargs)`: 환경 재설정.<br>
`step(action)`: 환경에서 한 단계 진행.<br>
`close()`: 환경 종료.<br>

<br>

**3. UnityEnv 클래스** <br>
**설명:** Unity ML-Agents 환경을 Gym 인터페이스로 감싸는 클래스입니다.

<br>

**주요 기능**
- 여러 Unity 환경을 병렬로 실행하여 벡터화된 환경 지원.
- 관측 공간과 액션 공간을 Gym 스페이스로 변환.
- 환경의 초기화, 재설정, 단계 진행, 종료 관리.

<br>

**주요 메서드**<br>
`__init__(self, env_id, num_envs=1, is_vectorized=False, **kwargs)`: 환경 초기화.<br>
`_initialize_env_info(self)`: 에이전트 수 및 스펙 초기화.<br>
`_define_observation_space(self)`: 관측 공간 정의.<br>
`_define_action_space(self, start=1)`: 액션 공간 정의.<br>
`_create_action_tuple(self, actions, env_idx)`: Unity 환경에 전달할 액션 생성.<br>
`reset(self, **kwargs)`: 환경 재설정 및 초기 관측값 반환.<br>
`step(self, actions)`: 한 단계 진행하고 결과 반환.<br>
`close(self)`: 환경 종료.<br>

---

### 설치 방법
1.**리포지토리 클론 및 이동**
```bash
git clone https://github.com/your_username/envgateway.git
cd envgateway
```
1.**필요 패키지 설치**
```bash
pip install flask numpy gymnasium
# Unity 환경 사용 시 추가 설치
pip install mlagents
```

--- 
### 사용 방법
#### 1. 서버 실행
**1.1 백엔드 선택 및 서버 실행 코드 작성**
`run_server.py` 파일을 생성하고 다음을 작성합니다:

```python
from environment_gateway import EnvGateway
from gym_env import GymEnv  # Unity 환경 사용 시 unity_env에서 UnityEnv 임포트

EnvGateway.run(backend=GymEnv, port=5000)  # Unity 환경 사용 시 backend=UnityEnv
```

**1.2 서버 실행**
```bash
python run_server.py
```

#### 2. API 사용
#### 2.1 환경 생성
**- 단일 환경 생성 (`POST /make`)**
```json
요청:
POST /make
{
  "env_id": "CartPole-v1",
  "env_key": "env1"
}
```
#### 2.2 환경 재설정
**- 환경 재설정 (`POST /reset`)**
```json
요청:
POST /reset
{
  "env_key": "env1"
}
```

#### 2.3 단계 진행
**- 액션 수행 (`POST /step`)**
```json
요청:
POST /step
{
  "env_key": "env1",
  "action": 0
}
```

#### 2.4 환경 종료
**- 환경 종료 (`POST /close`)**
```json
요청:
POST /close
{
  "env_key": "env1"
}
```

#### 3. 예제 코드
```
import requests

base_url = 'http://localhost:5000'

# 환경 생성
requests.post(f'{base_url}/make', json={"env_id": "CartPole-v1", "env_key": "env1"})

# 환경 재설정
requests.post(f'{base_url}/reset', json={"env_key": "env1"})

# 액션 수행
requests.post(f'{base_url}/step', json={"env_key": "env1", "action": 0})

# 환경 종료
requests.post(f'{base_url}/close', json={"env_key": "env1"})
```
---

### 주의사항
- **환경 키(`env_key`) 관리:** 여러 환경을 관리할 때 각 환경을 고유한 `env_key`로 식별합니다.
- **백엔드 확장:** 새로운 환경을 추가하려면 해당 백엔드를 구현하고 `EnvGateway.run()`에 등록하면 됩니다.




