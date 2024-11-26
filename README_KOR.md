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




