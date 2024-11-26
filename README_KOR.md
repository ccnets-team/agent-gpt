## 환경 생성 및 제어 프레임워크 <br>(UnityBackend, MujocoBackend, EnvironmentFactory)
현재 버전: 1.0.0 <br>
최신 업데이트: 2024-11-25

---

### 소개
이 프레임워크는 강화학습 환경을 손쉽게 생성하고 제어하기 위해 설계되었습니다. <br>
Unity ML-Agents와 OpenAI Gymnasium을 활용해 Unity 환경과 Mujoco 환경을 지원하며, <br>
EnvironmentFactory를 통해 다양한 환경 백엔드를 통합적으로 관리할 수 있습니다.

---

### 주요 기능
#### ㆍ강화학습 환경 생성 및 제어
단일 환경과 벡터화된 다중 환경을 손쉽게 생성하고 관리합니다.

#### ㆍ통합 인터페이스
Unity와 Mujoco 같은 다양한 환경 백엔드를 공통 인터페이스로 사용할 수 있습니다.

#### ㆍ확장성
Unity ML-Agents와 OpenAI Gymnasium API를 기반으로 강화학습 실험 및 연구에 유연하게 활용 가능합니다.

---

### 코드 구성
#### UnityBackend
ㆍ Unity ML-Agents 기반 강화학습 환경을 제어합니다. <br>
ㆍ 그래픽 사용 여부, 실행 속도, 다중 환경 등 다양한 설정 지원.

#### MujocoBackend
ㆍ Mujoco 환경을 위한 OpenAI Gymnasium 확장 클래스입니다. <br>
ㆍ 단일 및 벡터화 환경을 지원하여 병렬 학습 환경 구성 가능.

#### EnvironmentFactory
ㆍ Unity와 Mujoco 백엔드를 통합적으로 관리합니다. <br>
ㆍ 백엔드 등록 후 단일 인터페이스를 통해 환경 생성 및 제어 가능.

---

### 사용 방법
#### 1. 설치
1. **Unity ML-Agents 설치**
```bash
pip install mlagents
```
  
2. **Mujoco 및 Gymnasium 설치**
```bash
pip install gymnasium[mujoco]
```
---

#### 2. 백엔드 등록 및 환경 생성

**1.백엔드 등록**
```python
from UnityBackend import UnityBackend
from MujocoBackend import MujocoBackend
from EnvironmentFactory import EnvironmentFactory

# UnityBackend 등록
EnvironmentFactory.register(UnityBackend)

# MujocoBackend 등록
EnvironmentFactory.register(MujocoBackend)
```

 **2. 환경 생성**
##### ㆍ 단일 환경 생성
```python
env = EnvironmentFactory.make(env_id="3DBallHard", use_graphics=True, time_scale=64)```
```

##### ㆍ 벡터화 환경 생성
```python
vec_env = EnvironmentFactory.make_vec(env_id="HalfCheetah-v4", num_envs=4)
```

---

#### 3. 환경 초기화 및 학습

#### 1. 환경 초기화
```python
observations = env.reset()
```

#### 2.행동 실행
```python
action = env.action_space.sample()
next_observations, reward, terminated, info = env.step(action)
```

#### 3.환경 종료
```python
env.close()
```

---

### 코드 기능 설명
#### UnityBackend

**환경 생성:**
- Unity 환경 바이너리(env_id)로 단일 또는 다중 환경 생성.
- 그래픽 여부(use_graphics) 및 실행 속도(time_scale) 설정.

**예외 처리:**
-이미지 관찰 지원 불가 시:
```python
ValueError("Image observations are not supported.")
MujocoBackend
```

#### 환경 생성:
- Mujoco 환경 ID(env_id)로 단일 및 벡터화된 환경 생성.

**예외 처리:**
- Gymnasium 기본 예외 처리 활용.

#### EnvironmentFactory
#### - 백엔드 등록 및 환경 생성:

- 백엔드가 등록되지 않았을 경우:
```python
ValueError("No backend registered. Call 'EnvironmentFactory.register' first.")
```

- 백엔드가 make나 make_vec를 구현하지 않았을 경우:
```python
ValueError("Backend does not implement 'make' method.")
```

---

### 예외 처리 및 디버깅 팁
1. 백엔드 등록 여부 확인
환경 생성 전에 `EnvironmentFactory.register()`로 백엔드가 등록되었는지 확인하세요.

2. Unity 환경 경로 확인
`env_id`는 Unity 바이너리 파일 경로여야 합니다.

3. Mujoco 환경 ID 확인
OpenAI Gymnasium에서 제공하는 환경 ID를 사용하세요.

4. 오류 메시지 활용
발생한 예외 메시지를 통해 문제를 빠르게 디버깅할 수 있습니다.

