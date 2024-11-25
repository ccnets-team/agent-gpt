## 환경 생성 및 제어 프레임워크 (UnityBackend, MujocoBackend, EnvironmentFactory)

### 소개
이 프레임워크는 강화학습 환경을 손쉽게 생성하고 제어하기 위해 설계되었습니다. <br>
Unity ML-Agents와 OpenAI Gymnasium을 활용해 Unity 환경과 Mujoco 환경을 지원하며, 
EnvironmentFactory를 <br>통해 다양한 환경 백엔드를 통합적으로 관리할 수 있습니다.

 <br>

### 주요 기능
#### ㆍ강화학습 환경 생성 및 제어
단일 환경과 벡터화된 다중 환경을 손쉽게 생성하고 관리합니다.

#### ㆍ통합 인터페이스
Unity와 Mujoco 같은 다양한 환경 백엔드를 공통 인터페이스로 사용할 수 있습니다.

#### ㆍ확장성
Unity ML-Agents와 OpenAI Gymnasium API를 기반으로 강화학습 실험 및 연구에 유연하게 활용 가능합니다.

 <br>

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

<br>

### 사용 방법
#### 1. 설치
  1. Unity ML-Agents 설치 <br>
     'pip install mlagents'
  2. Mujoco 및 Gymnasium 설치 <br>
     'pip install gymnasium[mujoco]'


