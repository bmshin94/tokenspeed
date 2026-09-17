# TokenSpeed 한국어 정리 노트

이 문서는 TokenSpeed 저장소를 처음 접한 뒤, 코드와 문서를 직접 확인하며
정리한 한국어 학습/활용 노트입니다.

- 원본 저장소(upstream): https://github.com/lightseekorg/tokenspeed
- 이 저장소(fork): https://github.com/bmshin94/tokenspeed
- 공식 문서: https://lightseek.org/tokenspeed/
- 재단 홈페이지: https://lightseek.org/

> 정리 기준일: 2026-09-17
> 확인 방식: 저장소 내 `README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
> `docs/`, `.github/workflows/`, 소스 디렉터리 구조를 직접 열람

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [폴더 구조](#2-폴더-구조)
3. [핵심 아키텍처](#3-핵심-아키텍처)
4. [설치 및 사용법](#4-설치-및-사용법)
5. [플러그인·스킬·MCP 여부](#5-플러그인스킬mcp-여부)
6. [API 토큰 필요 여부](#6-api-토큰-필요-여부)
7. [GitHub에서 주목받는 이유](#7-github에서-주목받는-이유)
8. [로컬 에이전트 구축에 도움이 되는가](#8-로컬-에이전트-구축에-도움이-되는가)
9. [수익화 아이디어](#9-수익화-아이디어)
10. [React·PHP로 만들 수 있는가](#10-reactphp로-만들-수-있는가)
11. [실행 로드맵](#11-실행-로드맵)
12. [참고 링크](#12-참고-링크)

---

## 1. 한 줄 요약

**TokenSpeed는 LLM을 실제 서비스로 구동해 주는 추론(inference) 엔진입니다.**

모델 파일(가중치)만으로는 서비스가 되지 않습니다. 누군가 모델을 GPU에 올리고,
동시에 들어오는 수천 건의 요청을 줄 세우고, 메모리를 나눠 주고, 최대한 빨리
토큰을 내보내야 합니다. 그 역할을 하는 소프트웨어가 TokenSpeed입니다.

| 항목 | 내용 |
| --- | --- |
| 분류 | LLM 추론 서버 (독립 실행형 서버 소프트웨어) |
| 같은 카테고리 | vLLM, SGLang, TensorRT-LLM |
| 지향점 | "TensorRT-LLM 수준의 성능 + vLLM 수준의 사용성" |
| 주 타깃 | 에이전틱 워크로드 (긴 컨텍스트 + 반복 툴 호출 + 낮은 지연) |
| 운영 주체 | 비영리 LightSeek Foundation |
| 라이선스 | MIT (상업적 이용 가능) |
| 생태계 | PyTorch Ecosystem 등재 |

### 비유: "AI 식당의 주방"

- AI 모델 파일 = 요리사 (레시피는 알지만 혼자서는 아무것도 못 함)
- TokenSpeed = 주방 전체 (주문 접수, 순서 배정, 재료 자리 배분, 접시 배출)

---

## 2. 폴더 구조

저장소 루트를 직접 확인한 결과입니다.

```
tokenspeed/
├── python/                  런타임 (429개 파일) — 모델 실행부
├── tokenspeed-scheduler/    C++ 스케줄러 (101개) — 요청/캐시 제어
├── tokenspeed-kernel/       GPU 커널 (728개) — 가장 큰 덩어리
├── tokenspeed-kernel-amd/   AMD 전용 커널 (139개)
├── tokenspeed-kernel-npu/   Ascend NPU 커널 (13개)
├── tokenspeed-mla/          MLA 어텐션 독립 패키지
├── test/                    테스트 (469개)
├── docs/                    문서 (design/ 가 핵심)
└── .github/workflows/       CI (GB300/B300 실장비 나이트리 포함)
```

| 폴더 | 주방 비유 | 역할 |
| --- | --- | --- |
| `python/` | 조리대 | 모델을 실제로 실행 |
| `tokenspeed-scheduler/` | 주방장 | 요청 순서와 KV 캐시 자원 배분 |
| `tokenspeed-kernel/` | 칼·프라이팬 | GPU 연산 커널 |
| `test/` | 검수 | 정확성 검증 |
| `docs/design/` | 주방 규칙집 | 설계 불변 조건 기록 |

### 주요 하위 구조

**`python/tokenspeed/runtime/`**

`cache/`(KV 캐시), `moe/`(전문가 혼합), `distributed/`(멀티 GPU),
`sampling/`, `grammar/`(구조화 출력), `pd/`·`epd/`(프리필-디코드 분리),
`models/`(모델 구현), `entrypoints/`(HTTP 진입점)

지원 모델 계열: DeepSeek V3 / V3.2 / V4 / V4.1, Kimi K2.5 / K3,
Qwen3 / 3.5 / 4 (비전·오디오·ASR 포함), GLM 5 / 5.3 Flash, Llama,
gpt-oss, MiniMax M3, Inkling, LongCat Flash

**`tokenspeed-kernel/python/tokenspeed_kernel/ops/`**

`attention/` 아래에 `mla/`, `mha/`, `gdn/`, `dsa/`, `kda/`, `qsa/`, `msa/` 등
어텐션 변종이 10종 이상 존재. 그 외 `gemm/`, `moe/`, `quantization/`,
`kvcache/`, `layernorm/`, `embedding/` 등.

**`docs/design/` — 이 프로젝트의 헌법**

| 문서 | 내용 |
| --- | --- |
| `event-loop.md` | 이벤트 루프는 GPU 작업을 만질 수 없다는 제어/데이터 플레인 분리 원칙 |
| `cache-concepts.md` | 논리적 토큰 세계와 물리적 저장소 세계의 계층 분리 |
| `scheduler.md` | 요청 승인(admission)·회수(retraction)·복구 프로토콜 |
| `unified_path.md` | eager 디코드와 CUDA 그래프 디코드가 하나의 경로를 공유 |

`AGENTS.md`에는 "여기 적힌 규칙은 의도적으로 정해진 것이므로, 문서를 같은
변경에서 함께 수정하지 않는 한 위반은 버그다"라고 명시되어 있습니다.

---

## 3. 핵심 아키텍처

PyTorch 재단이 TokenSpeed를 생태계에 편입시키며 언급한 차별점입니다.

> 제어 플레인(control plane)과 실행 플레인(execution plane)을 분리한
> 최초의 오픈소스 LLM 추론 엔진

| 구분 | 언어 | 담당 | 설계 이유 |
| --- | --- | --- | --- |
| 제어 플레인 | C++ | 요청 생명주기, KV 캐시 소유권, 오버랩 타이밍 | 유한상태기계(FSM)로 모델링하고, 타입 시스템으로 자원 안전성을 **컴파일 타임**에 강제 |
| 실행 플레인 | Python | 모델 실행 | 연구자·엔지니어가 빠르게 반복 개발 |

정리하면 **"틀리면 안 되는 부분은 C++로 굳히고, 자주 바뀌는 부분은 Python으로
유연하게"** 라는 설계입니다.

### 추가 설계 원칙 (`AGENTS.md`)

- 스케줄링 경로도 하나, 실행 경로도 하나. 프리필/디코드 분리, 스펙 디코딩,
  CUDA 그래프 여부는 **같은 경로의 파라미터**이지 별도 경로가 아님
- 어텐션에 새 per-request 상태가 필요하면, 백엔드에 상태를 넣기 전에
  LCM 캐시 서브시스템과 C++ 스케줄러가 소유할 수 있는지 먼저 검토
- 커널은 벤더 중립 공개 API 뒤에 격리 (`tokenspeed-kernel`이 유일한 경계)

---

## 4. 설치 및 사용법

### 전제 조건 (`docs/guides/getting-started.md`)

- NVIDIA GPU 호스트
- GPU를 지원하는 Docker
- 충분한 공유 메모리
- 서빙할 모델 체크포인트 접근 권한

### 하드웨어 지원 범위 (`AGENTS.md` 기준)

| 벤더 | 지원 아키텍처 | 대표 제품 |
| --- | --- | --- |
| NVIDIA | `sm90`, `sm100`, `sm103`, `sm107` | H100, B200, GB300 급 |
| AMD | `gfx950`, `gfx1250` | MI355 급 |
| NPU | Ascend (모델 1~2종 한정) | — |

> **중요**: RTX 4090 / 5090 같은 소비자용 GPU는 지원 목록에 없습니다.
> 데이터센터급 GPU가 필요합니다.

### 설치 절차

```bash
# 1) 공식 러너 컨테이너
docker pull lightseekorg/tokenspeed-runner:latest

docker run -itd --shm-size 32g --gpus all \
  --ipc=host --network=host --pid=host --privileged \
  --name tokenspeed lightseekorg/tokenspeed-runner:latest /bin/bash

# 2) 컨테이너 안에서 소스 clone
git clone https://github.com/lightseekorg/tokenspeed.git
cd tokenspeed

# 3) 3개 패키지 설치 (런타임 / 커널 / 스케줄러)
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -e "./python" --no-build-isolation
pip install -e tokenspeed-kernel/python/ --no-build-isolation
pip install -e tokenspeed-scheduler/

# 4) 확인
tokenspeed env
tokenspeed serve --help
```

### 서버 실행

최소 실행:

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1
```

프로덕션 실행 예시 (`docs/guides/launching.md`):

```bash
tokenspeed serve nvidia/Kimi-K2.5-NVFP4 \
  --served-model-name kimi-k2.5 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --quantization nvfp4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --chunked-prefill-size 8192 \
  --max-num-seqs 256 \
  --attention-backend trtllm_mla \
  --moe-backend flashinfer_trtllm \
  --reasoning-parser kimi_k25 \
  --tool-call-parser kimik2
```

### 클라이언트 호출

OpenAI 호환 API이므로 기존 코드를 거의 그대로 재사용할 수 있습니다.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "안녕!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### 주요 튜닝 파라미터 (`docs/configuration/server.md`)

| 파라미터 | 용도 |
| --- | --- |
| `--max-model-len` | 최대 시퀀스 길이 |
| `--kv-cache-dtype fp8` | KV 캐시 정밀도 축소 → 메모리 절감 → 동시 처리량 증가 |
| `--quantization` | 가중치 양자화 (nvfp4 등) |
| `--chunked-prefill-size` | 스케줄러가 한 번에 발행하는 토큰 예산 (기본 8192) |
| `--max-total-tokens` | 전역 토큰 풀 크기 |
| `--tensor-parallel-size` / `--data-parallel-size` | 병렬화 토폴로지 |
| `--enable-expert-parallel` | MoE 전문가 병렬 |
| `--attention-backend` / `--moe-backend` | 커널 백엔드 선택 |
| `--enable-metrics` / `--metrics-reporters prometheus` | 메트릭 수집 |
| `--enable-log-request-stats` | 요청별 성능 통계 로깅 |
| `--kv-events-config` | KV 캐시 변경 이벤트 발행 (ZMQ 등) |

---

## 5. 플러그인·스킬·MCP 여부

**셋 다 아닙니다.**

| 분류 | 정의 | TokenSpeed |
| --- | --- | --- |
| 플러그인 | 다른 프로그램에 끼우는 확장 부품 | 아니오 |
| 스킬 | 에이전트에게 작업 방법을 알려주는 문서 | 아니오 |
| MCP | 에이전트가 외부 도구를 쓰게 하는 연결 규격 | 아니오 |
| **독립 서버 소프트웨어** | 단독으로 구동되는 서버 데몬 | **예** |

TokenSpeed는 nginx나 MySQL과 같은 급의 **독립 실행형 서버**입니다.

### 혼동하기 쉬운 지점

저장소 안에 다음 파일들이 존재합니다.

```
CLAUDE.md                            # 에이전트용 규칙
AGENTS.md                            # 에이전트용 규칙
.skills/bisect-triton-release.md     # 스킬 파일
.skills/optimize-amd-gpu-kernel.md   # 스킬 파일
```

이들은 **TokenSpeed를 개발할 때** 코딩 에이전트가 참고하는 지침입니다.
**TokenSpeed를 사용하기 위한** 스킬이 아닙니다.
(주방에 붙은 "직원용 위생 수칙"이지, 손님용 메뉴판이 아님)

### 다만 MCP로 감쌀 수는 있음

TokenSpeed 서버를 띄운 뒤, 그것을 호출하는 MCP 서버를 **별도로 작성**하면
에이전트가 자체 호스팅 모델을 사용할 수 있습니다. 단, 그 MCP 서버는
직접 만들어야 하며 저장소에 포함되어 있지 않습니다.

---

## 6. API 토큰 필요 여부

**외부에 비용을 지불하는 API 토큰은 필요하지 않습니다.**

```
OpenAI 사용 시   : 내 서버 → 외부 API → 토큰 단위 과금
TokenSpeed 사용 시: 내 서버 → 내 GPU에서 직접 연산 → 토큰 과금 없음
```

모델을 자체 하드웨어에서 구동하므로 토큰당 요금이 발생하지 않습니다.
대신 GPU 장비 비용과 전력 비용이 발생합니다.

### 문서에 등장하는 두 종류의 "키"

**1) `--api-key`** (`docs/configuration/server.md`)

> SMG gateway API key for authorization with upstream workers.

내 서버에 아무나 접근하지 못하게 막는, **내가 직접 정하는 인증 키**입니다.
외부 결제와 무관합니다. 공식 클라이언트 예제가 `api_key="EMPTY"`인 것에서
기본값은 인증 없음임을 알 수 있습니다.

**2) Hugging Face 토큰**

모델 가중치를 내려받을 때 필요할 수 있습니다(게이트된 모델의 경우).
무료 계정 토큰입니다.

---

## 7. GitHub에서 주목받는 이유

> 별(star) 수치는 네트워크 제약으로 직접 확인하지 못했습니다.
> 아래는 저장소 내부 근거를 기반으로 한 분석입니다.

### (1) Day 0 모델 지원 전략 — 가장 큰 요인

`README.md`의 News 섹션에서 확인되는 패턴입니다.

| 시점 | 모델 | 비고 |
| --- | --- | --- |
| 2026/08 | Qwen3.8 (2.4T 규모) | 출시 당일 지원 |
| 2026/08 | GLM 5.3 Flash | 출시 당일 지원 |
| 2026/07 | Kimi K3 | 출시 당일 지원 |
| 2026/07 | TML Inkling (FP4) | 출시 당일 지원 |

신규 모델이 공개되면 Hugging Face 모델 카드의 "배포 방법" 항목에
TokenSpeed 링크가 함께 실립니다. 해당 모델을 쓰려는 전 세계 사용자가
자연스럽게 이 저장소로 유입되는 구조입니다.

### (2) PyTorch 재단 공식 인정

PyTorch 공식 블로그에 4회 이상 등장했고, "제어 플레인과 실행 플레인을 분리한
**최초의**" 엔진이라는 서술이 붙었습니다.

### (3) 공개된 성능 기록

2026/05 기준 Qwen3.5-397B-A17B 에이전틱 워크로드에서 580 TPS 달성이
PyTorch 블로그로 공개되었습니다.

### (4) 비영리 중립 포지션

상업 회사가 아닌 LightSeek Foundation 소속이라 NVIDIA와 AMD 양쪽의
지원을 동시에 받습니다. README는 이를 "오픈소스 LLM 추론 엔진의 스위스"라고
표현합니다.

### (5) 대형 실장비 CI

`.github/workflows/` 에서 확인됩니다.

```
gb300-slurm-nightly.yml       # GB300 나이트리
gb300-slurm-per-commit.yml    # 커밋 단위 GB300 테스트
b300-deepswe.yml
pr-test-amd.yml
pr-test-nvidia-arm.yml
```

커밋마다 GB300급 장비로 검증한다는 것은 프로젝트 뒤에 상당한 하드웨어
지원이 있다는 신호입니다.

---

## 8. 로컬 에이전트 구축에 도움이 되는가

### 직접적으로는: 적합하지 않음

개인 PC/노트북에서는 구동할 수 없습니다(데이터센터급 GPU 요구).
로컬 에이전트에는 다음이 적합합니다.

| 도구 | 특징 |
| --- | --- |
| Ollama | 설치·사용이 가장 쉬움 |
| LM Studio | GUI 기반 |
| llama.cpp | 가볍고 CPU 구동 가능 |
| vLLM | 소형 모델 + 준수한 GPU면 가능 |

### 간접적으로는: 매우 유용

**(1) 무중단 전환 설계가 가능해집니다**

Ollama도 TokenSpeed도 OpenAI 호환 API를 제공하므로, 처음부터 OpenAI 호환
방식으로 설계해 두면 나중에 엔드포인트 한 줄만 교체하면 됩니다.

```python
# 초기 (로컬, 비용 0)
base_url = "http://localhost:11434/v1"       # Ollama

# 확장기 (자체 GPU 서버)
base_url = "http://my-gpu-server:8000/v1"    # TokenSpeed
```

이는 AI 서비스의 비용 구조를 **사용량 비례 변동비 → 고정비**로 전환시키는
핵심 설계입니다.

**(2) 성능 병목의 원리를 이해하게 됩니다**

| 개념 | 왜 중요한가 |
| --- | --- |
| Prefix Cache | 동일한 프롬프트 앞부분 재계산 생략. 에이전트는 같은 시스템 프롬프트를 반복하므로 효과가 큼 |
| KV Cache | 대화가 길어질수록 메모리를 점유. 동시 사용자 수의 실질적 상한을 결정 |
| Speculative Decoding | 작은 모델이 먼저 추측하고 큰 모델이 검증하여 지연 단축 |
| Chunked Prefill | 긴 입력을 잘라 처리해 다른 요청의 지연을 방지 |

이 개념들은 어떤 엔진을 쓰든 동일하게 적용됩니다.

---

## 9. 수익화 아이디어

### 9.1 먼저: 이 시장의 돈 흐름

```
① 칩 제조사 (NVIDIA)             수익 매우 큼
② 클라우드 (AWS, CoreWeave 등)    수익 큼
③ 엔진 SW (TokenSpeed, vLLM)     0원 (무료 배포)
④ 구축·운영 (사람의 기술)          희소성 기반 수익
⑤ 최종 앱 (문제 해결 서비스)        수익 큼
```

**③번은 수익 구간이 아닙니다.** 비영리 재단이 의도적으로 무료 배포합니다.
따라서 공략 지점은 **④ 또는 ⑤** 입니다.

- ④: 자본 없이 시작 가능, 현금 흐름이 빠름
- ⑤: 시간이 오래 걸리지만 상한이 높음

### 9.2 아이디어 1 — LLM 서빙 성능 감사(Audit)

| 투자금 | 회수 기간 | 난이도 | 현실성 |
| --- | --- | --- | --- |
| 거의 0원 | 2~3개월 | 중 | **최상** |

이미 GPU를 보유하고 기본 설정으로 운영 중인 기업이 다수입니다.
동일 장비에서 처리량을 2배로 올리면, 고객 입장에서는 GPU를 추가 구매한 것과
같은 효과입니다.

판매하는 기술 항목:

- `--kv-cache-dtype fp8` 적용 → 메모리 절감 → 동시 처리량 증가
- `--quantization` 전략 수립
- `--chunked-prefill-size` 조정 → 긴 프롬프트 안정화
- `--tensor-parallel-size` / `--enable-expert-parallel` 토폴로지 최적화
- `--attention-backend` / `--moe-backend` 선택 (기본값이 최적이 아닌 경우 다수)
- Speculative decoding 도입 판단
- Prefix cache 적중률 개선 (에이전트 워크로드에서 효과가 큼)

가격 구조:

```
1단계: 무료 진단 (반나절)   → 현재 처리량 측정 + 개선 여지 리포트
2단계: 유료 최적화 (2주)    → 300~800만 원
3단계: 성과 기반 보너스      → 목표 처리량 달성 시 추가
```

시작 방법:

1. 클라우드 GPU를 시간 단위로 임대
2. 동일 모델을 기본 설정 vs 튜닝 설정으로 벤치마크
3. "설정 N개 변경으로 처리량 N배" 기록을 공개
4. 이 기록이 포트폴리오이자 영업 자료가 됨

### 9.3 아이디어 2 — 온프레미스 LLM 구축 대행

| 투자금 | 회수 기간 | 난이도 | 현실성 |
| --- | --- | --- | --- |
| 0원 (장비는 고객 부담) | 3~6개월 | 중상 | 상 |

국내에서 외부 AI API를 쓸 수 없는 산업이 명확히 존재합니다.

| 산업 | 제약 사유 |
| --- | --- |
| 금융 | 망분리 규제 |
| 의료 | 환자 데이터 외부 전송 제한 |
| 공공 | 보안 인증 요구 |
| 제조 | 설계·공정 데이터 기밀 |
| 법무 | 의뢰인 비밀유지 의무 |

제공 패키지:

1. 요구사항 분석 (동시 사용자, 문서 길이, 지연 목표)
2. 하드웨어 사양 산정
3. 설치·구성 (Docker + 엔진 + 모델 배포)
4. 성능 튜닝 (9.2의 기술 재사용)
5. 모니터링 구축 — `--enable-metrics`, `prometheus`, `--kv-events-config` 활용
6. 인수인계 교육 및 매뉴얼
7. 유지보수 계약

가격 구조:

| 항목 | 금액 |
| --- | --- |
| 초기 구축 | 1,000~3,000만 원 |
| 월 유지보수 | 100~300만 원 |

초기 구축비보다 **유지보수 계약의 반복 수익**이 사업 안정성의 핵심입니다.

### 9.4 아이디어 3 — 한국어 기술 콘텐츠 선점

| 투자금 | 회수 기간 | 난이도 | 현실성 |
| --- | --- | --- | --- |
| 0원 | 즉시~3개월 | 하 | **최상** |

이 저장소의 문서는 전부 영어이며, 국내 자료가 희소합니다.
또한 이 분야는 주기가 짧아 기존 자료가 빠르게 낡습니다.

콘텐츠 라인업:

- 입문: "vLLM vs SGLang vs TokenSpeed 비교", "GPU 없이 서빙 공부하기"
- 중급: "설정 변경만으로 처리량 N배 만든 실측 기록", "KV 캐시와 동시 접속자 한계"
- 고급: "제어/실행 플레인 분리 아키텍처 해부", "Speculative Decoding 실효과"

수익 경로:

| 경로 | 성격 |
| --- | --- |
| 블로그 광고 | 소액 |
| 온라인 강의 | 중간 |
| 기업 세미나 | 건당 100~300만 원 |
| **컨설팅 문의 유입** | **실제 목적** |

콘텐츠 자체보다 **전문성 증명 수단**으로서의 가치가 큽니다.

### 9.5 아이디어 4 — 버티컬 AI 서비스 (상한이 가장 높음)

| 투자금 | 회수 기간 | 난이도 | 현실성 |
| --- | --- | --- | --- |
| 중 | 6~18개월 | 상 | 중 |

사용자는 "AI"가 아니라 **자신의 문제 해결**에 비용을 지불합니다.

구조:

```
[React 프론트엔드]
      ↕
[백엔드 (PHP/Node 등)]
      ↕
[Ollama → 성장 후 TokenSpeed]
```

초기에는 로컬/저비용 엔진으로 시작하고, 사용자가 늘어 API 비용이 변동비로
누적되기 시작하면 자체 서빙으로 전환해 고정비 구조로 바꿉니다.

### 9.6 아이디어 5 — 오픈소스 기여를 통한 커리어 자산

`CONTRIBUTING.md`가 명시적으로 환영하는 항목:

- 명백한 버그 수정
- 작고 검증 가능한 프로덕션 필요 기능
- 기존 스타일에 맞는 성능 최적화
- 문서, 툴링, 벤치마킹 개선

동시에 다음도 명시되어 있습니다.

> 핵심 기능은 코어 팀이 설계하고 구현하며, 외부 기여에 의도적으로 선별적이다.
> 큰 기능이나 아키텍처 변경은 구현 전에 RFC/설계 논의부터 시작할 것을 권장한다.

따라서 전략은 **문서·벤치마크·버그 리포트부터 시작**하는 것입니다.

### 9.7 아이디어 6 — 서빙 모니터링 대시보드

TokenSpeed는 이미 메트릭을 방출하지만, 시각화는 사용자 몫입니다.

시각화 대상: 실시간 처리량/지연, KV 캐시 사용률 경고, GPU 사용률,
사용자별 토큰 사용량(사내 과금), Prefix 캐시 적중률.

단독 제품보다는 **9.3의 구축 패키지 구성 요소**로 포함하는 편이 현실적입니다.

### 9.8 비권장 — 범용 추론 API 재판매

| 문제 | 설명 |
| --- | --- |
| 경쟁 | 대형 사업자들이 원가 근처로 판매 |
| 마진 | 사실상 0에 수렴 |
| 리스크 | GPU는 유휴 상태에서도 과금 |
| 전환비용 | 고객이 엔드포인트 한 줄로 이탈 가능 |

단, **규제 준수를 파는 좁은 니치**(국내 리전, 데이터 미저장 보장, 특정 도메인
특화)라면 가격이 아닌 요건으로 경쟁할 수 있어 마진 확보가 가능합니다.

### 9.9 피해야 할 함정

| 함정 | 이유 |
| --- | --- |
| GPU를 먼저 구매 | 고객 확보 전 고정비 발생 |
| 엔진 자체를 판매 | 무료 오픈소스 |
| 범용 API 재판매 | 자본 경쟁 |
| "AI 도입 컨설팅"으로 뭉뚱그림 | 차별화 불가. 숫자로 증명해야 함 |
| 실적 없이 컨설팅부터 시작 | 첫 프로젝트 실패 시 평판 손실 |

### 9.10 핵심 명제

> 판매 대상은 TokenSpeed가 아니라 **"동일한 GPU에서 더 많은 처리량을 뽑아내는
> 능력"** 이며, 그 능력을 증명하는 유일한 방법은 **실측 벤치마크 기록**이다.

---

## 10. React·PHP로 만들 수 있는가

### 10.1 TokenSpeed 자체를 React/PHP로 재구현 — 불가능

| 필요 요소 | React | PHP | 사유 |
| --- | --- | --- | --- |
| GPU 메모리 직접 제어 | 불가 | 불가 | 브라우저/웹서버 런타임의 영역 밖 |
| CUDA·Triton 커널 실행 | 불가 | 불가 | C++/CUDA 전용 |
| 수십 GB 단위 메모리 관리 | 불가 | 불가 | 용도 자체가 다름 |
| 고성능 행렬 연산 | 불가 | 불가 | PyTorch/CUDA 스택 필요 |

실제 저장소 구성: Python 약 1,520개 파일, C++/CUDA 약 160개 파일.
React/PHP 코드는 없습니다.

### 10.2 TokenSpeed를 사용하는 앱을 React/PHP로 개발 — 완전히 가능

이것이 정상적인 사용 방식입니다.

```
[React 화면] 또는 [PHP 웹앱]   ← 직접 개발하는 영역
        ↕ HTTP
[TokenSpeed 서버]              ← 가져다 쓰는 영역
        ↕
[GPU 연산]
```

**React 예시**

```jsx
async function askAI(question) {
  const res = await fetch("http://<서버>:8000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "kimi-k2.5",
      messages: [{ role: "user", content: question }],
    }),
  });
  const data = await res.json();
  return data.choices[0].message.content;
}
```

`openai` npm 패키지를 그대로 쓰고 `baseURL`만 교체해도 됩니다.

**PHP 예시**

```php
<?php
$ch = curl_init("http://<서버>:8000/v1/chat/completions");
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
    "model" => "kimi-k2.5",
    "messages" => [["role" => "user", "content" => "안녕!"]],
]));
$result = json_decode(curl_exec($ch), true);
echo $result["choices"][0]["message"]["content"];
```

---

## 11. 실행 로드맵

### 0~3개월: 증명 단계 (투자 최소)

| 기간 | 활동 |
| --- | --- |
| 1~2주차 | Ollama로 로컬 실습. prefix cache / KV cache 개념 체득 |
| 3~4주차 | 클라우드 GPU 시간 단위 임대. vLLM·TokenSpeed 실제 구동 |
| 5~8주차 | 벤치마크 실험 및 기록. "설정 변경 → N배" 데이터 확보 |
| 9~12주차 | 기술 블로그 3~5편 발행, PR 1~2건 기여 |

목표: 매출 0원, 대신 **증명 자료 확보**

### 3~6개월: 첫 매출 단계

- 무료 성능 진단을 5곳에 제안 → 1~2곳 유료 전환
- 동시에 버티컬 앱 MVP를 React로 개발

목표: 첫 계약 + 레퍼런스 1건

### 6~12개월: 확장 단계

- 구축 대행으로 단가 상승
- 유지보수 계약으로 고정 수입 확보
- 버티컬 앱 사용자 유입 시작

목표: 월 고정 수입 + 자산형 제품 병행

### 이번 주에 할 일

1. Ollama 설치 및 로컬 실습 (비용 0)
2. 처리량/지연 측정 벤치마크 스크립트 작성
3. 기술 블로그 1편 초안

---

## 12. 참고 링크

### 저장소

- 원본(upstream): https://github.com/lightseekorg/tokenspeed
- 이 저장소(fork): https://github.com/bmshin94/tokenspeed

### 공식 문서

- 문서 색인: https://lightseek.org/tokenspeed/
- 시작하기: https://lightseek.org/tokenspeed/guides/getting-started
- 서버 실행: https://lightseek.org/tokenspeed/guides/launching
- 모델 레시피: https://lightseek.org/tokenspeed/recipes/models
- 서버 파라미터: https://lightseek.org/tokenspeed/configuration/server
- 호환 파라미터: https://lightseek.org/tokenspeed/configuration/compatible-parameters
- 병렬화: https://lightseek.org/tokenspeed/serving/parallelism

### 재단 및 블로그

- LightSeek Foundation: https://lightseek.org/
- 기술 블로그: https://lightseek.org/blog/
- 스폰서·파트너: https://lightseek.org/sponsors

### 저장소 내 필독 문서

| 경로 | 내용 |
| --- | --- |
| `README.md` | 프로젝트 개요와 차별점 |
| `AGENTS.md` | 개발 규칙, 설계 원칙, 하드웨어 지원 범위 |
| `CONTRIBUTING.md` | 외부 기여 정책 |
| `GOVERNANCE.md` | 거버넌스 및 코어 메인테이너 |
| `docs/design/event-loop.md` | 이벤트 루프 설계 원칙 |
| `docs/design/cache-concepts.md` | 캐시 개념 계층 |
| `docs/design/scheduler.md` | 스케줄러 승인·회수·복구 |
| `docs/design/unified_path.md` | 통합 디코드 경로 |
| `docs/configuration/server.md` | 전체 서버 파라미터 |

### 인용

```bibtex
@misc{tokenspeed2026,
  author       = {{TokenSpeed Team}},
  title        = {{TokenSpeed}: A Speed-of-Light {LLM} Inference Engine},
  year         = {2026},
  howpublished = {\url{https://github.com/lightseekorg/tokenspeed}}
}
```
