# IGRIS-C NEPES Dataset

CRS를 카트에서 꺼내 JIG에 적재하는 **수집·증강·상태 replay** 배포본입니다.
HugWBC, 자유 골반, 접촉 기반 2지 파지를 사용합니다. 손바꿈 없이 2→2는 왼팔,
4→4는 오른팔로 수행합니다. 슬롯 번호는 세척기 쪽 3인칭 화면 기준입니다.

> 이 배포본은 **Pink IK + 동시 수집** 버전입니다. 본 수집 전에 경로별 1개씩 smoke test를 실행하세요.

## 설치

Linux x86_64, Python 3.11, NVIDIA 드라이버/Vulkan 및 RTX 렌더링 지원 GPU가 필요합니다.
검증 스택: Isaac Sim 5.1.0, IsaacLab 2.3.2, RTX 5080 16GB.
SDK·가상환경은 포함하지 않습니다. 신규 설치에는 인터넷과 SDK 저장 공간이 필요합니다.

```bash
cd igris_nepes_dataset
git lfs install    # Git clone으로 받은 경우
git lfs pull       # ZIP에는 실제 에셋이 포함되어 있음
python3 scripts/check_release.py
bash scripts/setup.sh --python python3.11
source .venv/env.sh
```

이미 호환되는 Isaac 환경이 있으면 설치 명령을 다음으로 대체합니다.
기존 Isaac 환경은 수정하지 않고 별도 데이터셋 환경을 만듭니다.

```bash
bash scripts/setup.sh --sim-python /path/to/isaac_env/bin/python
source .venv/env.sh
```

신규 SDK는 수집 전에 터미널에서 한 번 실행해 약관 안내를 읽고 동의 여부를 입력합니다.
승인 기록은 해당 SDK에 저장됩니다.

```bash
"$NEPES_SIM_PYTHON" -c 'import isaacsim'
```

<!-- BEGIN SECTION: Pink IK backend -->
## Pink IK 백엔드 (이 배포본의 기본값)

### 무엇이 바뀌었나

수집 중 허리 3축과 작업 팔 7축 명령을 계산하는 온라인 IK(`UpperBodyServo`)의 수치 계산을
Pinocchio(FK·Jacobian)와 Pink(`pink.solve_ik`, quadprog QP)로 수행합니다.
목표·제약·자세 유도·feedforward·허리 고정, 이전 명령을 기준으로 적분하는 방식은 그대로입니다.
바뀌는 것은 계산 라이브러리뿐이며, 경로·성공 기준·HugWBC·카메라는 같습니다.

- Pinocchio/Pink는 Isaac Python에 설치하지 않습니다. 별도 Python 3.11 환경(`<env-root>/ik`)의
  작업자 `ik/pink/worker.py`가 **Isaac 프로세스마다 1개** 실행되고 표준입출력(JSON lines)으로 통신합니다.
  포트·임시 이름·공유 파일을 쓰지 않으므로 여러 Isaac 프로세스를 동시에 실행해도 서로 간섭하지 않습니다.
- 작업자는 런타임 `upper_body_control.py`의 연결 코드(hook) **앞부분, 즉 기존 solver 코드를 그대로** 실행합니다.
  이 앞부분의 SHA-256이 검증된 운영본과 같다는 것을 테스트가 확인합니다. 생성자·명령 범위 등 나머지는 Isaac 쪽 기존 코드입니다.
- 코드 출처와 변경점은 `ik/pink/provenance.json`, 테스트 기록 데이터의 출처는 `tests/ik_pink/fixtures/provenance.json`에 있습니다.

### 왜 안전한가

- **CPU 수치 검증:** 실제 성공 에피소드에서 추출한 50개 상태(양팔 5단계 × 5)를 1,500회 계산했을 때
  기존 solver와의 명령 차이는 최대 `2.2e-16 rad`(Pink), 명령 경계 위반은 0건이었습니다.
  이 배포본의 `tests/ik_pink`는 같은 50개 기록 상태에서 기록된 실제 명령을 다시 재현하는지 검사합니다(허용 오차 1e-9 rad).
- **실제 Isaac A/B(2026-10-30, seed 10000):** 오른손 4→4·왼손 2→2를 기존 방식과 Pink로 한 번씩, 4회 모두 성공·검증을 통과했습니다.
  Parquet의 모든 열(관절 상태·액션·시간·단계)이 완전히 같았고 프레임 수도 같았습니다(2,737 / 4,316).
  전체 수집 시간은 Pink가 0.7~3.8% 짧았지만 조건당 1회라 속도 개선으로 단정하지 않습니다.
- **비트 단위 동일성은 아님:** 기존 solver와의 차이는 약 1e-16 rad이며 비트까지 같은 경우는 일부입니다.
  A/B에서 Parquet이 완전히 같았던 것은 float32 저장에서 이 차이가 사라졌기 때문이므로, 다른 seed에서도 항상 같다고 단정하지 않습니다.
- **한계:** 같은 목표·제약을 풀기 때문에 새로운 회피 경로나 성공률 향상은 기대하지 않습니다.
  A/B는 실험 묶음으로 실행했고, 이 배포본 자체는 CPU 테스트(Isaac Python에서 Kit 없이 작업자 호출 포함)로 검증했습니다.
  새 PC에서는 아래처럼 먼저 소량을 수집해 확인하세요.

### 설치와 IK 확인

```bash
cd igris_nepes_dataset                            # ZIP을 빈 새 폴더에 푼 배포본 폴더
python3 scripts/check_release.py                  # 파일 무결성 (IK 환경이 아직 없으면 건너뜀)
bash scripts/setup.sh --python python3.11          # 신규: sim + dataset + ik 환경
# 기존 Isaac 환경 재사용: bash scripts/setup.sh --sim-python /path/to/isaac_env/bin/python
source .venv/env.sh                               # NEPES_SIM/DATASET/IK_PYTHON (절대 경로)
"$NEPES_SIM_PYTHON" -c 'import isaacsim'          # 신규 SDK만: 약관(EULA) 안내를 읽고 직접 1회 동의
python3 scripts/check_release.py --require-ik     # IK 환경 버전·import·Pink QP 확인 (EULA 미동의도 경고)
"$NEPES_DATASET_PYTHON" -B -m unittest discover -s tests   # CPU 회귀(IK 포함)
```

아래 "수집부터 증강까지"의 경로별 1개 예시로 설치를 확인한 뒤 본 수집을 진행합니다.

- EULA 동의 기록이 없으면 첫 시도가 화면에 보이지 않는 약관 질문에서 멈춰 시도 제한 시간(5,400초)까지 기다립니다.
  그래서 batch는 시작 전에 이를 확인하고, 동의 기록이 없으면 아무것도 만들지 않고 멈춥니다.

- `setup.sh`는 `<env-root>/ik`에 `requirements/ik_pink.txt`(pin 3.8.0, pin-pink 4.4.0, quadprog 0.1.14,
  numpy 2.3.5 등 21개 고정)를 `--no-deps --only-binary=:all:`로 설치한 뒤 `pip check`와
  `ik/pink/worker.py --check`로 확인합니다. IK 패키지는 Isaac·dataset Python에 설치하지 않습니다.
  PyPI 접속이 필요하고, 컴파일러는 필요 없습니다(manylinux_2_28 wheel, glibc 2.28 이상).
- 이미 sim/dataset 환경이 있는 env-root에 IK 환경만 추가할 때:
  `bash scripts/setup.sh --ik-only --sim-python /path/to/isaac_env/bin/python --env-root /path/to/.venv`
  (env.sh에는 `NEPES_IK_PYTHON` 줄만 추가·갱신). `--dry-run`은 설치 계획만 출력합니다.
- 수집·증강·campaign 명령은 기존과 같습니다. **`source .venv/env.sh` 후 실행하세요.**
  `NEPES_IK_PYTHON`이 없으면 배포본의 `.venv/ik/bin/python`을 찾습니다.
- **batch 사전 점검:** 수집 batch(`collect.py`, `pipeline.py`·campaign의 수집 단계)와 렌더링 증강은 batch 출력 폴더를 만들거나
  Isaac을 띄우기 전에, 런타임과 같은 규칙으로 IK 백엔드·IK Python을 정하고 `worker.py --check`(고정 버전·import·Pink QP)와
  Isaac EULA를 확인합니다. `collect.py --dry-run`도 같은 점검을 하며, 문제가 있으면 원인을 출력하고 아무것도 만들지 않고 멈춥니다.
  `pipeline.py`·`campaign.py`는 자기 출력 폴더를 먼저 만들고(campaign은 기존 데이터 재검증도 먼저 함) 수집 단계에서 이 점검을 합니다.
  이 둘의 `--dry-run`은 계획·명령만 출력하고 IK·EULA는 점검하지 않으므로, 새 환경에서는 같은 환경에서 `collect.py … --dry-run`을 먼저 실행하세요.
- `NEPES_IK_PYTHON`은 절대 경로여야 합니다(env.sh는 절대 경로를 씁니다). Isaac 시도는 배포본 폴더에서 시작하므로
  상대 경로는 사전 점검·고정한 것과 다른 인터프리터를 가리킬 수 있어 `collect.py`·`pipeline.py`·`campaign.py`가 거부합니다.
- 확인된 백엔드·IK Python 경로·버전·패키지 버전·작업자 소스 해시는 batch 계획(`batch.json`의 `plan.upper_body_ik`,
  `--dry-run` 출력)에 고정됩니다. 다른 백엔드나 다른 IK 환경으로 `--resume`하면 거부됩니다.
- 렌더링 증강도 물리를 다시 실행하므로 같은 Pink 작업자를 사용하고, 같은 사전 점검과 계획 고정을 거칩니다.

### 백엔드 전환(디버깅)

`NEPES_IK_BACKEND`를 비우면 `pink`(기본)입니다. `native`는 기존 프로세스 안 NumPy/SciPy solver이며, 그 밖의 값은 오류입니다.

- `native`는 **단일 실행 디버깅 전용**입니다. batch 도구(collect·pipeline·campaign 수집·렌더링 증강)는 `native`를 거부하고,
  검증(`merge_seeded.py`)도 이 배포본의 백엔드(`pink`)로 끝난 시도만 성공·물리 실패로 인정하므로 한 batch에 두 solver가 섞이지 않습니다.
  기존 solver로 데이터셋을 만들려면 이전 배포본을 사용하세요.
- 단일 시도 디버깅은 `collect.py --dry-run`이 출력하는 `examples[].command`를 배포본 폴더에서
  `NEPES_IK_BACKEND=native`와 수집기와 같은 `PYTHONPATH`(`runtime/<profile>/source/robros_motion_planning:runtime/<profile>/source/robros_lab`)로
  직접 실행합니다. 결과는 별도 폴더에 두고 batch에 넣지 않습니다.
- 실제 사용한 백엔드는 각 실행 `run.json`의 `upper_body_ik_backend`에 기록됩니다:
  선택 방법, 작업자 Python 경로·버전, 설치 패키지 버전, 작업자·연결 코드·기존 solver의 SHA-256, URDF 해시,
  최종 상태(`closed`/`failed`)와 작업자 종료 코드. 같은 내용(시작 시점, `ready`)이 데이터셋 에피소드 provenance에도 들어가며,
  Pink이면 trace의 IK 단계마다 `ik_backend` 표시가 남습니다.

### 실패 정책

- 작업자 시작 실패(인터프리터 없음, 패키지 버전·소스 불일치), 시간 초과(시작 120초, 요청 30초),
  작업자 종료, 원격 오류, 잘못된 결과는 모두 `IKBackendError`로 **그 시도를 실패**시킵니다.
  기존 solver로 자동 전환하지 않고, 실패한 작업자를 다시 띄우지도 않습니다.
- 원인은 `run.json`(`error` 또는 에피소드 `reason`)과 `upper_body_ik_backend.status/error`에 남습니다.
  `run.json`의 이 항목은 에피소드가 끝날 때와 작업자를 닫은 뒤 다시 기록되므로 최종 상태와 종료 코드(`worker.returncode`)가 남고,
  작업자 stderr는 해당 시도 로그에 함께 기록됩니다.
- 수집기는 이를 물리 실패가 아닌 프로세스 오류로 분류해 batch를 `needs_review`로 멈춥니다(기존 프로세스 오류와 같은 정책).
  env.sh 미적용·IK 환경 손상·EULA 미동의는 batch 시작 전 사전 점검에서 걸러지므로, 이 중단은 주로 실행 중 작업자 이상입니다.
  원인을 고치고 `check_release.py --require-ik`로 확인한 뒤, 겹치지 않는 seed 범위의 새 `--output`으로 다시 시작합니다.
- 성공 시도와 재시도 가능한 물리 실패는 `run.json`에 이 배포본의 백엔드가 `closed`(정상 종료)로 기록되고 작업자 소스·requirements 해시가
  배포본과 같을 때만 인정됩니다. 성공 시도는 에피소드 provenance의 백엔드·작업자 PID도 `run.json`과 같아야 합니다.
- 정상 종료·예외·SIGTERM·SIGKILL 때 작업자도 함께 끝납니다(stdin EOF, 소유 프로세스 감시).
  SIGINT는 무시하므로 수집기의 정상 정지 절차가 끝날 때까지 IK 계산이 유지됩니다.

### 테스트

위의 문서화된 명령이 IK 연결 테스트를 포함합니다. `NEPES_IK_PYTHON`(또는 `.venv/ik`)이 있으면
`tests/ik_pink`(기록 상태 50개 동일성, Pinocchio·Pink 검증, 작업자 통신)를 IK Python에서 실행하고,
Isaac과 같은 NumPy 1.26 Python에서 작업자를 동시에 여러 개 띄우는 검사와 종료·충돌 검사, `run.json` 최종 상태 기록,
batch 사전 점검 결과 고정 검사도 실행합니다. `tests/ik_pink`에는 Pink 백엔드가 실제로 `pink.solve_ik`만 호출하고
SciPy BVLS는 호출하지 않는지 확인하는 검사가 있습니다.
IK 환경이 없으면 이 검사들은 이유를 표시하고 건너뜁니다. IK 묶음만 직접 실행할 수도 있습니다.

```bash
"$NEPES_IK_PYTHON" -B -m unittest discover -s tests/ik_pink -t tests/ik_pink
```
<!-- END SECTION: Pink IK backend -->

## 기존 데이터 포함 200 + 100 + 100

`scripts/campaign.py`는 검증된 기존 원본을 합산하고 부족한 수량만 수집합니다.
왼손 2→2·오른손 4→4 각각 100개를 채운 뒤, 새 원본 중 좌우 50개씩을 골라
밝기 변환 100개와 조명·재질 증강 100개를 별도 저장합니다.
기존 수집은 원래 버전의 검증 코드로 재검증하며 같은 상태·액션이 반복된 데이터는 중복 집계하지 않습니다.

```bash
python3 scripts/campaign.py --imports /path/to/import_candidates.json \
  --base-per-pair 100 --augmentation-episodes 100 --first-seed "$FIRST_SEED" \
  --output ../runs/base200_aug200
# 같은 명령에 --resume: 완료분 검증 후 재개
python3 scripts/campaign.py --output ../runs/base200_aug200 --status
# 안전하게 정지: 현재 프로세스가 출력과 자식 프로세스를 정리하도록 요청
touch ../runs/base200_aug200/STOP
```

`FIRST_SEED`는 `scripts/import_candidates.py`가 출력하는 `suggested_first_seed` 이상이어야 하고,
가져온 batch를 만든 캠페인이 아직 쓸 수 있는 시드 범위와도 겹치지 않아야 합니다.
`--dry-run`은 부족 수량과 명령만 표시합니다(IK·EULA 점검은 `collect.py --dry-run`). `--stop-after-new-successes 2`는
새 성공 2개에서 검토용으로 일시 정지합니다. 계속하려면 이 인자를 빼고 `--resume`합니다.
`import_candidates.json`은 원본 batch·bundle·attempt·검증 해시를 묶은 입력 목록이므로,
원본 데이터와 원래 검증 코드도 접근 가능한 위치에 보관해야 합니다.

```text
base200_aug200/
  campaign.json                         전체 진행 상태
  new_base/                             새 물리 실행·MP4·압축 replay 기록
  basic_200/                            기존+신규 원본, LeRobot v3
  augmentation/video/visible_darker_v2/  밝기·대비 증강 100개
  augmentation/render/dataset_100/      조명·재질 증강 100개
  datasets.json                         모두 완료되면 최종 경로·수량
```

기본 밝기 증강은 변화가 확실한 `darker` 프리셋입니다. `--video-preset brighter`로
새 실행의 프리셋을 바꿀 수 있습니다. 렌더링 증강은 밝은 조명·매끈한 재질과
어두운 조명·거친 재질을 적용하고 물리를 다시 실행해 성공 여부를 확인합니다.
이전 시험용 증강은 보존하고, 이번 증강 수량은 선택한 새 부모 100개로 구성합니다.
동일 부모와 모든 파생본은 `split_group`으로 묶어 학습/평가 누수를 막으세요.

## 수집부터 증강까지

출력은 배포 디렉터리 **밖**에 저장합니다. 먼저 각 경로 1개씩 확인하세요.

```bash
# 원본 2개 + 밝기 변환 증강 2개
python3 scripts/pipeline.py --per-pair 1 --first-seed 1000 \
  --augmentation-mode video --output ../runs/smoke

# 원본 50개 + 영상 증강 25개 + 조명·재질 증강 25개
python3 scripts/pipeline.py --per-pair 25 --first-seed 1000 \
  --augmentation-mode both --augmentation-episodes 25 \
  --output ../runs/pilot50
```

`--dry-run`은 명령·예정 수량만 표시합니다. 같은 명령에 `--resume`을 붙이면 이어서 수행합니다.
진행 상태는 `python3 scripts/collect.py --output ../runs/pilot50/collection --status`로 확인합니다.
성공분만 채택하고 MP4·Parquet 검증 후 임시 PNG를 정리합니다.
정상 물리 실패는 재시도하며 연속 실패 횟수로 멈추지 않습니다. 기본 최대 시도는 경로별 목표의 3배입니다.
`--max-attempts-per-pair`로 조절할 수 있습니다. 손상 파일·프로세스 오류는 검토를 위해 중단합니다.

수집만 실행하거나, 완료된 수집에 증강만 적용할 수도 있습니다.

```bash
python3 scripts/collect.py --per-pair 25 --first-seed 1000 --output ../runs/source50
python3 scripts/augment.py --source-batch ../runs/source50 --output ../runs/video50 \
  --mode video --video-presets brighter
python3 scripts/augment.py --source-batch ../runs/source50 --output ../runs/render50 \
  --mode render
```

- **시작 배치:** 시드마다 골반 초기 XY를 왼팔 ±3 mm, 오른팔 ±5 mm 범위에서 바꿉니다.
  런타임 IK·제어기를 실행하므로 관절 상태와 액션도 새로 기록됩니다. 성공률은 시드·장비에 따라 달라집니다.
- **영상 증강:** `darker` / `brighter`로 밝기·대비를 크게 바꾸며 상태·액션·시간은 보존합니다.
  프리셋 1개당 선택한 부모 수만큼 생성합니다. 두 프리셋이면 두 배입니다.
- **렌더링 증강:** 조명·재질을 바꾸어 물리 작업부터 다시 실행하고 성공·미디어를 검증합니다.
  새 실행의 상태·액션을 기록합니다. `--max-attempts-per-parent` 기본값은 5입니다.
- `--episodes K`는 증강에 사용할 부모 수입니다. `--mode both`는 영상 프리셋 1개일 때 총 2K개를 만듭니다.
  부모와 파생 데이터는 같은 `split_group`으로 묶어 학습/평가에 나뉘어 들어가지 않게 합니다.

카메라는 Head 1280×720, 좌·우 Wrist 각각 640×480, Third-person 1280×720,
30 fps·8 samples입니다. 데이터는 LeRobot v3 MP4＋Parquet＋메타데이터입니다.
위 `pilot50` 예시의 학습 데이터는 `collection/dataset_50/`,
`augmentation/video/visible_darker_v2/`, `augmentation/render/dataset_25/`에 저장됩니다.

## 동시 수집 (`--parallel`)

PC 한 대에서 Isaac 에피소드를 최대 3개까지 동시에 실행해 원본 수집 시간을 줄입니다.
`--parallel`을 생략하거나 `1`로 지정하면 기존과 같은 직렬 수집입니다(같은 스크립트, 같은 명령).

- 수집 관리자 하나가 전역 잠금을 가진 채 시도 N개를 동시에 실행합니다.
  그래서 다른 수집·증강·replay 도구와는 지금처럼 동시에 실행되지 않습니다.
- 시도를 시작하기 전에 GPU 메모리(`nvidia-smi`), RAM, 디스크 여유, 시작 간격, Isaac 물리 설정을 확인합니다.
  부족하면 실패하지 않고 기다립니다. 막 시작한 Isaac은 메모리를 천천히 늘리므로,
  시작 후 300초 동안은 예상 사용량(VRAM 4.5 GiB, RAM 6.5 GiB)을 미리 빼고 계산합니다.

  | 확인 항목 | 기본 기준 | 옵션 |
  |---|---|---|
  | GPU 여유 메모리 | 4.5 GiB 이상 | `--min-free-vram-gib`, `--kit-vram-gib` |
  | 사용 가능 RAM | 10 GiB 이상 | `--min-available-ram-gib`, `--kit-ram-gib` |
  | 디스크 여유 | 20 GiB + 진행 중인 시도당 10 GiB | `--disk-reserve-gib-per-attempt` |
  | Isaac 시작 간격 | 90초(최소 60초) | `--launch-spacing-seconds` |
  | Isaac 물리 설정(`user.config.json`의 `numThreads` 등) | 이번 실행을 시작할 때와 같고 읽을 수 있음 | 없음(재개하면 그때 값을 새 기준으로 사용) |
  | 이 batch가 실행하지 않은 Isaac | 끝날 때까지 대기 | `--foreign-kit fail`: 새 시도 없이 정지(재개 가능) |

  `nvidia-smi`는 별도 스레드에서 실행합니다. 응답이 늦어도 새 시도만 기다리고, 정지 요청과 진행 중인 시도 감시는 늦어지지 않습니다.

- RTX 5080 16 GB에서 화면 등이 VRAM을 약 5 GB 쓰면 실제로 동시에 도는 시도는 2개입니다. `--parallel 2`를 권장합니다.
  `3`을 지정해도 세 번째 시도는 VRAM이 충분할 때만 시작합니다.
- 시드 배정, 경로별 목표 수량, 최대 시도 수, 성공·실패 판정, 검증, 병합은 직렬과 같습니다.
  진행 중인 시도까지 세어 목표에 도달하면 그 경로의 새 시도를 시작하지 않으므로 목표를 넘는 성공은 생기지 않습니다.
  결과는 끝난 순서와 관계없이 경로별 시도 번호 순서로 기록·병합됩니다.
  시드별 결과가 같다면 직렬 수집과 같은 에피소드가 채택됩니다.

```bash
python3 scripts/collect.py --per-pair 25 --first-seed 1000 --parallel 2 --output ../runs/source50
python3 scripts/pipeline.py --per-pair 25 --first-seed 1000 --parallel 2 --output ../runs/pilot50
python3 scripts/campaign.py --imports /path/to/import_candidates.json --parallel 2 \
  --base-per-pair 100 --augmentation-episodes 100 --first-seed "$FIRST_SEED" --output ../runs/base200_aug200_parallel
# 진행 상황: 슬롯별 진행 중인 시도와 대기 이유(gate) 포함
python3 scripts/collect.py --output ../runs/source50 --status
# 즉시 정지: 진행 중인 시도는 중단되고(부분 결과 보존, 집계 제외) 그 시도 번호는 소모됩니다
touch ../runs/source50/STOP
# 경계 정지: 새 시도는 시작하지 않고, 진행 중인 시도를 끝까지 마치고 검증한 뒤 정지합니다
# (그 시도들로 목표 수량이 채워지면 병합까지 마칩니다)
touch ../runs/source50/DRAIN
# 재개: 같은 명령에 --resume (병렬 수는 batch에 고정되며, 생략하면 저장된 값을 사용)
python3 scripts/collect.py --per-pair 25 --first-seed 1000 --output ../runs/source50 --resume
```

- 직렬로 시작한 batch는 직렬로, 병렬 batch는 같은 N으로만 재개합니다.
  동시 실행 수를 잠시 줄이려면 `--min-free-vram-gib`를 높여 재개합니다(예: 9).
- 계획된 중단에는 `DRAIN`을 쓰세요. `STOP`은 진행 중인 시도를 모두 중단하므로 시도 번호가 최대 N개 소모됩니다.
  캠페인 폴더의 `DRAIN`은 수집 단계에 전달되고, 다른 단계 중이면 다음 단계를 시작하기 전에 멈춥니다.
  마지막 단계인 증강 중에는 다음 단계가 없어 `DRAIN`이 효과가 없습니다(렌더링 증강은 직렬이라 오래 걸립니다).
  그때 멈추려면 `STOP`을 쓰세요. 진행 중인 렌더링 시도 1개가 중단되어 그 부모의 시도 한도(`--max-attempts-per-parent`, 기본 5)에서 1회가 소모됩니다.
  재개가 시작된 뒤에 만든 `STOP`/`DRAIN`은 그 재개에 적용되고, 이전 실행을 멈춘 파일만 이름을 바꿔 보관합니다.
- 터미널이나 SSH 연결이 끊기면(SIGHUP) `DRAIN`처럼 진행 중인 시도를 마치고 멈춥니다. `nohup`으로 시작했다면 무시하고 계속합니다.
  밤샘 수집은 `tmux`나 `nohup … &`로 실행하세요. `campaign.py`와 `pipeline.py`는 SIGHUP을 처리하지 않으므로 반드시 이렇게 실행합니다.
  그렇지 않으면 연결이 끊길 때 이 관리 프로세스가 끝나, 수집 단계는 따로 계속되지만 다음 단계(병합·증강)가 시작되지 않습니다.
- 각 Isaac 시도는 작은 감시 프로세스를 거쳐 실행되고, 끝나면 종료 코드를 `attempt_*/native_exit.json`에 남깁니다.
  수집 관리자가 강제 종료되어도(`kill -9`, 메모리 부족 등) Isaac 시도는 끝까지 실행됩니다. 이 batch가 실행한 프로세스가
  하나라도 남아 있으면 재개를 거부하고, 모두 끝난 뒤 `--resume`하면 이 기록으로 성공·물리 실패를 직렬과 같은 규칙으로 판정해 이어 갑니다.
  정지 신호를 받던 시도는 `interrupted`로 기록합니다.
- PC 재시작처럼 Isaac 시도가 종료 기록 없이 사라졌다면, `--resume`은 아무것도 바꾸지 않고 해당 시도를 알려 주며 거부합니다.
  확인한 뒤 `--resume --interrupt-unfinished`를 붙이면 그 시도를 `interrupted`로 기록하고(시도 번호 소모, 부분 파일 보존) 이어 갑니다.
  캠페인은 `campaign.py … --resume --interrupt-unfinished`로 같은 옵션을 수집 단계에 전달합니다.
- 슬롯마다 별도 임시 폴더(`TMPDIR`)를 쓰고, 같은 슬롯의 다음 시도가 다시 사용합니다. Isaac SDK 폴더의 로그·캐시·설정 파일은
  여러 Isaac이 함께 쓰므로 시작 시각을 최소 60초 띄웁니다.
- 기존 데이터를 가져오는 병렬 캠페인은 새 시드 범위(`--first-seed`부터 경로별 최대 시도 수만큼)가 가져온 에피소드의 같은 경로 시드와
  겹치면 시작하지 않습니다. 같은 경로·같은 시드는 같은 시작 자세를 뽑기 때문입니다. 직렬 캠페인도 같은 규칙을 지켜야 합니다.
  `scripts/import_candidates.py`로 가져올 목록을 만들면 남은 수량, 가능한 증강 수, 안전한 `--first-seed`를 알려 줍니다.
- 조명·재질 렌더링 증강은 이 버전에서도 직렬로 실행됩니다. 전체 옵션은
  `python3 scripts/seeded/collect_parallel.py --help`에서 볼 수 있습니다.

## 여러 PC

각 PC에 배포본과 환경을 설치하고 **PC마다 수집 관리자 1개**를 실행합니다(그 안에서 `--parallel`로 동시 실행 가능).
시작 시드와 출력 디렉터리를 다르게 지정합니다. 아래는 원본 500개＋증강 500개를 새로 6대로 나누는 예입니다.
이미 수집한 PC와 함께 쓸 때는 기존 batch의 시드 범위와 겹치지 않게 지정하세요.

| PC | `--per-pair` | `--first-seed` | 원본 | 영상＋렌더 증강 |
|---|---:|---:|---:|---:|
| 1 | 42 | 1000 | 84 | 42＋42 |
| 2 | 42 | 11000 | 84 | 42＋42 |
| 3 | 42 | 21000 | 84 | 42＋42 |
| 4 | 42 | 31000 | 84 | 42＋42 |
| 5 | 41 | 41000 | 82 | 41＋41 |
| 6 | 41 | 51000 | 82 | 41＋41 |

```bash
# PC 1 예시; 다른 PC는 위 표의 값과 출력 이름을 사용
python3 scripts/pipeline.py --per-pair 42 --first-seed 1000 \
  --augmentation-mode both --augmentation-episodes 42 --output ../runs/pc01
```

작업 중인 batch는 경로와 코드 해시를 고정하므로, 다른 PC로 이동해 `--resume`하는 용도는 아닙니다.
각 PC에서 새 batch를 시작하고 해당 PC에서 증강까지 마치세요.

## Replay

```bash
# ATTEMPT는 collection/left2 또는 right4 아래의 attempt_* 디렉터리
ATTEMPT=../runs/pilot50/collection/right4/attempt_000000_seed_001000
python3 scripts/replay.py --attempt "$ATTEMPT" --validate
python3 scripts/replay.py --attempt "$ATTEMPT" --render --views third_person \
  --output ../runs/replay --start-seconds 0 --duration-seconds 5
# 모든 카메라: --views all / 창 표시: --gui
```

`--start-seconds`와 `--duration-seconds`를 생략하면 보관된 에피소드 전체를 재생합니다.

저장된 골반·관절·물체 상태를 복원하는 **시각 재생**입니다. 액션을 물리에 다시 입력해
동일한 접촉을 재현하는 기능은 아닙니다. 재렌더링 픽셀이 원본 MP4와 완전히 같다는 뜻도 아닙니다.
원본 영상을 그대로 보려면 데이터셋의 MP4를 사용하세요.

Replay를 위해 해당 `attempt_*` 폴더의 `profile/`, `collection/`과 압축 `trace.jsonl.gz`를 보관하세요.
학습용 `dataset/`의 Parquet·MP4만으로는 전체 장면의 상태를 복원할 수 없습니다.
렌더링 증강은 `--attempt`에 `render/children/episode_*/trial_*`을 지정합니다.
이때 `profile/config.json`, `visual.json`, `scene_visual_overlay.usda`도 함께 보관해야
조명·재질 설정을 복원할 수 있습니다. 영상 후처리로 적용한 밝기·대비는 증강 MP4로 확인하세요.

## 구성 및 GitHub

```text
assets/          씬·로봇·재질·HugWBC 정책
configs/         두 경로의 동작·카메라 설정
runtime/         Isaac 실행·물리 보정 코드
ik/pink/         Pink IK 작업자(별도 Python 3.11 환경)
scripts/         설치·수집·증강·replay·무결성 검증
requirements/    고정 버전 의존성
tests/           수집·증강·replay 검증
```

기존 데이터셋·영상·로그·개발 기록·가상환경은 제외했습니다. `.gitattributes`에 에셋용 Git LFS 규칙이 있습니다.
새 GitHub 저장소를 만든 뒤 이 디렉터리에서 실행하세요.

```bash
git init -b main
git lfs install
git add .
git lfs ls-files
git commit -m "Add portable NEPES collection, augmentation and replay"
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

배포 파일을 수정했다면 `python3 scripts/build_release.py`로 manifest를 갱신하고 새 batch를 시작합니다.
CPU 회귀 검증은 `"$NEPES_DATASET_PYTHON" -B -m unittest discover -s tests`로 실행합니다.
SDK·에셋 라이선스 정보는 `THIRD_PARTY_NOTICES.txt`를 참고하세요.
