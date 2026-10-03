# Claude 를 병렬로 돌리기

## 먼저 알아둘 것

- 세션 하나 = 대화 하나. 터미널을 여러 개 열어 `claude` 를 각각 띄우면 그게 병렬이다.
- 같은 폴더의 같은 파일을 두 세션이 동시에 고치면 서로 덮어쓴다. 그래서 **담당 파일을 나누거나**(CLAUDE.md 의 표),
  **워크트리**(`claude -w 이름`, 세션마다 폴더 복사본)를 쓴다.
- 새 세션은 이 프로젝트의 `CLAUDE.md` 를 자동으로 읽는다. 지금까지의 결정·환경·규칙이 거기 있으니,
  대화 전체를 넘기지 않아도 대부분 이어진다.

## 방법 1 — 새 세션 여러 개 (추천, 가볍다)

터미널마다:

```bash
cd ~/pinky_traffic_ws
claude -n 인식        # -n 은 세션 이름. /resume 목록과 터미널 제목에 보인다
```

그리고 아래 프롬프트 중 하나를 붙여 넣는다. 오늘(1대)은 A 학습 · B 튜닝 · C 모니터링 세 개. D 는 2대로 넘어갈 때.

### A. 학습 세션

```
너는 pinky_traffic 의 인식·학습 담당이야. CLAUDE.md 와 docs/TRAINING.md 를 먼저 읽어.
담당 파일: core/perception.py, core/detectors.py, tools/, data/, models/, docs/TRAINING.md,
그리고 config/field.yaml 의 lane_hsv_lo / lane_hsv_hi 두 줄만. 다른 파일은 고치지 마.

현장: 로봇 1대(pinky1, 192.168.0.1). 차선도 횡단보도도 흰 테이프(같은 색 모드, 줄무늬로 횡단보도 구분).

지금 할 일:
1) 사진 모으는 명령(capture)을 알려 줘. 모이면 data/raw 에 있어.
2) hsv_tuner 로 지금 조명에서 흰 테이프 값을 찾는 방법을 알려 주고, 내가 찾은 값을 주면 field.yaml 에 넣어.
3) eval_detector 로 HSV 인식률(lane_ok, 횡단보도 인식)을 재고, 잘못 잡은 유형을 정리해 줘
   (흰 바닥 반사·조명을 테이프로 잡는지, 줄무늬를 차선으로 잡는지).
4) autolabel 명령을 줘(--review 는 내가 직접 해). data/lane_ds 가 생기면 ~/venv/yolo 로
   models/synth_best.pt 에서 이어서 60 에폭 학습을 백그라운드로 돌려 models/best.pt 를 만들어.
5) eval_detector --compare 로 hsv 와 비교해 docs/TRAINING.md 4번 표 기준 통과/미달을 표로 알려 줘.
로봇을 움직이는 명령은 실행하지 마. 단계가 끝날 때마다 docs/PROGRESS.md 맨 위에 한 줄 추가하고 담당 파일만 커밋해.
```

### B. 튜닝 세션

```
너는 pinky_traffic 의 주행·튜닝 담당이야. CLAUDE.md 와 docs/RUNBOOK.md 를 먼저 읽어.
담당 파일: core/controller.py, core/driver.py, nodes/, config/ (lane_hsv 두 줄은 학습 세션 담당), docs/RUNBOOK.md.
다른 파일은 고치지 마.

현장: 로봇 1대(pinky1, 192.168.0.1, 도메인 23). 차선도 횡단보도도 흰 테이프.
대시보드(localhost:8088)와 lane_driver 는 내가 띄워. 로봇을 움직이는 건 내가 대시보드로 해. 너는 직접 START 하지 마.

지금 할 일:
1) 처음 띄울 때 막히면(카메라 영상 안 뜸, 토픽 안 보임, 라이다 방향) RUNBOOK 순서대로 원인을 같이 찾아 줘.
2) 내가 증상을 말하면(예: "곡선에서 바깥 선을 밟아", "횡단보도에서 안 서") runs/ 최신 폴더의 pinky1.csv 에서
   그 구간의 offset·w·state 를 보고 원인을 설명하고, 바꿀 슬라이더 값을 하나씩 제안해.
3) 값이 맞으면 config/field.yaml 에 반영해.
4) 코드를 고쳐야 하면 먼저 시뮬레이터(run_sim --headless)로 재현하고, 고친 뒤 pytest 통과를 확인해.
값이나 코드를 바꿀 때마다 담당 파일만 커밋해.
```

### C. 모니터링·기록 세션

```
너는 pinky_traffic 의 모니터링·기록 담당이야. CLAUDE.md 와 docs/TESTCASES.md 를 먼저 읽어.
담당 파일: dashboard/, docs/TESTCASES.md, docs/PROGRESS.md. 다른 파일은 고치지 마. 로봇을 움직이는 명령은 실행하지 마.

현장: 로봇 1대(pinky1, 192.168.0.1). 차선도 횡단보도도 흰 테이프. 오늘은 1대 테스트만 한다.

지금 할 일:
1) 내가 "봐 줘" 하면 runs/ 최신 폴더의 pinky1.csv 를 읽어 한 바퀴 요약을 표로 줘:
   주행 시간, 차선 중심 최대·평균 이탈, lost 횟수와 위치(시각), 횡단보도 정지 횟수·정지 위치, 장애물 정지.
   이전 주행과 비교해서 좋아졌는지/나빠졌는지도.
2) testcase_results.json 을 읽어 통과/실패/남은 항목(1대 항목만)을 표로 정리하고,
   실패 항목마다 메모와 CSV 를 근거로 원인 후보를 적어.
3) 그 내용을 docs/PROGRESS.md 맨 위에 오늘 날짜·시각으로 추가하고, 노션 진행 기록 페이지에도 올려
   (접근이 안 되면 안 된다고 말하고 PROGRESS.md 만 써). 쓸 때마다 담당 파일만 커밋해.
4) 대시보드가 이상하면(영상 안 뜸, 값 안 바뀜) 원인을 찾아 고쳐.
```

### D. 2대 운용 세션 (선택)

```
너는 pinky_traffic 의 2대 운용 담당이야. CLAUDE.md, docs/RUNBOOK.md 6번을 먼저 읽어.
담당 파일: core/coordinator.py, sim/, test/test_sim.py. 다른 파일은 고치지 마.

지금 할 일: run_sim --robots 2 --coordinator --headless 로 현장 값(config/field.yaml)을 넣어 200초 돌리고,
추돌·최소 간격·횡단보도 동시 진입을 보고해. 문제가 있으면 원인을 시뮬레이션 로그로 보여 주고 수정안을 제안해
(적용은 내 확인 뒤에).
```

## 방법 2 — 이 세션(지금까지의 대화)을 그대로 이어받은 세션 만들기

이 대화는 `/home/jeongmin` 에서 시작했고 세션 ID 는 `c5fba352-4a57-4648-8fd6-1023d0e7cf3b` 다.

```bash
cd ~                       # 세션은 시작한 폴더 기준으로 찾는다
claude --resume c5fba352-4a57-4648-8fd6-1023d0e7cf3b --fork-session -n 인식
claude --resume c5fba352-4a57-4648-8fd6-1023d0e7cf3b --fork-session -n 주행     # 다른 터미널에서
```

- `--fork-session` 이 핵심이다. 이 대화 내용을 전부 가진 **새 세션 ID** 로 갈라진다. 원래 세션은 그대로 남는다.
- `--fork-session` 없이 같은 세션을 두 터미널에서 동시에 열면 한 대화에 둘이 번갈아 쓰게 된다. 하지 말 것.
- ID 를 모르면 `claude --resume` 만 치면 목록에서 고를 수 있다. 가장 최근 대화는 `claude -c --fork-session`.
- 주의: 이 대화는 PDF 60개와 위키를 읽어서 아주 길다. 갈라진 세션마다 그 길이를 다시 들고 가므로 사용량이 크다.
  수업 자료 내용까지 기억한 채로 일을 시켜야 할 때만 쓰고, 보통은 방법 1 로 충분하다.

## 방법 3 — 파일이 겹칠 것 같으면 워크트리

```bash
cd ~/pinky_traffic_ws
claude -w perception -n 인식     # 복사본 폴더(git worktree) + 새 브랜치를 만들고 거기서 작업
claude -w control -n 주행
```

각 세션이 자기 브랜치에서 작업하므로 서로 덮어쓰지 않는다. 끝나면 사용자가 merge 한다 (수업 52번 자료).
워크트리는 커밋된 내용에서 갈라지므로, 시작 전에 현재 작업을 커밋해 둔다.
`data/`, `models/`, `runs/` 는 git 에 안 올라가 있어 복사본에 없다 — 인식·학습 세션은 워크트리 대신 원래 폴더에서 돌리는 편이 낫다.

## 방법 4 — 창을 안 띄우고 백그라운드로

```bash
cd ~/pinky_traffic_ws
claude --bg -n 학습 "CLAUDE.md 와 docs/TRAINING.md 를 읽고, data/lane_ds 로 60 에폭 학습해서 models/best.pt 를 만들고 mAP 를 보고해"
claude agents          # 돌고 있는 것 목록
claude logs <id>       # 최근 출력
claude attach <id>     # 그 세션으로 들어가기
claude stop <id>
```

학습처럼 오래 걸리고 중간에 물어볼 게 없는 일에 맞다.

## 방법 5 — 한 세션 안에서 병렬 (서브에이전트)

세션에 이렇게 말하면 된다:

```
서브에이전트 두 개로 병렬로 해 줘.
1) eval_detector 로 data/raw 의 HSV 인식률 측정
2) run_sim 으로 v_max 0.08 / 0.12 / 0.16 각각 150초 돌려 최대 이탈 비교
결과만 표로 합쳐서 알려 줘.
```

조사·측정처럼 서로 독립이고 결과만 필요할 때 좋다. 같은 파일을 고치는 일에는 쓰지 않는다.

## 세션끼리 내용 맞추기

- 결정·환경·규칙 → `CLAUDE.md` (모든 세션이 시작할 때 읽는다)
- 진행 상황 → `docs/PROGRESS.md` 와 노션
- 코드 → git 커밋 (각 세션이 자기 담당 파일만 `git add <파일>` 로 커밋, 커밋 전용 세션은 두지 않는다). 다른 세션에게는 "git log 최근 5개와 docs/PROGRESS.md 를 읽고 이어서 해" 라고 하면 된다.
