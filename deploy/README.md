# 로컬 서버 공유 (호스트용)

Mac 한 대에서 MySQL 과 API 서버를 돌리고, Cloudflare Tunnel 로 팀에 연다.
이 문서는 **서버를 여는 사람** 것이다. 접속하는 사람은 [TEAM.md](TEAM.md) 를 본다.

```
팀원 브라우저·앱  ─── https://api.<도메인>  ─┐
                                            ├─ Cloudflare ─ 터널 ─ cloudflared ─┬─ 127.0.0.1:8000  uvicorn
팀원 DB 클라이언트 ─ cloudflared access tcp ─┘        (Access 정책)              └─ 127.0.0.1:3306  MySQL
```

포트를 공유기에 여는 것이 아니다. cloudflared 가 Cloudflare 쪽으로 **나가는** 연결을
만들고 그 위로 트래픽이 되돌아온다. 공유기 설정·고정 IP·방화벽 규칙이 필요 없다.

| 여는 것 | 주소 | 보호 |
| --- | --- | --- |
| API | `https://api.<도메인>` | 없음 (공개) |
| MySQL | `db.<도메인>` (TCP) | Cloudflare Access + MySQL 계정 |

API 를 공개로 두는 이유는 아래 [Access 정책](#access-정책) 에 적었다.

---

## 0. 준비물

| 항목 | 확인 |
| --- | --- |
| Cloudflare 계정에 등록된 도메인 | 네임서버가 Cloudflare 를 향하고 상태가 Active |
| Homebrew | `brew -v` |
| Python 3.10 | `python3.10 -V` — 없으면 `brew install python@3.10` |

도메인이 Cloudflare 에 등록되어 있지 않으면 이 문서의 절반(고정 주소·Access)이
성립하지 않는다. Cloudflare 대시보드 → Websites 에서 상태부터 확인한다.

---

## 1. MySQL

```bash
brew install mysql@8.0
brew services start mysql@8.0
```

`mysql@8.0` 은 keg-only 라 PATH 에 자동 등록되지 않는다.

```bash
echo 'export PATH="/opt/homebrew/opt/mysql@8.0/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
mysql -V        # 8.0.x
```

Intel Mac 이면 `/usr/local/opt/...` 다. `brew --prefix mysql@8.0` 으로 확인한다.

### 설정

[`mysql/my.cnf.snippet`](mysql/my.cnf.snippet) 의 내용을 `/opt/homebrew/etc/my.cnf` 에 붙인다.
타임존·문자셋·바인딩 세 가지가 들어 있고, **셋 다 나중에 조용히 문제를 만드는 항목**이다.

```bash
brew services restart mysql@8.0
```

### 계정과 스키마

```bash
mysql_secure_installation      # root 비밀번호 설정. 처음 한 번만
```

[`mysql/init_user.sql`](mysql/init_user.sql) 에서 `CHANGE_ME_APP` 과 `CHANGE_ME_TEAM` 을
바꾼 뒤 실행한다.

```bash
mysql -u root -p < deploy/mysql/init_user.sql
```

끝에 나오는 확인 쿼리에서 이 값들을 본다.

| 항목 | 기대값 |
| --- | --- |
| `global_tz` | `+09:00` |
| `charset` / `collation` | `utf8mb4` / `utf8mb4_unicode_ci` |
| `bind_address` | `127.0.0.1` |

---

## 2. 백엔드

venv 와 conda 중 편한 쪽을 쓴다. **파이썬이 3.10 인 것만 지키면 된다.**

```bash
cd PetFit_Backend

# venv 를 쓰는 경우
python3.10 -m venv .venv
source .venv/bin/activate

# conda 를 쓰는 경우
conda create -n petfit python=3.10
conda activate petfit
```

conda 를 쓴다면 **`base` 에 설치하지 않는다.** 다른 작업과 의존성이 뒤섞이고
파이썬 버전을 프로젝트에 맞춰 고정할 수 없다. `conda config --set auto_activate_base false`
로 자동 활성화를 꺼두면 실수가 줄어든다.

```bash
python -V                    # Python 3.10.x
pip install -r requirements.txt

cp .env.example .env
```

### Apple Silicon 이면 greenlet 을 따로 깔아야 한다

`requirements.txt` 가 `sqlalchemy>=2.0` 만 지정하면 **Mac Studio 에서는 greenlet 이
설치되지 않는다.** SQLAlchemy 가 greenlet 을 자동으로 끌어오는 플랫폼 목록에
`aarch64` 는 있지만 macOS 가 보고하는 `arm64` 는 빠져 있기 때문이다.

인텔 맥과 윈도우 팀원은 아무 문제가 없고, Apple Silicon 에서만 `init_db` 실행 시
`the greenlet library is required` 로 멈춘다.

```bash
pip install greenlet
```

`requirements.txt` 의 `sqlalchemy>=2.0` 을 `sqlalchemy[asyncio]>=2.0` 으로 바꾸면
플랫폼과 무관하게 함께 설치되므로 이 단계가 없어진다.

`.env` 의 `DB_PASSWORD` 에 `CHANGE_ME_APP` 대신 넣은 값을 적는다.
`DB_HOST` 는 `localhost` 가 아니라 `127.0.0.1` 이다. 이유는 `.env.example` 주석에 있다.

```bash
python -m scripts.init_db
```

`검증 통과` 가 나와야 한다. 타임존 줄이 `+09:00` 이 아니면 1단계의 my.cnf 가
반영되지 않은 것이다.

```bash
chmod +x deploy/serve.sh
./deploy/serve.sh
```

http://127.0.0.1:8000/docs 가 열리면 여기까지 정상이다.

`serve.sh` 는 가상환경·파이썬 버전·`.env`·DB·포트를 먼저 확인하고 하나라도 어긋나면
서버를 띄우지 않는다. **`app/main.py` 의 lifespan 은 DB 연결 실패를 삼키고 앱을 띄운다.**
`/animals` `/spaces` 는 DB 없이 응답하므로 겉보기에 멀쩡하고, 분석 요청만 전부 실패한다.
팀원이 그 상태를 디버깅하게 두지 않으려고 앞단에서 막는다.

---

## 3. Cloudflare Tunnel

```bash
brew install cloudflared
cloudflared tunnel login          # 브라우저에서 도메인 선택
cloudflared tunnel create petfit
```

`create` 가 UUID 와 `~/.cloudflared/<UUID>.json` 경로를 출력한다. **이 json 이 터널의
비밀키다.** 저장소에 올리지 않고 메신저로 보내지 않는다.

```bash
cp deploy/cloudflared/config.yml.example ~/.cloudflared/config.yml
```

`<TUNNEL_UUID>` 와 `<도메인>` 을 채운 뒤 검증한다.

```bash
cloudflared tunnel ingress validate
```

DNS 레코드를 만든다. 이 명령이 Cloudflare 에 CNAME 을 자동으로 넣는다.

```bash
cloudflared tunnel route dns petfit api.<도메인>
cloudflared tunnel route dns petfit db.<도메인>
```

실행한다.

```bash
cloudflared tunnel run petfit
```

다른 창에서 `./deploy/serve.sh` 가 돌고 있어야 한다. 순서는 상관없지만 **둘 다**
떠 있어야 한다.

```bash
curl https://api.<도메인>/animals
```

응답이 오면 터널이 연결된 것이다.

---

## Access 정책

**DB 호스트네임에는 반드시 건다. API 에는 걸지 않는다.**

Access 는 브라우저 로그인 화면으로 사람을 걸러낸다. API 에 걸면 프론트의 `fetch` 와
Capacitor 앱 요청이 로그인 페이지 HTML 을 받는다. `X-Device-Id` 만으로 동작하도록
설계된 API 라 인증 흐름을 태울 곳도 없다. 그래서 API 는 공개로 둔다.

### DB 잠그기

Cloudflare Zero Trust 대시보드 → Access → Applications → Add an application → **Self-hosted**

| 항목 | 값 |
| --- | --- |
| Application domain | `db.<도메인>` |
| Policy action | Allow |
| Include | Emails → 팀원 이메일을 하나씩 |

정책을 만들지 않으면 주소를 아는 누구나 3306 에 붙는다. 그 뒤에 남는 것은 MySQL
비밀번호 하나뿐이고, 그건 인터넷에 노출된 DB 를 지키는 수단이 못 된다.

### 공개 API 에서 감수하는 것

`api.<도메인>` 은 주소를 아는 누구나 분석을 요청할 수 있다. 실질적 위험은 침입보다
**디스크와 순번**이다. 영상 한 건이 최대 100MB 고, 큐는 10건까지 받는다.

완화책은 세 가지다.

- 서브도메인을 `api` 대신 추측하기 어려운 이름으로 둔다. 팀 개발용이라 예쁠 필요가 없다.
- Cloudflare 대시보드 → Security → WAF → Rate limiting rules 에 규칙을 하나 만든다.
  `POST /analysis` 를 IP 당 1분에 5회로 제한하면 충분하다. 무료 플랜도 규칙 1개를 쓴다.
- **작업이 끝나면 터널을 내린다.** 상시 구동을 걸어두고 잊는 것이 가장 흔한 사고다.

---

## 상시 구동

기본은 터미널 창 두 개다. Mac 을 켜 둔 동안만 열린다. 그 편이 통제하기 쉽고, 개발
기간 동안은 대개 이걸로 충분하다.

계속 띄워두려면 cloudflared 만 서비스로 등록한다.

```bash
sudo cloudflared service install
```

macOS 는 `~/.cloudflared/config.yml` 을 읽는 LaunchDaemon 을 만든다. 내릴 때는

```bash
sudo cloudflared service uninstall
```

API 서버까지 서비스로 만들지는 않는다. `--reload` 여부, 로그, 재시작 시점을 손에
쥐고 있는 편이 낫다.

### 잠자기

**Mac 이 잠들면 터널도 같이 끊긴다.** 팀원 입장에서는 서버가 죽은 것과 구분되지 않는다.

시스템 설정 → 디스플레이 → 고급 → "디스플레이가 꺼져 있을 때 자동으로 잠자지 않음" 을 켜거나,
서버를 띄운 창에서

```bash
caffeinate -i ./deploy/serve.sh
```

로 실행한다. `-i` 는 유휴 잠자기만 막는다.

---

## Workbench 로 DB 관리

```bash
brew install --cask mysqlworkbench
```

`brew install mysql@8.0` 에는 포함되지 않는다. GUI 는 별도 cask 다.

### 연결 만들기

호스트 본인은 터널을 거치지 않는다. 같은 장비의 MySQL 에 바로 붙는다.

| 항목 | 값 |
| --- | --- |
| Connection Method | Standard (TCP/IP) |
| Hostname | `127.0.0.1` |
| Port | `3306` |
| Username | `petfit` |
| Default Schema | `petfit` |

계정 관리·권한 변경처럼 스키마 밖의 일을 할 때만 `root` 로 연결을 하나 더 만든다.
평소 데이터를 볼 때는 `petfit` 을 쓴다. 앱과 같은 권한으로 보는 편이,
"Workbench 에서는 되는데 앱에서는 안 되는" 상황을 미리 잡아준다.

### 스키마를 Workbench 로 바꾸지 않는다

**이게 이 문서에서 제일 중요한 줄이다.**

이 저장소의 스키마 정본은 `app/models/` 다. `migrations/001_initial.sql` 조차
`scripts/gen_ddl.py` 가 뽑아내는 파생물이다.

Workbench 의 Alter Table · Forward Engineer · Synchronize Model 로 컬럼을 더하면
그 변경은 **DB 에만 남고 모델에는 없다.** 그 상태로 커밋하면

- 팀원이 `python -m scripts.init_db` 를 돌려도 그 컬럼이 생기지 않는다
- 내 장비에서만 코드가 동작하고 원인은 코드 어디에도 없다
- `gen_ddl` 로 뽑은 SQL 이 실제 DB 와 어긋난다

컬럼이나 제약을 바꿔야 하면 `app/models/` 를 고치고 `init_db` 를 다시 돌린다.
Workbench 는 **데이터를 보고 고치는 용도**로 쓴다. 스키마는 건드리지 않는다.

개발 중이라 데이터를 버려도 되면 이게 가장 빠르다.

```bash
python -m scripts.init_db --drop      # 전체 삭제 후 모델대로 재생성
```

### 쓸 만한 것

역공학으로 ER 다이어그램을 뽑을 수 있다. Database → Reverse Engineer → 스키마 선택.
DB 설계서에 넣을 그림이 필요할 때 유용하다. **읽기만 하는 동작이라 안전하다.**
단, 이렇게 만든 모델에서 Forward Engineer 로 되돌리지 않는다. 위와 같은 문제가 생긴다.

### Server Status 는 안 될 것이다

Workbench 의 시작·정지·상태 확인은 Oracle 공식 설치본의 launchd 항목을 전제한다.
Homebrew 는 `homebrew.mxcl.mysql@8.0` 이라는 다른 이름을 쓰므로 Workbench 가 찾지 못한다.

맞추려고 System Profile 을 손대지 말고, 서버 제어는 그냥 터미널에서 한다.

```bash
brew services list
brew services restart mysql@8.0
```

Options File 편집기만 쓰고 싶으면 연결 설정의 System Profile 탭에서
Configuration File 을 `/opt/homebrew/etc/my.cnf` 로 지정한다.
다만 **1단계에서 넣은 설정을 Workbench 가 다시 쓰면서 주석이 날아갈 수 있다.**
`my.cnf` 는 에디터로 직접 여는 편이 안전하다.

### 팀원도 Workbench 를 쓴다면

접속값은 [TEAM.md](TEAM.md) 와 같고 포트만 `3307` 이다. 터널이 끊기면 Workbench 는
"연결이 없다" 대신 오래 멈춰 있다가 죽는다. `cloudflared access tcp` 창이 살아 있는지
먼저 보라고 알려준다.

---

## 알아둘 것

### 업로드 100MB 가 경계에 걸려 있다

Cloudflare 무료·Pro 플랜의 요청 본문 상한이 **100MB** 다.
`app/core/constants.py` 의 `VIDEO_MAX_BYTES` 도 정확히 100MB 다.

두 값이 같으면 통과할 것 같지만 그렇지 않다. multipart 인코딩이 바운더리와 헤더를
더하므로 **100MB 영상의 실제 본문은 100MB 를 넘는다.** 그러면 요청은 백엔드에 닿지도
못하고 Cloudflare 가 413 을 낸다.

문제는 그 응답이 Cloudflare 의 HTML 오류 페이지라는 것이다. `{code, message, field, status}`
형식이 아니므로 프론트의 `ApiError` 파싱이 깨지고, 사용자는 "용량 초과" 대신 정체불명의
오류를 본다.

대응은 프론트 쪽이다. `PetFit_frontend/README.md` 의 남은 작업에 있는 업로드 전
클라이언트 검증을 넣되, **임계값을 100MB 가 아니라 95MB 로 잡는다.** 백엔드 상수는
건드리지 않는다. 그건 서비스 사양이고 터널은 지금 쓰는 임시 경로다.

### 서버를 재시작하면 진행 중이던 분석이 사라진다

큐가 인프로세스라 프로세스와 함께 없어진다. 시작할 때 lifespan 이 DB 에 남은
`PENDING` · `PROCESSING` 행을 정리하므로 기기가 잠기지는 않지만, 그때 돌던 분석은
결과 없이 끝난다. 팀원이 테스트 중일 때 재시작하지 않는다.

### 지금 붙어 있는 파이프라인은 StubPipeline 이다

`app/main.py` 의 lifespan 이 `StubPipeline` 을 물린다. 실제 점수가 아니라 형식이 맞는
가짜 결과가 나온다. 프론트 연동 확인에는 충분하고, 팀원이 결과값을 진짜로 오해하지
않도록 공유할 때 한 줄 알려준다.

### 프론트 설정

팀원은 `PetFit_frontend/.env` 에 이렇게 둔다.

```
VITE_API_BASE_URL=https://api.<도메인>
```

마킹 이미지는 백엔드가 `/images/<uuid>.jpg` 상대 경로로 내려주므로 이 값만 바꾸면
이미지도 같이 따라온다. CORS 는 `app/main.py` 에서 이미 `*` 로 열려 있다.

---

## 문제 해결

| 증상 | 확인 |
| --- | --- |
| `curl` 이 502 · 1033 | 8000 포트에 서버가 안 떠 있다. `./deploy/serve.sh` 상태 확인 |
| 터널이 `Inactive` | `cloudflared tunnel info petfit`. 나가는 7844 포트가 막혔는지 |
| 413 · 오류 형식이 이상함 | 위 [업로드 100MB](#업로드-100mb-가-경계에-걸려-있다) |
| 524 | 응답이 100초를 넘었다. 엣지 제한이라 터널 설정으로 못 늘린다 |
| `init_db` 타임존 경고 | my.cnf 반영 안 됨. `brew services restart mysql@8.0` |
| 팀원 DB 접속이 `Access denied` | 계정 host 가 `127.0.0.1` 인지. 터널 경유는 전부 루프백으로 보인다 |
| 팀원 DB 접속이 멈춤 | Access 로그인 세션 만료. `cloudflared access login db.<도메인>` |
| 분석이 전부 실패 | `python -m scripts.init_db --check` 로 DB부터 |

---

## 매일 시작할 때

```bash
brew services list                      # mysql@8.0 started 확인

cd PetFit_Backend
source .venv/bin/activate
caffeinate -i ./deploy/serve.sh         # 창 1

cloudflared tunnel run petfit           # 창 2
```
