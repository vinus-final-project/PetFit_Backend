# 공유 서버 접속 (팀원용)

MacStudio에서 도는 API 서버와 MySQL 에 붙는 방법.

| 대상 | 주소                          | 준비물 |
| --- |-------------------------------| --- |
| API | `https://api.voice-in-us.com` | 없음 |
| MySQL | `db.voice-in-us.com`          | `cloudflared` 설치 + Access 승인 |

서버는 MacStudio가 켜져 있을 때만 열린다. 안 되면 먼저 물어본다.

---

## API

프론트에서는 `.env` 한 줄이면 된다.

```
VITE_API_BASE_URL=https://api.voice-in-us.com
```

확인.

```bash
curl https://api.voice-in-us.com/animals
```

브라우저로 `https://api.voice-in-us.com/docs` 를 열면 Swagger 에서 직접 호출해볼 수 있다.

### 알아둘 것

- **지금 결과는 가짜다.** 실제 AI 파이프라인 대신 `StubPipeline` 이 물려 있다.
  형식은 명세와 같고 값은 의미 없다. 화면 연동 확인용이다.
- **영상은 95MB 아래로 올린다.** 사양은 100MB 지만 중간 경로의 상한도 100MB 라
  경계에서 걸린다. 이때 오류 응답이 평소 형식(`{code, message, ...}`)이 아니라
  HTML 로 오므로 프론트에서 이상하게 보인다.
- **서버가 재시작되면 그때 돌던 분석은 사라진다.** 결과 없이 끝나므로 다시 요청한다.
- `X-Device-Id` 는 기기마다 다르다. 각자 이력만 보인다. 남의 분석은 안 보이는 게 정상이다.

---

## MySQL

터널 위로 붙는다. 주소를 DB 클라이언트에 그대로 넣는 방식이 아니다.
**내 PC 의 로컬 포트를 서버 3306 에 연결해 두고, 클라이언트는 그 로컬 포트에 붙는다.**

### 1. cloudflared 설치

```bash
brew install cloudflared              # macOS
winget install --id Cloudflare.cloudflared    # Windows
```

### 2. 터널 열기

```bash
cloudflared access tcp --hostname db.voice-in-us.com --url 127.0.0.1:3307
```

처음 실행하면 브라우저가 열리고 Cloudflare Access 로그인을 요구한다.
승인된 이메일로 로그인한다. 목록에 없으면 서버장에게 추가를 요청한다.

**이 창은 켜 둔 채로 둔다.** 닫으면 연결이 끊긴다.

로컬 포트를 3306 이 아니라 **3307** 로 쓰는 이유는, 내 PC 에 MySQL 이 깔려 있으면
3306 이 이미 점유되어 있기 때문이다. 비어 있으면 3306 을 써도 된다.

### 3. 클라이언트 접속

이제 `127.0.0.1:3307` 이 서버의 MySQL 이다.

| 항목 | 값                |
| --- |-------------------|
| Host | `127.0.0.1`       |
| Port | `3307`            |
| Database | `petfit`          |
| User | `petfit_team`     |
| Password | 서버장에게 받는다 |

```bash
mysql -h 127.0.0.1 -P 3307 -u petfit_team -p petfit
```

Workbench · DBeaver · TablePlus · IntelliJ 도 같은 값으로 넣는다.
**SSH 터널 옵션은 쓰지 않는다.** 이미 터널을 지나온 뒤라 그냥 로컬 접속이다.

MySQL Workbench 를 쓴다면 Connection Method 는 `Standard (TCP/IP)` 다.
설치는 `brew install --cask mysqlworkbench` (서버와 별개 패키지다).

터널이 끊기면 Workbench 는 "연결 없음" 대신 한참 멈춰 있다가 죽는다.
그럴 때는 DB 를 의심하기 전에 `access tcp` 창이 살아 있는지 먼저 본다.

### 스키마는 바꾸지 않는다

`petfit_team` 은 조회 전용이라 애초에 막혀 있지만, 개인 계정을 받았더라도
Workbench 의 Alter Table · Forward Engineer 로 테이블 구조를 바꾸지 않는다.
스키마 정본은 `app/models/` 이고 DB 는 거기서 생성된 결과다. DB 만 고치면
그 변경은 아무 코드에도 남지 않는다.

### 권한

`petfit_team` 은 **조회 전용**이다. `INSERT` · `UPDATE` · `DELETE` 는 막혀 있다.
계정을 여럿이 나눠 쓰므로 실수 한 번의 범위를 좁혀 둔 것이고, 쓰기가 필요하면
개인 계정을 따로 받는다.

### 시각

모든 시각은 **KST(UTC+9)** 로 저장되어 있다. `created_at` 을 UTC 로 해석하지 않는다.
직접 `INSERT` 하게 되면 세션 타임존을 확인한다.

```sql
SELECT @@session.time_zone;    -- +09:00 이어야 한다
```

---

## 안 될 때

| 증상 | 원인                                                       |
| --- |------------------------------------------------------------|
| API 가 502 · 1033 | 서버가 안 떠 있다. MacStudio 확인                          |
| API 응답이 아예 없음 | MacStudio가 잠들었을 수 있다                               |
| `cloudflared` 에서 브라우저가 안 열림 | `cloudflared access login db.voice-in-us.com` 을 먼저 실행 |
| DB 접속 `Access denied` | 계정·비밀번호 확인. Host 는 반드시 `127.0.0.1`             |
| DB 연결이 갑자기 끊김 | `access tcp` 창이 닫혔거나 세션 만료. 다시 실행            |
| 413 또는 이상한 HTML 오류 | 영상이 너무 크다. 95MB 아래로                              |
