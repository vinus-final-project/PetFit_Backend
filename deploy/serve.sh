#!/usr/bin/env bash
#
# 팀 공유용 API 서버 기동.
#
#   ./deploy/serve.sh              # 공유 모드 (reload 없음)
#   ./deploy/serve.sh --reload     # 내 개발용
#
# 저장소 루트에서 실행한다. 가상환경·파이썬 버전·DB 접속을 먼저 확인하고,
# 하나라도 어긋나면 uvicorn 을 띄우지 않는다. 절반만 뜬 서버를 팀원이
# 디버깅하게 두지 않는다.

set -euo pipefail

HOST=127.0.0.1
PORT=8000
RELOAD=""

for arg in "$@"; do
  case "$arg" in
    --reload) RELOAD="--reload" ;;
    --port=*) PORT="${arg#*=}" ;;
    *) echo "알 수 없는 인자: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

fail() { echo "  ! $1" >&2; exit 1; }
ok()   { echo "  + $1"; }

echo "점검"

# --- 가상환경 -----------------------------------------------------------
# venv 와 conda 를 모두 받는다. conda 는 VIRTUAL_ENV 가 아니라 CONDA_PREFIX 를 쓴다.
# 이미 활성화된 환경이 있으면 그것을 존중하고, 없을 때만 저장소의 .venv 를 찾는다.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ok "venv ${VIRTUAL_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  # base 는 거른다. 여기에 프로젝트 의존성을 깔면 다른 작업과 뒤섞이고,
  # 파이썬 버전을 프로젝트에 맞춰 고정할 수도 없다.
  [[ "${CONDA_DEFAULT_ENV:-}" != "base" ]] \
    || fail "conda base 다. 전용 환경을 만들어 활성화한다: conda create -n petfit python=3.10"
  ok "conda ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ok "venv ${VIRTUAL_ENV}"
else
  fail "활성화된 가상환경이 없다. conda activate petfit 또는 python3.10 -m venv .venv"
fi

# --- 파이썬 버전 --------------------------------------------------------
PYV="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$PYV" == "3.10" ]] || fail "Python ${PYV} 다. 3.10 이어야 한다"
ok "Python ${PYV}"

# --- 설정 ---------------------------------------------------------------
[[ -f .env ]] || fail ".env 가 없다. cp .env.example .env 후 채운다"
ok ".env"

# --- DB -----------------------------------------------------------------
# 여기서 막히면 uvicorn 은 뜨지만 분석 요청이 전부 실패한다.
# lifespan 이 DB 오류를 삼키고 앱을 띄우도록 되어 있어(app/main.py) 겉보기엔 정상이다.
if ! python -m scripts.init_db --check > /tmp/petfit_dbcheck.log 2>&1; then
  cat /tmp/petfit_dbcheck.log >&2
  fail "DB 점검 실패. 위 출력 확인. brew services list 로 mysql@8.0 상태부터 본다"
fi
ok "DB 연결·스키마"

# --- 포트 ---------------------------------------------------------------
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN > /dev/null 2>&1; then
  fail "${PORT} 포트를 이미 누가 쓰고 있다. lsof -nP -iTCP:${PORT} -sTCP:LISTEN 로 확인"
fi
ok "${PORT} 포트 비어 있음"

echo
echo "기동  http://${HOST}:${PORT}  (문서 /docs)"
echo "터널이 붙는 주소도 이 포트다. cloudflared 는 별도 창에서 돌린다."
echo

# 0.0.0.0 으로 열지 않는다. 외부 노출은 터널만 담당한다.
# 그래야 같은 공유기의 다른 기기가 터널을 우회해 붙는 경로가 생기지 않는다.
exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" ${RELOAD}
