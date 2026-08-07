"""analysis 테이블 CHECK 제약 검증.

모델 정의를 눈으로 읽는 것만으로는 제약이 의도대로 동작하는지 알 수 없다.
모델에서 CHECK 표현식을 꺼내 SQL 엔진에 걸고 INSERT를 시도한다.

MySQL 대신 SQLite를 사용한다. MySQL 서버 없이 실행되어야 CI에서 돌릴 수 있고,
검증 대상인 CHECK 표현식은 두 엔진에서 동일하게 평가된다.
"""

import sqlite3

import pytest
from sqlalchemy import CheckConstraint

from app.models import Analysis
from app.schemas.enums import AnalysisStage, AnalysisStatus, progress_for

#: 상태 정합성을 다루는 제약. 이 테스트의 검증 대상이다.
TARGET_CONSTRAINTS = (
    "ck_analysis_status",
    "ck_analysis_stage",
    "ck_analysis_progress",
    "ck_analysis_stage_consistency",
    "ck_analysis_progress_consistency",
    "ck_analysis_completed_at",
    "ck_analysis_error_message",
)


def _check_expressions() -> dict[str, str]:
    """모델에 정의된 CHECK 표현식을 이름으로 찾아 반환한다."""
    return {
        c.name: str(c.sqltext)
        for c in Analysis.__table__.constraints
        if isinstance(c, CheckConstraint) and c.name
    }


@pytest.fixture(scope="module")
def con():
    """대상 제약만 적용한 SQLite 테이블을 만든다."""
    expressions = _check_expressions()
    missing = [n for n in TARGET_CONSTRAINTS if n not in expressions]
    assert not missing, f"모델에 없는 제약: {missing}"

    clauses = ", ".join(f"CHECK ({expressions[n]})" for n in TARGET_CONSTRAINTS)
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE analysis ("
        " status TEXT NOT NULL,"
        " stage TEXT,"
        " progress INTEGER NOT NULL,"
        " completed_at TEXT,"
        " error_message TEXT,"
        f" {clauses})"
    )
    return conn


#: 인자를 생략했음을 나타낸다. 명시적 None(값을 비운다)과 구별해야 한다.
_AUTO = object()


def insert(
    con,
    status: str,
    stage: str | None,
    progress: int,
    *,
    completed_at: str | None = _AUTO,
    error_message: str | None = _AUTO,
) -> bool:
    """INSERT를 시도하고 제약 통과 여부를 반환한다.

    `completed_at` 과 `error_message` 는 생략하면 상태에 맞는 값을 자동으로 채운다.
    명시적으로 None을 넘기면 비운 채로 시도한다.
    """
    if completed_at is _AUTO:
        completed_at = "2026-08-06 05:31:00" if status == "COMPLETED" else None
    if error_message is _AUTO:
        error_message = "분석에 실패했습니다." if status == "FAILED" else None
    try:
        con.execute(
            "INSERT INTO analysis VALUES (?,?,?,?,?)",
            (status, stage, progress, completed_at, error_message),
        )
        return True
    except sqlite3.IntegrityError:
        return False


class TestConstraintsExist:
    def test_all_target_constraints_defined(self) -> None:
        assert set(TARGET_CONSTRAINTS) <= set(_check_expressions())


class TestStageConsistency:
    def test_failed_preserves_stage(self, con) -> None:
        """API-006 실패 응답은 실패한 단계를 포함한다.

        이 제약이 잘못되면 실패 지점을 저장할 수 없다. 그러면 클라이언트가
        재촬영과 재시도 중 무엇을 안내할지 분기할 수 없다.
        """
        assert insert(con, "FAILED", "OBJECT_DETECTION", 20)

    def test_failed_before_pipeline_has_no_stage(self, con) -> None:
        """대기열 초과 등 파이프라인 진입 전 실패는 stage가 없다."""
        assert insert(con, "FAILED", None, 0)

    def test_processing_requires_stage(self, con) -> None:
        assert insert(con, "PROCESSING", "OBJECT_DETECTION", 20)
        assert not insert(con, "PROCESSING", None, 20)

    @pytest.mark.parametrize("status", ["PENDING", "COMPLETED"])
    def test_terminal_states_have_no_stage(self, con, status: str) -> None:
        progress = 100 if status == "COMPLETED" else 0
        assert not insert(con, status, "RISK_MARKING", progress)

    def test_unknown_stage_rejected(self, con) -> None:
        assert not insert(con, "PROCESSING", "UPLOADING", 20)


class TestProgressConsistency:
    def test_pending_is_zero(self, con) -> None:
        assert insert(con, "PENDING", None, 0)
        assert not insert(con, "PENDING", None, 10)

    def test_completed_is_100(self, con) -> None:
        assert insert(con, "COMPLETED", None, 100)
        assert not insert(con, "COMPLETED", None, 82)

    @pytest.mark.parametrize("progress", [-1, 101])
    def test_out_of_range_rejected(self, con, progress: int) -> None:
        assert not insert(con, "PROCESSING", "OBJECT_DETECTION", progress)


class TestStateColumns:
    def test_completed_requires_timestamp(self, con) -> None:
        assert not insert(con, "COMPLETED", None, 100, completed_at=None)

    def test_failed_requires_message(self, con) -> None:
        assert not insert(con, "FAILED", None, 0, error_message=None)

    def test_processing_has_neither(self, con) -> None:
        assert not insert(
            con, "PROCESSING", "OBJECT_DETECTION", 20, completed_at="2026-08-06 05:31:00"
        )
        assert not insert(
            con, "PROCESSING", "OBJECT_DETECTION", 20, error_message="실패"
        )

    def test_unknown_status_rejected(self, con) -> None:
        assert not insert(con, "CANCELLED", None, 0)


class TestProgressFor:
    """progress는 stage에서 파생된다. 이 함수가 유일한 정본이다."""

    def test_pending(self) -> None:
        assert progress_for(AnalysisStatus.PENDING, None) == 0

    def test_completed(self) -> None:
        assert progress_for(AnalysisStatus.COMPLETED, None) == 100

    @pytest.mark.parametrize("stage", list(AnalysisStage))
    def test_processing_matches_stage(self, stage: AnalysisStage) -> None:
        assert progress_for(AnalysisStatus.PROCESSING, stage) == stage.progress

    @pytest.mark.parametrize("stage", list(AnalysisStage))
    def test_failed_keeps_stage_progress(self, stage: AnalysisStage) -> None:
        """실패해도 진행률을 0으로 되돌리지 않는다.

        10에서 실패한 것과 82까지 진행하다 실패한 것을 구분해야 한다.
        """
        assert progress_for(AnalysisStatus.FAILED, stage) == stage.progress

    def test_failed_without_stage(self) -> None:
        assert progress_for(AnalysisStatus.FAILED, None) == 0

    def test_processing_without_stage_raises(self) -> None:
        with pytest.raises(ValueError):
            progress_for(AnalysisStatus.PROCESSING, None)

    @pytest.mark.parametrize("status", [AnalysisStatus.PROCESSING, AnalysisStatus.FAILED])
    @pytest.mark.parametrize("stage", list(AnalysisStage))
    def test_result_satisfies_db_constraint(
        self, con, status: AnalysisStatus, stage: AnalysisStage
    ) -> None:
        """함수가 낸 값이 DB 제약을 통과해야 한다.

        애플리케이션과 스키마가 서로 다른 규칙을 갖고 있지 않은지 확인한다.
        """
        assert insert(con, status.value, stage.value, progress_for(status, stage))
