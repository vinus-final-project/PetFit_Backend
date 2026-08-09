"""분석 API DTO 검증.

API 명세서의 예시 JSON을 기준값으로 삼는다.
필드명·타입·정렬·시각 형식이 어긋나면 프론트엔드 연동이 깨진다.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.core.exceptions import ErrorCode, PetFitError
from app.schemas.analysis import (
    AnalysisCreateForm,
    AnalysisDetailOut,
    AnalysisListOut,
    AnalysisListQuery,
    AnalysisStatusOut,
    AnalysisSummaryOut,
    AnimalListResponse,
    DetectedObjectOut,
    ErrorResponse,
    Pagination,
    RiskFactorOut,
    SpaceListResponse,
    sort_detected_objects,
    sort_risk_factors,
    validate_device_id,
)
from app.schemas.enums import AnalysisStatus, RiskLevel, RiskSource

DEVICE_ID = "3f2b8c10-9d7e-4a51-8f6c-2e4b7a9d0c35"


def row(**kwargs):
    """analysis 행을 흉내 내는 객체."""
    base = dict(
        analysis_id=1,
        device_id=DEVICE_ID,
        animal_group="small_dog",
        space_type="living_room",
        status="COMPLETED",
        stage=None,
        progress=100,
        error_message=None,
        retry_count=0,
        thumbnail_path="/images/8f14e45f-ceea-467a-9c2b-1d3a7f6b90e1.jpg",
        total_score=56,
        safety_score=50,
        activity_score=67,
        rest_score=60,
        environment_score=50,
        risk_factors=[],
        analysis_result=[],
        # DB에는 KST 기준 naive 값이 저장된다.
        created_at=datetime(2026, 8, 6, 14, 30, 0),
        completed_at=datetime(2026, 8, 6, 14, 30, 42),
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def obj(object_id, name, risk, confidence, marked=None):
    return SimpleNamespace(
        object_id=object_id,
        object_name=name,
        risk_level=risk,
        confidence=confidence,
        marked_image_path=marked,
    )


# =============================================================================


class TestCamelCase:
    def test_response_keys_are_camel_case(self) -> None:
        """프론트엔드 계약이다. snake_case가 새어 나가면 안 된다."""
        dumped = AnalysisSummaryOut.from_model(row(), can_retry=False).model_dump(
            by_alias=True
        )
        assert set(dumped) == {
            "analysisId",
            "animalGroup",
            "spaceType",
            "status",
            "petFitScore",
            "thumbnailImage",
            "canRetry",
            "createdAt",
        }

    def test_no_snake_case_leaks_in_detail(self) -> None:
        dumped = AnalysisDetailOut.from_model(row(), [], []).model_dump(by_alias=True)
        assert not any("_" in key for key in dumped)


class TestTimezone:
    def test_serialized_format_matches_spec(self) -> None:
        dumped = AnalysisSummaryOut.from_model(row(), can_retry=False).model_dump(
            by_alias=True
        )
        assert dumped["createdAt"] == "2026-08-06T14:30:00+09:00"


class TestMetaApi:
    def test_animals_excludes_reserved_groups(self) -> None:
        """확장 그룹은 분석 기준이 없어 노출하지 않는다."""
        payload = AnimalListResponse.build().model_dump(by_alias=True)
        assert payload == {
            "animals": [
                {"code": "small_dog", "name": "소형견"},
                {"code": "large_dog", "name": "중·대형견"},
                {"code": "cat", "name": "고양이"},
            ]
        }

    def test_spaces(self) -> None:
        payload = SpaceListResponse.build().model_dump(by_alias=True)
        assert payload == {
            "spaces": [
                {"code": "living_room", "name": "거실"},
                {"code": "bedroom", "name": "침실"},
                {"code": "kitchen", "name": "주방"},
                {"code": "balcony", "name": "베란다"},
            ]
        }


class TestDeviceId:
    def test_valid_uuid_v4(self) -> None:
        assert validate_device_id(DEVICE_ID) == DEVICE_ID

    def test_whitespace_stripped(self) -> None:
        assert validate_device_id(f"  {DEVICE_ID}  ") == DEVICE_ID

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_missing(self, raw) -> None:
        with pytest.raises(PetFitError) as e:
            validate_device_id(raw)
        assert e.value.code is ErrorCode.DEVICE_ID_REQUIRED

    @pytest.mark.parametrize(
        "raw",
        [
            "not-a-uuid",
            "3f2b8c10",
            "3f2b8c10-9d7e-1a51-8f6c-2e4b7a9d0c35",  # v1
        ],
    )
    def test_invalid(self, raw: str) -> None:
        with pytest.raises(PetFitError) as e:
            validate_device_id(raw)
        assert e.value.code is ErrorCode.DEVICE_ID_INVALID


class TestCreateForm:
    def test_valid(self) -> None:
        group, space = AnalysisCreateForm(
            animal_group="small_dog", space_type="living_room"
        ).validated()
        assert (group.value, space.value) == ("small_dog", "living_room")

    def test_missing_animal_group(self) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisCreateForm(space_type="living_room").validated()
        assert e.value.code is ErrorCode.ANIMAL_GROUP_REQUIRED
        assert e.value.field == "animalGroup"

    @pytest.mark.parametrize("group", ["small_animal", "bird", "reptile"])
    def test_reserved_group_rejected(self, group: str) -> None:
        """예약 그룹은 Enum에는 있지만 분석 요청은 거부한다."""
        with pytest.raises(PetFitError) as e:
            AnalysisCreateForm(animal_group=group, space_type="living_room").validated()
        assert e.value.code is ErrorCode.ANIMAL_GROUP_UNSUPPORTED

    def test_unknown_group(self) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisCreateForm(animal_group="dragon", space_type="living_room").validated()
        assert e.value.code is ErrorCode.ANIMAL_GROUP_UNSUPPORTED

    def test_missing_space_type(self) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisCreateForm(animal_group="cat").validated()
        assert e.value.code is ErrorCode.SPACE_TYPE_REQUIRED

    def test_invalid_space_type(self) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisCreateForm(animal_group="cat", space_type="garage").validated()
        assert e.value.code is ErrorCode.SPACE_TYPE_INVALID


class TestListQuery:
    def test_defaults(self) -> None:
        assert AnalysisListQuery().validated() == (1, 20, None)

    def test_status_filter(self) -> None:
        _, _, status = AnalysisListQuery(status="COMPLETED").validated()
        assert status is AnalysisStatus.COMPLETED

    @pytest.mark.parametrize("page", [0, -1])
    def test_page_invalid(self, page: int) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisListQuery(page=page).validated()
        assert e.value.code is ErrorCode.PAGE_INVALID

    @pytest.mark.parametrize("size", [0, 51, 100])
    def test_size_invalid(self, size: int) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisListQuery(size=size).validated()
        assert e.value.code is ErrorCode.SIZE_INVALID

    def test_size_boundary(self) -> None:
        assert AnalysisListQuery(size=50).validated()[1] == 50

    def test_status_invalid(self) -> None:
        with pytest.raises(PetFitError) as e:
            AnalysisListQuery(status="RUNNING").validated()
        assert e.value.code is ErrorCode.STATUS_INVALID


class TestSorting:
    def test_detected_object_order(self) -> None:
        """risk → confidence → 이름 → 식별자 순으로 정렬한다."""
        items = [
            (3, DetectedObjectOut(name="소파", risk=RiskLevel.SAFE, confidence=0.98)),
            (1, DetectedObjectOut(name="전선", risk=RiskLevel.HIGH, confidence=0.94)),
            (2, DetectedObjectOut(name="창문", risk=RiskLevel.LOW, confidence=0.91)),
            (4, DetectedObjectOut(name="카펫", risk=RiskLevel.SAFE, confidence=0.89)),
        ]
        assert [o.name for o in sort_detected_objects(items)] == [
            "전선",
            "창문",
            "소파",
            "카펫",
        ]

    def test_korean_name_tiebreak(self) -> None:
        """동일 risk·confidence 에서는 가나다순이다."""
        items = [
            (1, DetectedObjectOut(name="침대", risk=RiskLevel.SAFE, confidence=0.9)),
            (2, DetectedObjectOut(name="가구", risk=RiskLevel.SAFE, confidence=0.9)),
            (3, DetectedObjectOut(name="소파", risk=RiskLevel.SAFE, confidence=0.9)),
        ]
        assert [o.name for o in sort_detected_objects(items)] == ["가구", "소파", "침대"]

    def test_identifier_tiebreak_makes_order_deterministic(self) -> None:
        """동일 객체가 여러 번 탐지되어도 순서가 고정된다."""
        items = [
            (9, DetectedObjectOut(name="전선", risk=RiskLevel.HIGH, confidence=0.9)),
            (2, DetectedObjectOut(name="전선", risk=RiskLevel.HIGH, confidence=0.9)),
        ]
        assert sort_detected_objects(items) == sort_detected_objects(list(reversed(items)))

    def test_risk_factor_order(self) -> None:
        """DETECTED가 먼저 오고 같은 source 안에서는 생성 순서를 유지한다."""
        factors = [
            RiskFactorOut(text="A", source=RiskSource.OBSERVED),
            RiskFactorOut(text="B", source=RiskSource.DETECTED),
            RiskFactorOut(text="C", source=RiskSource.OBSERVED),
            RiskFactorOut(text="D", source=RiskSource.DETECTED),
        ]
        assert [f.text for f in sort_risk_factors(factors)] == ["B", "D", "A", "C"]


class TestStatusResponse:
    def test_processing(self) -> None:
        payload = AnalysisStatusOut.from_model(
            row(status="PROCESSING", stage="OBJECT_DETECTION", progress=20),
            can_retry=False,
        ).model_dump(by_alias=True, mode="json")
        assert payload == {
            "analysisId": 1,
            "status": "PROCESSING",
            "stage": "OBJECT_DETECTION",
            "progress": 20,
            "retryCount": 0,
            "canRetry": False,
            "message": None,
        }

    def test_completed_progress_is_100(self) -> None:
        payload = AnalysisStatusOut.from_model(
            row(status="COMPLETED", stage=None, progress=82), can_retry=False
        ).model_dump(by_alias=True, mode="json")
        assert payload["progress"] == 100
        assert payload["stage"] is None

    def test_failed_keeps_stage_and_progress(self) -> None:
        """실패 지점을 남겨야 재촬영과 재시도를 구분해 안내할 수 있다."""
        payload = AnalysisStatusOut.from_model(
            row(
                status="FAILED",
                stage="OBJECT_DETECTION",
                progress=20,
                retry_count=1,
                error_message="객체 탐지에 실패했습니다.",
            ),
            can_retry=True,
        ).model_dump(by_alias=True, mode="json")
        assert payload == {
            "analysisId": 1,
            "status": "FAILED",
            "stage": "OBJECT_DETECTION",
            "progress": 20,
            "retryCount": 1,
            "canRetry": True,
            "message": "객체 탐지에 실패했습니다.",
        }


class TestSummary:
    def test_completed_includes_score(self) -> None:
        payload = AnalysisSummaryOut.from_model(row(), can_retry=False).model_dump(
            by_alias=True, mode="json"
        )
        assert payload["petFitScore"] == {
            "total": 56,
            "safety": 50,
            "activity": 67,
            "rest": 60,
            "environment": 50,
        }
        assert payload["thumbnailImage"].startswith("/images/")

    @pytest.mark.parametrize("status", ["PENDING", "PROCESSING", "FAILED"])
    def test_incomplete_hides_score_and_thumbnail(self, status: str) -> None:
        """COMPLETED가 아니면 점수와 썸네일은 null이다."""
        payload = AnalysisSummaryOut.from_model(
            row(status=status), can_retry=(status == "FAILED")
        ).model_dump(by_alias=True, mode="json")
        assert payload["petFitScore"] is None
        assert payload["thumbnailImage"] is None


class TestDetail:
    def test_matches_spec_example(self) -> None:
        detail = AnalysisDetailOut.from_model(
            row(
                risk_factors=[
                    {"text": "화분이 놓여 있습니다.", "source": "OBSERVED"},
                    {"text": "전선이 노출되어 있습니다.", "source": "DETECTED"},
                ],
                analysis_result=["활동 공간은 충분하지만 전선이 위험 요소입니다."],
            ),
            objects=[
                obj(1, "전선", "HIGH", 0.94, "/images/a.jpg"),
                obj(2, "창문", "LOW", 0.91, "/images/b.jpg"),
                obj(3, "카펫", "SAFE", 0.89),
                obj(4, "소파", "SAFE", 0.98),
            ],
            recommendations=[
                SimpleNamespace(
                    recommendation_type="SAFETY",
                    recommendation_text="화분을 옮겨주세요.",
                    priority=2,
                    source="OBSERVED",
                ),
                SimpleNamespace(
                    recommendation_type="SAFETY",
                    recommendation_text="전선을 정리해주세요.",
                    priority=1,
                    source="DETECTED",
                ),
            ],
        )
        payload = detail.model_dump(by_alias=True, mode="json")

        assert [o["name"] for o in payload["detectedObjects"]] == [
            "전선",
            "창문",
            "소파",
            "카펫",
        ]
        assert payload["detectedObjects"][2]["markedImage"] is None
        assert [f["source"] for f in payload["riskFactors"]] == ["DETECTED", "OBSERVED"]
        assert [r["priority"] for r in payload["recommendations"]] == [1, 2]
        assert payload["completedAt"] == "2026-08-06T14:30:42+09:00"

    def test_empty_collections_are_arrays_not_null(self) -> None:
        payload = AnalysisDetailOut.from_model(row(), [], []).model_dump(by_alias=True)
        assert payload["detectedObjects"] == []
        assert payload["riskFactors"] == []
        assert payload["analysis"] == []
        assert payload["recommendations"] == []

    def test_null_json_columns_tolerated(self) -> None:
        """JSON 컬럼이 NULL이어도 빈 배열로 처리한다."""
        payload = AnalysisDetailOut.from_model(
            row(risk_factors=None, analysis_result=None), [], []
        ).model_dump(by_alias=True)
        assert payload["riskFactors"] == []
        assert payload["analysis"] == []


class TestPagination:
    @pytest.mark.parametrize(
        "page, size, total, pages, has_next",
        [
            (1, 20, 3, 1, False),
            (1, 20, 20, 1, False),
            (1, 20, 21, 2, True),
            (2, 20, 21, 2, False),
            (1, 20, 0, 0, False),
            (1, 10, 100, 10, True),
        ],
    )
    def test_build(self, page, size, total, pages, has_next) -> None:
        p = Pagination.build(page, size, total)
        assert (p.total_pages, p.has_next) == (pages, has_next)

    def test_empty_list_returns_array(self) -> None:
        payload = AnalysisListOut(
            analyses=[], pagination=Pagination.build(1, 20, 0)
        ).model_dump(by_alias=True)
        assert payload["analyses"] == []


class TestErrorResponse:
    def test_matches_spec_example(self) -> None:
        payload = ErrorResponse(
            code="ANIMAL_GROUP_UNSUPPORTED",
            message="현재 지원하지 않는 반려동물 그룹입니다.",
            field="animalGroup",
        ).model_dump(by_alias=True, mode="json")
        assert payload == {
            "code": "ANIMAL_GROUP_UNSUPPORTED",
            "message": "현재 지원하지 않는 반려동물 그룹입니다.",
            "field": "animalGroup",
            "status": None,
        }

    def test_conflict_includes_status(self) -> None:
        payload = ErrorResponse(
            code="ANALYSIS_IN_PROGRESS",
            message="이미 진행 중인 분석이 있습니다.",
            status=AnalysisStatus.PROCESSING,
        ).model_dump(by_alias=True, mode="json")
        assert payload["status"] == "PROCESSING"

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_petfit_error_matches_schema(self, code: ErrorCode) -> None:
        """모든 오류 코드가 ErrorResponse 스키마로 검증을 통과해야 한다."""
        body = PetFitError(code).to_response()
        parsed = ErrorResponse.model_validate(body)
        assert parsed.code == code.value
        assert parsed.message

    def test_error_body_keys_match_schema_aliases(self) -> None:
        aliases = {f.alias or n for n, f in ErrorResponse.model_fields.items()}
        assert set(PetFitError(ErrorCode.INTERNAL_ERROR).to_response()) == aliases
