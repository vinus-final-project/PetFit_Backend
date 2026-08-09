"""Qwen3-VL 어댑터 검증.

**mlx-vlm 없이 검증한다.** 이 패키지는 Apple Silicon macOS 에서만 설치되므로,
개발 PC에서 돌지 않는 테스트를 두면 아무도 실행하지 않는 테스트가 된다.

검증 대상은 변환이다. 대화 구성·이미지 전달·응답 추출·오류 처리는 모델 없이
전부 확인할 수 있고, 버그도 실제로 여기서 난다. 추론 자체의 품질은 성능평가에서
사람이 판정한다.

`YoloDetector` 를 ultralytics 없이 검증하는 것과 같은 구조다.
"""

import json

import pytest

from app.ai.llm.base import LLMError, VisionLLM
from app.ai.llm.qwen_mlx import (
    DEFAULT_MODEL,
    MISSING_PACKAGE,
    QwenMLX,
    _image_files,
    build_messages,
    extract_text,
)
from app.ai.pipeline import DetectedObject
from app.ai.prompts import AnalysisContext, build
from app.ai.score_generator import generate
from app.schemas.enums import AnimalGroup, RiskLevel, SpaceType


def context() -> AnalysisContext:
    objects = [
        DetectedObject(
            name="전선",
            risk=RiskLevel.HIGH,
            confidence=0.94,
            detection_frame_count=12,
            frame_number=3,
            x=0.1,
            y=0.7,
            width=0.2,
            height=0.08,
        )
    ]
    group, space = AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM
    return AnalysisContext(
        group=group,
        space=space,
        objects=objects,
        occupancy_ratio=0.35,
        score=generate(group, space, {"전선"}, 0.35),
        thumbnail_frame=7,
    )


# =============================================================================
# 대화 구성
# =============================================================================


class TestMessages:
    async def test_system_comes_first(self) -> None:
        messages = build_messages(build(context()))
        assert messages[0]["role"] == "system"

    async def test_group_instruction_is_in_system(self) -> None:
        messages = build_messages(build(context()))
        assert "소형견 기준으로 분석합니다" in messages[0]["content"]

    async def test_few_shots_become_turns(self) -> None:
        """예시를 지시문에 이어 붙이면 모델이 분석 대상으로 오해한다."""
        messages = build_messages(build(context()))

        roles = [m["role"] for m in messages]
        assert roles == ["system"] + ["user", "assistant"] * 3 + ["user"]

    async def test_last_turn_is_the_real_input(self) -> None:
        prompt = build(context())
        messages = build_messages(prompt)

        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == prompt.user

    async def test_few_shot_content_is_valid_json(self) -> None:
        messages = build_messages(build(context()))
        for message in messages[1:-1]:
            assert isinstance(json.loads(_json_part(message["content"])), dict)

    async def test_without_few_shots(self) -> None:
        prompt = build(context(), include_few_shots=False)
        messages = build_messages(prompt)

        assert [m["role"] for m in messages] == ["system", "user"]

    async def test_every_message_has_role_and_content(self) -> None:
        for message in build_messages(build(context())):
            assert set(message) == {"role", "content"}
            assert message["content"]


def _json_part(text: str) -> str:
    end = text.find("\n\n##")
    return text if end == -1 else text[:end]


# =============================================================================
# 응답 추출
# =============================================================================


class TestExtractText:
    async def test_plain_string(self) -> None:
        assert extract_text('{"a": 1}') == '{"a": 1}'

    @pytest.mark.parametrize("attribute", ["text", "output", "content"])
    async def test_object_with_text_attribute(self, attribute) -> None:
        """mlx-vlm 은 버전에 따라 문자열 또는 객체를 돌려준다."""
        result = type("Result", (), {attribute: '{"a": 1}'})()
        assert extract_text(result) == '{"a": 1}'

    async def test_unknown_shape_falls_back_to_str(self) -> None:
        assert extract_text(42) == "42"

    async def test_response_is_not_altered(self) -> None:
        """어댑터가 응답을 손질하면 형식 준수율을 측정할 수 없다."""
        raw = '```json\n{"a": 1}\n```'
        assert extract_text(raw) == raw


# =============================================================================
# 이미지 전달
# =============================================================================


class TestImageFiles:
    async def test_writes_each_image(self) -> None:
        with _image_files([b"first", b"second"]) as paths:
            assert len(paths) == 2
            assert open(paths[0], "rb").read() == b"first"

    async def test_order_is_preserved(self) -> None:
        """첫 장이 분석 대표 프레임이다. 순서가 바뀌면 맥락이 어긋난다."""
        with _image_files([b"a", b"b", b"c"]) as paths:
            contents = [open(p, "rb").read() for p in paths]
        assert contents == [b"a", b"b", b"c"]

    async def test_empty_gives_no_paths(self) -> None:
        with _image_files([]) as paths:
            assert paths == []

    async def test_blank_entries_are_skipped(self) -> None:
        with _image_files([b"a", b"", b"b"]) as paths:
            assert len(paths) == 2

    async def test_files_are_removed_after_use(self) -> None:
        """생활공간 사진이 임시 폴더에 남으면 안 된다."""
        import os

        with _image_files([b"data"]) as paths:
            kept = paths[0]
            assert os.path.exists(kept)
        assert not os.path.exists(kept)

    async def test_cleanup_runs_on_error(self) -> None:
        import os

        kept = None
        with pytest.raises(RuntimeError):
            with _image_files([b"data"]) as paths:
                kept = paths[0]
                raise RuntimeError("생성 실패")
        assert not os.path.exists(kept)


# =============================================================================
# 규약과 설정
# =============================================================================


class TestContract:
    async def test_satisfies_protocol(self) -> None:
        assert isinstance(QwenMLX(), VisionLLM)

    async def test_default_model_matches_guide(self) -> None:
        assert DEFAULT_MODEL == "mlx-community/Qwen3-VL-32B-Instruct-4bit"

    async def test_name_reports_model(self) -> None:
        assert QwenMLX().name == DEFAULT_MODEL
        assert QwenMLX("local/path").name == "local/path"

    async def test_construction_needs_no_package(self) -> None:
        """생성만으로 가중치를 읽으면 앱이 뜨지 않는다. 첫 호출에 읽는다."""
        QwenMLX()


# =============================================================================
# 미설치 환경
# =============================================================================


class TestMissingPackage:
    async def test_call_raises_llm_error(self) -> None:
        """개발 PC(Windows)에는 mlx-vlm 이 없다. 그때 나는 오류를 고정한다."""
        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("mlx-vlm 이 설치된 환경이다")

        with pytest.raises(LLMError) as exc:
            await QwenMLX().complete(build(context()), [])

        assert MISSING_PACKAGE in str(exc.value)

    async def test_message_tells_how_to_install(self) -> None:
        assert "requirements-ai.txt" in MISSING_PACKAGE
        assert "Apple Silicon" in MISSING_PACKAGE
