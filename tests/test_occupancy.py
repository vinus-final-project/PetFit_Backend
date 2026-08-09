"""활동 공간 점유율 검증.

합집합 면적 자체는 ``test_geometry.py`` 가 이미 검증한다. 여기서 보는 것은
**프레임 단위 처리**다. 어떤 프레임을 세고, 여러 프레임을 어떻게 하나로 줄이는가.

점유율은 활동성 점수의 유일한 입력이다. 임계값 0.40 / 0.60 / 0.75 를 넘으면
감점률이 0.0 에서 0.4, 0.7, 1.0 으로 계단식으로 뛴다.
"""

import pytest

from app.ai.vision.occupancy import frame_ratios, occupancy_ratio
from app.ai.vision.types import Detection
from app.rules.penalty import occupancy_penalty


def det(x: float, y: float, w: float, h: float, frame: int = 0) -> Detection:
    return Detection("sofa", 0.9, frame, x, y, w, h)


class TestFrameRatios:
    def test_one_ratio_per_frame(self) -> None:
        assert len(frame_ratios([[det(0, 0, 0.5, 0.5)], [], [det(0, 0, 0.2, 0.2)]])) == 3

    def test_empty_frame_is_zero_not_skipped(self) -> None:
        """빈 벽이나 바닥을 비춘 구간도 활동 공간의 일부다.

        제외하면 가구가 많이 잡힌 프레임만 남아 점유율이 실제보다 높아진다.
        """
        assert frame_ratios([[], []]) == [0.0, 0.0]

    def test_no_frames_yields_no_ratios(self) -> None:
        assert frame_ratios([]) == []

    def test_overlapping_boxes_counted_once(self) -> None:
        """중복 계산하면 점유율이 1.0을 넘는다."""
        box = det(0.1, 0.1, 0.4, 0.4)
        assert frame_ratios([[box, box]]) == [pytest.approx(0.16)]

    def test_disjoint_boxes_are_summed(self) -> None:
        row = [det(0.0, 0.0, 0.2, 0.2), det(0.5, 0.5, 0.2, 0.2)]
        assert frame_ratios([row]) == [pytest.approx(0.08)]

    def test_full_frame_box(self) -> None:
        assert frame_ratios([[det(0.0, 0.0, 1.0, 1.0)]]) == [pytest.approx(1.0)]


class TestOccupancyRatio:
    def test_no_frames(self) -> None:
        assert occupancy_ratio([]) == 0.0

    def test_all_empty_frames(self) -> None:
        assert occupancy_ratio([[], [], []]) == 0.0

    def test_single_frame(self) -> None:
        assert occupancy_ratio([[det(0.0, 0.0, 0.5, 0.5)]]) == pytest.approx(0.25)

    def test_uses_median_not_mean(self) -> None:
        """카메라가 소파를 정면으로 크게 비춘 프레임 하나가 평균을 끌어올린다."""
        rows = [
            [det(0.0, 0.0, 0.3, 0.3)],   # 0.09
            [det(0.0, 0.0, 0.3, 0.3)],   # 0.09
            [det(0.0, 0.0, 1.0, 1.0)],   # 1.00  <- 벽을 가득 채운 프레임
        ]
        # 평균이면 0.3933 으로 임계값 0.40 근처까지 올라간다.
        assert occupancy_ratio(rows) == pytest.approx(0.09)

    def test_even_count_averages_two_middles(self) -> None:
        rows = [
            [det(0.0, 0.0, 0.2, 0.2)],   # 0.04
            [det(0.0, 0.0, 0.4, 0.4)],   # 0.16
            [det(0.0, 0.0, 0.5, 0.5)],   # 0.25
            [det(0.0, 0.0, 0.6, 0.6)],   # 0.36
        ]
        assert occupancy_ratio(rows) == pytest.approx(0.205)

    def test_order_does_not_matter(self) -> None:
        rows = [[det(0.0, 0.0, 0.2, 0.2)], [det(0.0, 0.0, 0.6, 0.6)], []]
        assert occupancy_ratio(rows) == occupancy_ratio(list(reversed(rows)))


class TestContract:
    """DB 제약과 감점 규칙이 요구하는 조건."""

    def test_never_negative(self) -> None:
        assert occupancy_ratio([[], [det(0.0, 0.0, 0.1, 0.1)]]) >= 0.0

    def test_never_exceeds_one(self) -> None:
        """ck_analysis_occupancy_ratio 가 0~1 을 요구한다."""
        rows = [[det(0.0, 0.0, 1.0, 1.0), det(0.0, 0.0, 1.0, 1.0)] for _ in range(3)]
        assert occupancy_ratio(rows) <= 1.0

    def test_out_of_range_box_is_capped(self, caplog) -> None:
        """탐지기가 규약을 어겼을 때 DB 제약 위반 대신 경고를 남긴다.

        그대로 두면 분석 전체가 INSERT 단계에서 실패하고, 원인이 탐지기라는
        사실은 오류 메시지에 남지 않는다.
        """
        rows = [[Detection("sofa", 0.9, 0, 0.0, 0.0, 2.0, 2.0)]]
        assert occupancy_ratio(rows) == 1.0
        assert "1.0" in caplog.text or "초과" in caplog.text

    def test_precision_matches_the_column(self) -> None:
        """NUMERIC(5,4) 이므로 넷째 자리까지만 유지한다.

        저장 전후로 값이 달라지면 임계값 판정이 뒤집힐 수 있다.
        """
        rows = [[det(0.0, 0.0, 0.3333, 0.3333)]]
        value = occupancy_ratio(rows)
        assert value == round(value, 4)

    def test_feeds_the_penalty_rule(self) -> None:
        """산출값이 감점 규칙에 그대로 들어가야 한다."""
        rows = [[det(0.0, 0.0, 0.8, 0.8)]]           # 0.64
        assert occupancy_penalty(occupancy_ratio(rows)) == 0.7

    @pytest.mark.parametrize(
        "side,expected_penalty",
        [
            (0.6000, 0.0),   # 0.3600
            (0.7000, 0.4),   # 0.4900
            (0.8000, 0.7),   # 0.6400
            (0.9000, 1.0),   # 0.8100
        ],
    )
    def test_threshold_boundaries(self, side: float, expected_penalty: float) -> None:
        """계단이 갈리는 지점에서 산출값과 감점률이 맞물리는지 확인한다."""
        rows = [[det(0.0, 0.0, side, side)]]
        assert occupancy_penalty(occupancy_ratio(rows)) == expected_penalty


class TestWithDetector:
    """탐지기 출력을 변환 없이 받는지 확인한다."""

    def test_accepts_detector_output(self) -> None:
        from PIL import Image

        from app.ai.vision.detector import StubDetector
        from app.ai.vision.types import Frame

        frames = [Frame(i, i * 0.33, Image.new("RGB", (16, 9))) for i in range(25)]
        rows = StubDetector().detect(frames)

        value = occupancy_ratio(rows)
        assert 0.0 < value < 1.0
        assert len(frame_ratios(rows)) == 25
