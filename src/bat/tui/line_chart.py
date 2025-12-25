from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from rich.text import Text
from textual.widget import Widget

SeriesSpec = Tuple[Iterable[Optional[float]], str, str]


class LineChart(Widget):
    DEFAULT_CSS = """
LineChart {
    height: 6;
    width: 100%;
    color: $text;
    background: $surface;
}
"""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._series: Dict[str, SeriesSpec] = {}

    def set_series(self, series: Dict[str, SeriesSpec]) -> None:
        self._series = series
        self.refresh()

    def render(self) -> Text:
        width = max(int(self.size.width), 10)
        height = max(int(self.size.height), 4)
        if not self._series:
            return Text("無資料", style="dim")

        all_values = []
        for values, _, _ in self._series.values():
            all_values.extend([v for v in values if v is not None])
        if not all_values:
            return Text("無資料", style="dim")

        min_val = min(all_values)
        max_val = max(all_values)
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0
        pad = (max_val - min_val) * 0.05
        min_val -= pad
        max_val += pad

        grid = [[(" ", None) for _ in range(width)] for _ in range(height)]

        for values, style, marker in self._series.values():
            sampled = _resample(values, width)
            prev_y = None
            for x, val in enumerate(sampled):
                if val is None:
                    prev_y = None
                    continue
                y = _scale_value(val, min_val, max_val, height)
                grid[y][x] = (marker, style)
                if prev_y is not None and prev_y != y:
                    y_start, y_end = sorted((prev_y, y))
                    for y_fill in range(y_start, y_end + 1):
                        if grid[y_fill][x][0] == " ":
                            grid[y_fill][x] = ("│", style)
                prev_y = y

        text = Text()
        for row in grid:
            row_text = Text()
            for ch, style in row:
                if style:
                    row_text.append(ch, style=style)
                else:
                    row_text.append(ch)
            text.append_text(row_text)
            text.append("\n")
        return text


def _scale_value(value: float, min_val: float, max_val: float, height: int) -> int:
    if max_val == min_val:
        return height - 1
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(ratio, 1.0))
    return int(round((height - 1) * (1.0 - ratio)))


def _resample(values: Iterable[Optional[float]], width: int) -> list[Optional[float]]:
    data = list(values)
    if not data:
        return [None] * width
    if len(data) == 1:
        return [data[0]] * width
    if width <= 1:
        return [data[-1]]
    step = (len(data) - 1) / (width - 1)
    sampled = []
    for i in range(width):
        idx = int(round(i * step))
        sampled.append(data[min(idx, len(data) - 1)])
    return sampled
