import re
import unittest
from pathlib import Path


STYLE_PATH = Path(__file__).resolve().parents[1] / "src" / "bat" / "tui" / "styles.tcss"


def css_block(selector: str) -> str:
    css = STYLE_PATH.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.DOTALL)
    if not match:
        raise AssertionError(f"Missing CSS block for {selector}")
    return match.group("body")


class TrainLayoutCssTest(unittest.TestCase):
    def test_train_input_columns_are_independently_scrollable(self):
        block = css_block("#train_controls_left,\n#train_controls_right")

        self.assertIn("height: 100%", block)
        self.assertIn("overflow-y: auto", block)

    def test_train_actions_panel_is_scrollable(self):
        block = css_block("#train_actions")

        self.assertIn("height:", block)
        self.assertIn("overflow-y: auto", block)


if __name__ == "__main__":
    unittest.main()
