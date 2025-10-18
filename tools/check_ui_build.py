import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import herbs_eval.ui as ui


if __name__ == "__main__":
    ui.build_ui()
    print("UI_BUILD_OK")
