try:
    from herbs_eval.ui import build_ui
    app = build_ui()
    print("UI_OK")
except Exception as e:
    print("UI_ERR", type(e).__name__, str(e))

