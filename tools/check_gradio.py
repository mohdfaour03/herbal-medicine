try:
    import gradio  # noqa: F401
    print("GRADIO_OK")
except Exception as e:
    print("GRADIO_ERR:", type(e).__name__, str(e))

