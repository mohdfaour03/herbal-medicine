from herbs_eval.ui import build_ui

# Load .env for API keys if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def main():
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
