try:
    import audioop
    print("AUDIOOP_OK", getattr(audioop, "__doc__", "")[:30])
except Exception as e:
    print("AUDIOOP_ERR", type(e).__name__, str(e))

