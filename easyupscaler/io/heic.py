_heif_registered = False


def ensure_heif_registered() -> None:
    global _heif_registered
    if _heif_registered:
        return
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _heif_registered = True
