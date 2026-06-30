"""
Stub egl_probe for Windows — returns an empty device list so robomimic
falls back to the default GPU device instead of trying EGL detection.
"""
def get_available_devices():
    return []
