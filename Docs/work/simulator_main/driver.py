"""Driver template for a pygame app.

Why this exists: pygame apps draw multiple regions per frame (toolbar, main
canvas, side panels, overlays). If we only call one of the app's private
``_draw_*`` methods we get a screenshot where most regions are black. The
correct approach is to let the app's own game loop execute for a brief
period, then screenshot the display surface. We achieve that by posting a
``pygame.QUIT`` event from a background thread after a short delay — the
app's ``run()`` loop sees it, finishes the current frame (including every
draw call), and returns. We screenshot after.

Template variables (same substitution scheme as other drivers):
    IMPORT_LINE   e.g. "from simulator import SimulatorApp"
    CONSTRUCT     e.g. "app = SimulatorApp()"  (MUST assign to name `app`)
    RUN_ATTR      name of the run method, e.g. "run" (default "run")
    SCREEN_NAME   human screen name
    OUT_DIR       absolute path for PNG + JSON
    SETUP_PRE     optional pre-construct setup (sys.path, env vars)
    RUN_SECONDS   how long to let the app run before QUITting (default 1.0)

The driver writes:
    screen.png      from pygame.display.get_surface() after run() returns
    widgets.json    schema-compatible with the Tk/Qt drivers. Since pygame
                    has no widget tree, the "widgets" array is populated by
                    scanning the app instance for ``*_rect`` attributes that
                    hold ``pygame.Rect`` values — those are conventionally
                    the clickable regions in pygame UIs.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import sys
sys.path.insert(0, r"C:\GITHUB Repos\RoboSim_Navigation_Trainer")



# Stub tkinter dialogs in case the app calls them during startup.
try:
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd
    _fd.askopenfilename   = lambda *a, **kw: ""
    _fd.askopenfilenames  = lambda *a, **kw: ()
    _fd.asksaveasfilename = lambda *a, **kw: ""
    _fd.askdirectory      = lambda *a, **kw: ""
    _mb.showinfo = _mb.showwarning = _mb.showerror = lambda *a, **kw: "ok"
    _mb.askyesno = _mb.askokcancel = _mb.askquestion = lambda *a, **kw: True
    _sd.askstring  = lambda *a, **kw: ""
    _sd.askinteger = lambda *a, **kw: 0
    _sd.askfloat   = lambda *a, **kw: 0.0
except Exception:
    pass

import pygame


# --- text-render capture -------------------------------------------------
# The pygame driver's biggest blind spot is text readouts drawn via
# pygame.font.Font.render(...): they have no Rect attribute on the app
# instance, so _snapshot_rect_widgets never sees them.
#
# pygame-ce makes both pygame.font.Font and pygame.Surface immutable C
# types, so we can't monkey-patch their methods at the class level. We can,
# however:
#   1. subclass pygame.font.Font, override render(), and replace the symbol
#      pygame.font.Font with our subclass — new Font(...) calls now use it;
#   2. subclass pygame.Surface to override blit(), and ask the app to draw
#      onto our subclassed surface rather than the real SDL display by
#      patching pygame.display.set_mode / get_surface / flip / update.
#      Each frame, our shadow display is blitted onto the real SDL display
#      just before flip/update so the actual window still paints correctly.
#
# Capture dedups by (x, y, w, h) — last-text-wins — so only the final
# frame's text layout is preserved (pygame apps typically redraw every
# frame, which would otherwise produce N duplicates per run).

_TEXT_SURFS: dict = {}      # id(text_surface) -> text string
_CAPTURED_TEXTS: dict = {}  # (x, y, w, h) -> text

_REAL_DISPLAY = None
_SHADOW_DISPLAY = None


def _install_text_capture():
    """Install the text-capture shim. Returns True if active."""
    try:
        _OrigFont = pygame.font.Font
        _orig_sysfont = pygame.font.SysFont

        class _TrackedFont(_OrigFont):
            def render(self, *args, **kwargs):
                surf = super().render(*args, **kwargs)
                text = args[0] if args else kwargs.get("text", "")
                if isinstance(text, str) and text.strip():
                    _TEXT_SURFS[id(surf)] = text
                return surf

        class _FontProxy:
            """Delegates every attribute to a real Font; intercepts render().
            Used when we can't construct a _TrackedFont directly (e.g. when
            pygame.font.SysFont resolves fallback fonts internally)."""
            def __init__(self, real):
                self._real = real
            def render(self, *args, **kwargs):
                surf = self._real.render(*args, **kwargs)
                text = args[0] if args else kwargs.get("text", "")
                if isinstance(text, str) and text.strip():
                    _TEXT_SURFS[id(surf)] = text
                return surf
            def __getattr__(self, name):
                return getattr(self._real, name)

        def patched_sysfont(*args, **kwargs):
            return _FontProxy(_orig_sysfont(*args, **kwargs))

        class _ShadowDisplay(pygame.Surface):
            """A normal pygame.Surface subclass used as the app's draw target.
            We can override blit here (unlike on pygame.Surface itself)."""
            def blit(self, source, dest, *args, **kwargs):
                rect = super().blit(source, dest, *args, **kwargs)
                try:
                    text = _TEXT_SURFS.get(id(source))
                    if text is not None:
                        if hasattr(dest, "x") and hasattr(dest, "y"):
                            x, y = int(dest.x), int(dest.y)
                        elif isinstance(dest, (list, tuple)) and len(dest) >= 2:
                            x, y = int(dest[0]), int(dest[1])
                        else:
                            x, y = int(rect.x), int(rect.y)
                        w, h = source.get_size()
                        if w > 1 and h > 1:
                            _CAPTURED_TEXTS[(x, y, int(w), int(h))] = text
                except Exception:
                    pass
                return rect

        orig_set_mode = pygame.display.set_mode
        orig_flip = pygame.display.flip
        orig_update = pygame.display.update
        orig_get_surface = pygame.display.get_surface

        def _sync_shadow_to_real():
            if _SHADOW_DISPLAY is not None and _REAL_DISPLAY is not None:
                try:
                    pygame.Surface.blit(_REAL_DISPLAY, _SHADOW_DISPLAY, (0, 0))
                except Exception:
                    pass

        def patched_set_mode(*args, **kwargs):
            global _REAL_DISPLAY, _SHADOW_DISPLAY
            _REAL_DISPLAY = orig_set_mode(*args, **kwargs)
            size = _REAL_DISPLAY.get_size()
            _SHADOW_DISPLAY = _ShadowDisplay(size, pygame.SRCALPHA)
            return _SHADOW_DISPLAY

        def patched_flip(*args, **kwargs):
            _sync_shadow_to_real()
            return orig_flip(*args, **kwargs)

        def patched_update(*args, **kwargs):
            _sync_shadow_to_real()
            return orig_update(*args, **kwargs)

        def patched_get_surface():
            return _SHADOW_DISPLAY if _SHADOW_DISPLAY is not None else orig_get_surface()

        pygame.font.Font = _TrackedFont
        pygame.font.SysFont = patched_sysfont
        pygame.display.set_mode = patched_set_mode
        pygame.display.flip = patched_flip
        pygame.display.update = patched_update
        pygame.display.get_surface = patched_get_surface
        return True
    except Exception:
        return False


def _current_shadow_surface():
    """Return the shadow display (or None)."""
    return _SHADOW_DISPLAY


def _group_captured_texts(img_w, img_h, rect_widgets):
    """Turn _CAPTURED_TEXTS into widget dicts.

    Drops text that's fully contained inside a known Rect widget — those
    are button/radio labels, already captured at the rect level. Remaining
    texts are standalone readouts (status panels, whisker lengths, etc.).
    """
    # Only filter text inside CONTROL rects (buttons/radios/checkboxes) —
    # not display/layout containers like `panel_rect`, which legitimately
    # wrap the readouts we want to label.
    control_rects = [r for r in rect_widgets if r.get("role") == "control"]

    def _inside_rect(tx, ty, tw, th):
        for rw in control_rects:
            if (rw["x"] <= tx and rw["y"] <= ty
                    and tx + tw <= rw["x"] + rw["w"]
                    and ty + th <= rw["y"] + rw["h"]):
                return True
        return False

    widgets = []
    for i, (k, text) in enumerate(sorted(_CAPTURED_TEXTS.items())):
        x, y, w, h = k
        if not (0 <= x < img_w and 0 <= y < img_h):
            continue
        if w < 6 or h < 6:
            continue
        if _inside_rect(x, y, w, h):
            continue
        sample = text.strip()
        if not sample:
            continue
        widgets.append({
            "id": f"text:{i}:{sample[:20].replace(' ', '_').replace(chr(39), '')}",
            "cls": "PygameText",
            "text": sample[:120],
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "role": "display",
        })
    return widgets


def _post_quit_after(delay_s):
    """Background thread: after delay, shove QUIT onto the pygame event queue."""
    time.sleep(delay_s)
    try:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    except Exception:
        pass


def _snapshot_rect_widgets(app):
    """Scan the app instance for ``pygame.Rect`` attributes — those are the
    clickable regions in virtually every pygame UI. Returns widget dicts in
    absolute screen coords (rect coords are already in pygame display space,
    and for standalone pygame apps the display origin == screen origin at
    the window's current screen position, which we fold in below).
    """
    widgets = []
    try:
        for name in dir(app):
            if name.startswith("__"):
                continue
            try:
                val = getattr(app, name)
            except Exception:
                continue
            if not isinstance(val, pygame.Rect):
                continue
            # Only take non-degenerate rects with something a user could click.
            if val.w <= 1 or val.h <= 1:
                continue
            # Role heuristic: names ending in *_rect on a button, radio,
            # checkbox → control. Others (status, readout, bar) → display.
            low = name.lower()
            is_control = any(
                k in low for k in ("button", "btn", "radio", "check", "slider", "dial", "toggle")
            )
            widgets.append({
                "id": f"rect:{name}",
                "cls": "PygameRect",
                "text": name.replace("_rect", "").replace("_", " "),
                "x": int(val.x),
                "y": int(val.y),
                "w": int(val.w),
                "h": int(val.h),
                "role": "control" if is_control else "display",
            })
    except Exception:
        pass
    return widgets


def main():
    out_dir = Path(r"C:\\GITHUB Repos\\RoboSim_Navigation_Trainer\\Docs3\\work\\simulator_main")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Install text-render capture BEFORE the app is constructed, so any text
    # the constructor draws during init is also captured.
    text_capture_ok = _install_text_capture()

    try:
        from simulator import SimulatorApp
        app = SimulatorApp()

        # Post QUIT after a delay so the app's own run() loop exits after a
        # few full frames. 1.0s at 60 FPS = 60 full draw cycles — plenty.
        run_seconds = float("1.0" or "1.0")
        threading.Thread(target=_post_quit_after, args=(run_seconds,), daemon=True).start()

        # Many pygame apps call pygame.quit() at the bottom of their run()
        # method, which invalidates the display surface. If we screenshot
        # AFTER run() returns, pygame.image.save segfaults on a dead surface.
        # Stub pygame.quit to a no-op for the duration of run() so the
        # surface stays alive; we do the real teardown ourselves at the end.
        _real_quit = pygame.quit
        pygame.quit = lambda: None

        # Run the app's own main loop — this is what actually triggers the
        # full draw path (top bar + canvas + side panel + overlays).
        run_method = getattr(app, "run", None)
        if run_method is None or not callable(run_method):
            raise RuntimeError(
                "pygame app must expose a run() method (or set run). "
                "Called attribute not found."
            )
        try:
            run_method()
        finally:
            pygame.quit = _real_quit

        # Grab one final frame just in case the app's last loop iteration
        # cleared the screen before breaking out.
        surf = pygame.display.get_surface()
        if surf is None:
            surf = getattr(app, "screen", None)
        if surf is None:
            raise RuntimeError("no pygame display surface available to screenshot")

        png_path = out_dir / "screen.png"
        pygame.image.save(surf, str(png_path))
        img_w, img_h = surf.get_size()

        widgets = _snapshot_rect_widgets(app)
        if text_capture_ok:
            widgets.extend(_group_captured_texts(img_w, img_h, widgets))

        # pygame's display surface uses origin (0, 0) internally, so widget
        # rect coords are already image-local. origin_x/origin_y == 0.
        meta = {
            "screen_name": "simulator_main",
            "image_width": int(img_w),
            "image_height": int(img_h),
            "scaling_factor": 1.0,
            "origin_x": 0,
            "origin_y": 0,
            "foreground_ok": True,
            "runtime_seconds": run_seconds,
            "text_capture_enabled": text_capture_ok,
            "widgets": widgets,
        }
        (out_dir / "widgets.json").write_text(json.dumps(meta, indent=2))

        try:
            pygame.quit()
        except Exception:
            pass
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
