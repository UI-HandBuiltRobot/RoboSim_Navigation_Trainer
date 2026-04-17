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
    from simulator import SimulatorApp   e.g. "from simulator import SimulatorApp"
    app = SimulatorApp()     e.g. "app = SimulatorApp()"  (MUST assign to name `app`)
    run      name of the run method, e.g. "run" (default "run")
    main_simulator   human screen name
    C:\\GITHUB Repos\\Auto_Labeler\\auto_doc\\work\\main_simulator       absolute path for PNG + JSON
    import sys
sys.path.insert(0, r"C:\GITHUB Repos\RoboSim_Navigation_Trainer")     optional pre-construct setup (sys.path, env vars)
    1.5   how long to let the app run before QUITting (default 1.0)

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
    out_dir = Path(r"C:\\GITHUB Repos\\Auto_Labeler\\auto_doc\\work\\main_simulator")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from simulator import SimulatorApp
        app = SimulatorApp()

        # Post QUIT after a delay so the app's own run() loop exits after a
        # few full frames. 1.0s at 60 FPS = 60 full draw cycles — plenty.
        run_seconds = float("1.5" or "1.0")
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

        # pygame's display surface uses origin (0, 0) internally, so widget
        # rect coords are already image-local. origin_x/origin_y == 0.
        meta = {
            "screen_name": "main_simulator",
            "image_width": int(img_w),
            "image_height": int(img_h),
            "scaling_factor": 1.0,
            "origin_x": 0,
            "origin_y": 0,
            "foreground_ok": True,
            "runtime_seconds": run_seconds,
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
