"""Driver for a Tkinter screen. Rendered from a template.

Template variables (rendered by auto_labeler before execution):
    IMPORT_LINE   e.g. "from myproject.ui import MainWindow"
    CONSTRUCT     e.g. "MainWindow(root)"
    SCREEN_NAME   human screen name
    OUT_DIR       absolute path where PNG + JSON are written
    SETUP_PRE     optional pre-construct setup (sys.path lines, env vars)

The driver writes into its output dir:
    screen.png      cropped to target window's rootx/rooty/width/height
    widgets.json    {image_width, image_height, scaling_factor, origin_x, origin_y, widgets: [...]}
Exits 0 on success, non-zero with traceback on stderr on failure.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Make the process DPI-aware BEFORE Tk initializes. Without this, on a
# high-DPI Windows display Tk reports logical coords (96 DPI virtualization)
# while mss captures physical pixels — the two frames disagree, and the
# crop rectangle lands on a fraction of the real window surrounded by
# whatever else is physically behind it. SetProcessDpiAwareness(1) makes Tk
# query the real system DPI; widgets scale naturally and winfo_rootx/width
# return physical coords matching what mss grabs.
if sys.platform.startswith("win"):
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware
        except (OSError, AttributeError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import sys
sys.path.insert(0, r"C:\GITHUB Repos\RoboSim_Navigation_Trainer")



import tkinter as tk
from tkinter import ttk

# Stub all blocking OS-level dialogs so driver captures never hang waiting
# for a human to interact with a file-picker or message box.
try:
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd
    _fd.askopenfilename   = lambda *a, **kw: ""
    _fd.askopenfilenames  = lambda *a, **kw: ()
    _fd.asksaveasfilename = lambda *a, **kw: ""
    _fd.askdirectory      = lambda *a, **kw: ""
    _mb.showinfo = _mb.showwarning = _mb.showerror = lambda *a, **kw: "ok"
    _mb.askyesno = _mb.askokcancel = _mb.askquestion = lambda *a, **kw: True
    _mb.askretrycancel = _mb.askyesnocancel = lambda *a, **kw: True
    _sd.askstring  = lambda *a, **kw: ""
    _sd.askinteger = lambda *a, **kw: 0
    _sd.askfloat   = lambda *a, **kw: 0.0
except Exception:
    pass


_CONTROL = (
    tk.Button, tk.Checkbutton, tk.Radiobutton, tk.Entry, tk.Listbox,
    tk.Scale, tk.Spinbox, tk.Menubutton,
    ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Entry, ttk.Combobox,
    ttk.Scale, ttk.Spinbox, ttk.Notebook,
)

# Display widgets: things that show information but do not accept input.
# We capture these too so the agent can label readouts, status fields, etc.
# tk.Text is included here even though it's technically editable — most apps
# use it as a readout.
_DISPLAY = (
    tk.Label, ttk.Label, tk.Message,
    ttk.Progressbar, tk.Canvas, tk.Text,
)


def _classify(widget):
    for cls in _CONTROL:
        if isinstance(widget, cls):
            return "control"
    for cls in _DISPLAY:
        if isinstance(widget, cls):
            return "display"
    return None  # skip pure layout widgets


def _safe_text(widget):
    try:
        opts = widget.keys()
    except Exception:
        return None
    for key in ("text", "label"):
        if key in opts:
            try:
                val = widget.cget(key)
                if val:
                    return str(val)
            except Exception:
                pass
    return None


def _callback_repr(widget):
    try:
        if "command" in widget.keys():
            cmd = widget.cget("command")
            return str(cmd) if cmd else None
    except Exception:
        return None
    return None


def _target_window(root):
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    if tops:
        return tops[-1]
    return root


def _walk(widget, out):
    for child in widget.winfo_children():
        role = _classify(child)
        if role is not None:
            try:
                text = _safe_text(child)
                # Skip display widgets that carry no user-visible content —
                # those are structural padding, not readouts worth labeling.
                if role == "display" and not text and not isinstance(
                    child, (ttk.Progressbar, tk.Canvas)
                ):
                    pass
                else:
                    out.append({
                        "id": str(child),
                        "cls": type(child).__name__,
                        "text": text,
                        "x": child.winfo_rootx(),
                        "y": child.winfo_rooty(),
                        "w": child.winfo_width(),
                        "h": child.winfo_height(),
                        "callback": _callback_repr(child),
                        "role": role,
                    })
            except Exception:
                pass
        _walk(child, out)


def _bring_to_front_win32(hwnd):
    """Use SetWindowPos(TOPMOST)→(NOTOPMOST)+SetForegroundWindow to force
    the window above any other app that may be in front. Returns True if we
    at least made the call without exception — actual foreground state should
    still be verified afterward."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
        SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def _image_has_content(png_bytes_or_path) -> bool:
    """Return True when the image has meaningful variance — not a solid
    fill that would indicate the compositor hasn't painted the window yet.
    Uses a cheap per-channel min/max check via PIL."""
    try:
        from PIL import Image
        img = Image.open(str(png_bytes_or_path)) if isinstance(png_bytes_or_path, Path) else Image.frombytes("RGB", (1,1), b"\x00\x00\x00")
        try:
            extrema = img.convert("RGB").getextrema()
        finally:
            img.close()
        # `extrema` is ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        spread = max(hi - lo for (lo, hi) in extrema)
        return spread >= 16  # anything below this is essentially solid
    except Exception:
        return True  # don't fail the capture on a probe error


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in (text or "").strip())
    out = out.strip("_")[:48]
    return out or "untitled"


def _enumerate_peer_toplevels(primary):
    """Return list of (toplevel_widget, slug) for every visible Tk Toplevel
    that is NOT the primary. Uses the Tk widget hierarchy rather than a gc
    scan — O(children) vs O(all Python objects), 50-100× faster.
    """
    out: list = []
    seen = {id(primary)}
    try:
        # Collect all tk.Tk root windows visible to Python via _default_root
        # and the primary itself, then walk their .winfo_children().
        roots = set()
        try:
            dr = tk._default_root
            if dr is not None:
                roots.add(dr)
        except Exception:
            pass
        roots.add(primary)

        def _walk_children(widget):
            try:
                children = widget.winfo_children()
            except Exception:
                return
            for child in children:
                if id(child) in seen:
                    continue
                seen.add(id(child))
                try:
                    if not isinstance(child, tk.Toplevel):
                        _walk_children(child)
                        continue
                    if not child.winfo_exists() or not child.winfo_viewable():
                        continue
                    w, h = int(child.winfo_width()), int(child.winfo_height())
                    if w < 100 or h < 60:
                        continue
                    title = ""
                    try: title = child.title() or ""
                    except Exception: pass
                    out.append((child, _slug(title) or f"sub_{id(child) & 0xffff:x}"))
                except Exception:
                    continue
                _walk_children(child)

        for root in roots:
            if id(root) not in seen or root is primary:
                _walk_children(root)
    except Exception:
        pass
    return out


def _capture_one(target, out_dir, screen_name, scaling) -> dict:
    """Bring `target` to front, wait for paint, screenshot, walk widgets.
    Retries on solid-color output up to 4 times with increasing sleeps.
    Writes screen.png + widgets.json into out_dir. Returns metadata dict."""
    import time as _time
    import mss, mss.tools

    out_dir.mkdir(parents=True, exist_ok=True)

    # Bring to front + force foreground
    try:
        target.lift()
        target.focus_force()
        try: target.attributes("-topmost", True)
        except Exception: pass
        target.update_idletasks(); target.update()
    except Exception:
        pass
    if sys.platform.startswith("win"):
        _bring_to_front_win32(_hwnd_of(target))

    png_path = out_dir / "screen.png"
    foreground_ok = True
    # Patient render loop: up to 5 attempts with escalating sleep.
    # On each attempt: re-settle events, sample geometry, grab pixels,
    # verify image variance. Keep retrying if the screenshot is solid.
    last_w = last_h = 0
    for attempt, delay in enumerate([0.30, 0.45, 0.80, 1.25, 2.00]):
        _time.sleep(delay)
        try: target.update_idletasks(); target.update()
        except Exception: pass

        # Verify foreground
        if sys.platform.startswith("win"):
            try:
                import ctypes
                fg = ctypes.windll.user32.GetForegroundWindow()
                our = _hwnd_of(target)
                foreground_ok = bool(our) and (fg == our)
                if not foreground_ok and attempt < 3:
                    _bring_to_front_win32(our)
                    continue
            except Exception:
                pass

        try:
            target.update_idletasks()
            x = int(target.winfo_rootx())
            y = int(target.winfo_rooty())
            w = int(target.winfo_width())
            h = int(target.winfo_height())
        except Exception:
            x = y = 0; w = h = 200

        # Move off negative coords so mss crop succeeds
        if x < 0 or y < 0:
            try:
                target.geometry(f"+{max(0, x)}+{max(0, y)}")
                target.update_idletasks(); target.update()
                _time.sleep(0.15)
                x = max(0, int(target.winfo_rootx()))
                y = max(0, int(target.winfo_rooty()))
            except Exception:
                pass

        sw, sh = int(target.winfo_screenwidth()), int(target.winfo_screenheight())
        x = max(0, x); y = max(0, y)
        w = max(1, min(w, sw - x)); h = max(1, min(h, sh - y))

        with mss.mss() as sct:
            shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            mss.tools.to_png(shot.rgb, shot.size, output=str(png_path))

        if _image_has_content(png_path):
            last_w, last_h = w, h
            break
        # Else: solid color, try again with longer delay
    else:
        last_w, last_h = w, h  # give up but accept the last attempt

    widgets: list = []
    _walk(target, widgets)

    meta = {
        "screen_name": screen_name,
        "image_width": last_w,
        "image_height": last_h,
        "scaling_factor": scaling,
        "origin_x": x,
        "origin_y": y,
        "foreground_ok": foreground_ok,
        "widgets": widgets,
    }
    _dep_sentinel = sys.modules.get("__auto_labeler_dep_stubs__")
    if _dep_sentinel is not None:
        meta["capture_mode"] = "layout_surrogate_deps_stubbed"
        meta["stubbed_modules"] = sorted(getattr(_dep_sentinel, "stubbed_modules", []))
    (out_dir / "widgets.json").write_text(json.dumps(meta, indent=2))
    return meta


def _hwnd_of(widget):
    """Best-effort retrieval of the Windows HWND for a Tk widget's frame."""
    try:
        frame_id = widget.winfo_toplevel().wm_frame()
        return int(frame_id, 16) if isinstance(frame_id, str) else int(frame_id)
    except Exception:
        try:
            return int(widget.winfo_id())
        except Exception:
            return 0


def main():
    out_dir = Path(r"C:\\GITHUB Repos\\RoboSim_Navigation_Trainer\\Docs3\\work\\mlp_trainer")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import time
        root = tk.Tk()
        try:
            root.tk.call("tk", "scaling")
        except Exception:
            pass



        from train_mlp import TrainerApp
        app = TrainerApp(); app.root.update_idletasks(); app.root.update()

        # If the app constructed its own tk.Tk() (a very common pattern:
        # `class App: def __init__(self): self.root = tk.Tk()`), our
        # pre-created root is empty while the real UI lives under a
        # different Tk instance. tk._default_root always points at the
        # FIRST Tk() (i.e. our empty one), so we can't use that. Instead,
        # scan live Python objects for any other Tk instance with children.
        try:
            if not root.winfo_children():
                import gc as _gc
                replacement = None
                for obj in _gc.get_objects():
                    try:
                        if isinstance(obj, tk.Tk) and obj is not root:
                            if obj.winfo_children():
                                replacement = obj
                                break
                    except Exception:
                        continue
                if replacement is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                    root = replacement
        except Exception:
            pass

        root.update_idletasks()
        root.update()

        # Give the event loop a few ticks to settle (tabs, late layout, etc.)
        for _ in range(3):
            root.update_idletasks()
            root.update()

        target = _target_window(root)
        try:
            target.update_idletasks()
            target.update()
        except Exception:
            pass

        # --- Auto-grow window to fit all children --------------------------
        # Apps frequently call root.geometry("WxH") with a size smaller than
        # the sum of their widgets — Tk happily lays widgets out past the
        # window border, and they end up clipped in the screenshot. To make
        # sure every labelable widget actually appears in the image, we
        # measure the children's bounding box and grow the window to match.
        try:
            target.update_idletasks()
            probe = []
            _walk(target, probe)
            tgt_x0 = int(target.winfo_rootx())
            tgt_y0 = int(target.winfo_rooty())
            cur_w = int(target.winfo_width())
            cur_h = int(target.winfo_height())
            laid = [p for p in probe if int(p["w"]) > 1 and int(p["h"]) > 1]
            if laid:
                needed_w = max(int(p["x"]) + int(p["w"]) for p in laid) - tgt_x0 + 20
                needed_h = max(int(p["y"]) + int(p["h"]) for p in laid) - tgt_y0 + 20
                grow_w = max(cur_w, needed_w)
                grow_h = max(cur_h, needed_h)
                if grow_w > cur_w or grow_h > cur_h:
                    try:
                        # Lift any fixed-size constraints the app may have set.
                        target.maxsize(10_000, 10_000)
                    except Exception:
                        pass
                    target.geometry(f"{grow_w}x{grow_h}")
                    for _ in range(3):
                        target.update_idletasks()
                        target.update()
        except Exception:
            pass

        scaling = 1.0
        try:
            scaling = float(root.tk.call("tk", "scaling"))
        except Exception:
            pass

        # --- Capture primary window (patient render + variance verify) ---
        _capture_one(target, out_dir, "mlp_trainer", scaling)

        # --- Capture any additional visible Toplevels (modals, prefs) -----
        # Treat each peer window as its own screen. Write to sibling work
        # directory `<parent>/<screen>__<slug>/` so run_driver can discover
        # them without needing intermediate coordination.
        additional: list[str] = []
        for peer, slug in _enumerate_peer_toplevels(target):
            sub_name = f"mlp_trainer__{slug}"
            sub_dir = out_dir.parent / f"{out_dir.name}__{slug}"
            try:
                _capture_one(peer, sub_dir, sub_name, scaling)
                additional.append(sub_dir.name)
            except Exception:
                import traceback as _tb
                _tb.print_exc()
        if additional:
            (out_dir / "additional_screens.json").write_text(
                json.dumps(additional, indent=2), encoding="utf-8")

        try:
            root.destroy()
        except Exception:
            pass
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
