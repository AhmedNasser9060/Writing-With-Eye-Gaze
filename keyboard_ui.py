"""
On-screen keyboard (row/column scan) - logic from `keyboard.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

KEYBOARD_LAYOUT = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPACE", "ENTER", "DEL", "BACK"],
]

KEY_W = 130
KEY_H = 80
KEY_GAP = 20
START_X = 60
START_Y = 20


@dataclass
class KeyboardState:
    scan_mode: str = "row"  # "row" | "col"
    row_index: int = 0
    col_index: int = 0
    typed_text: str = ""
    scan_delay: float = 1.0
    last_switch: float = field(default_factory=time.time)


def draw_key(img, x, y, w, h, text, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def draw_keyboard(
    canvas: np.ndarray,
    state: KeyboardState,
    origin_x: int = 0,
    origin_y: int = 0,
    typed_line_y: int | None = None,
):
    """Draw keys on `canvas` with top-left at (origin_x, origin_y)."""
    h, w = canvas.shape[:2]
    if typed_line_y is None:
        typed_line_y = h - 50
    start_y = origin_y + START_Y
    for i, row in enumerate(KEYBOARD_LAYOUT):
        x = origin_x + START_X
        for j, key in enumerate(row):
            color = (200, 200, 200)
            if state.scan_mode == "row" and state.row_index == i:
                color = (255, 200, 100)
            if state.scan_mode == "col" and state.row_index == i and state.col_index == j:
                color = (0, 0, 255)
            draw_key(canvas, x, start_y, KEY_W, KEY_H, key, color)
            x += KEY_W + KEY_GAP
        start_y += KEY_H + KEY_GAP
    cv2.putText(
        canvas,
        state.typed_text[-40:],
        (origin_x + 150, min(typed_line_y, h - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )


def advance_scan(state: KeyboardState, paused: bool = False) -> None:
    """Move highlight by time.

    If `paused` is True (e.g. user is blinking), we freeze the highlight.
    """
    if paused:
        # Avoid jump when unpausing by resetting the timer.
        state.last_switch = time.time()
        return

    now = time.time()
    if now - state.last_switch <= state.scan_delay:
        return
    state.last_switch = now
    if state.scan_mode == "row":
        state.row_index += 1
        if state.row_index >= len(KEYBOARD_LAYOUT):
            state.row_index = 0
    else:
        state.col_index += 1
        if state.col_index >= len(KEYBOARD_LAYOUT[state.row_index]):
            state.col_index = 0


def on_confirm_key(state: KeyboardState) -> bool:
    """Row->col, then type key. Returns True if BACK was chosen (caller: home + reset)."""
    if state.scan_mode == "row":
        state.scan_mode = "col"
        state.col_index = 0
        return False
    k = KEYBOARD_LAYOUT[state.row_index][state.col_index]
    if k == "SPACE":
        state.typed_text += " "
    elif k == "ENTER":
        state.typed_text += "\n"
    elif k == "DEL":
        state.typed_text = state.typed_text[:-1]
    elif k == "BACK":
        return True
    else:
        state.typed_text += k
    state.scan_mode = "row"
    return False
