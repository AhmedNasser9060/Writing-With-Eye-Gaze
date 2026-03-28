"""
On-screen keyboard (row/column scan) - logic from `keyboard.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

ENGLISH_LAYOUT = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPACE", "ENTER", "DEL", "BACK"],
]

ARABIC_LAYOUT = [
    ["أ", "ب", "ت", "ث", "ج", "ح", "خ"],
    ["د", "ذ", "ر", "ز", "س", "ش", "ص"],
    ["ض", "ط", "ظ", "ع", "غ", "ف", "ق"],
    ["ك", "ل", "م", "ن", "ه", "و", "ي"],
    ["SPACE", "ENTER", "DEL", "BACK"]
]

KEY_H = 80
START_X = 60
START_Y = 20

import i18n

WHITE_BGR = (245, 245, 245)
BLACK_BGR = (30, 30, 30)
GRAY_BGR = (200, 200, 200)
BLUE_BGR = (255, 150, 100)
GREEN_BGR = (150, 255, 100)
RED_BGR = (120, 120, 255)

def get_current_layout():
    return ARABIC_LAYOUT if i18n.get_language() == "ar" else ENGLISH_LAYOUT

def get_layout_params():
    return (115, 15) if i18n.get_language() == "ar" else (130, 20)


@dataclass
class KeyboardState:
    scan_mode: str = "row"  # "row" | "col"
    row_index: int = 0
    col_index: int = 0
    typed_text: str = ""
    scan_delay: float = 1.0
    last_switch: float = field(default_factory=time.time)


def draw_key(img, x, y, w, h, text, color):
    i18n.draw_rounded_rect(img, (x, y), (x + w, y + h), color, -1, r=15)
    i18n.draw_rounded_rect(img, (x, y), (x + w, y + h), BLACK_BGR, 2, r=15)
    i18n.put_text(img, i18n.tr(text), (x + w//2, y + h//2), 0.8, BLACK_BGR, 2, center=True)


def draw_keyboard(
    canvas: np.ndarray,
    state: KeyboardState,
    origin_x: int = 0,
    origin_y: int = 0,
    typed_line_y: int | None = None,
):
    """Draw keys on `canvas` with top-left at (origin_x, origin_y)."""
    layout = get_current_layout()
    key_w, key_gap = get_layout_params()
    h, w = canvas.shape[:2]
    if typed_line_y is None:
        typed_line_y = h - 50
    start_y = origin_y + START_Y
    for i, row in enumerate(layout):
        x = origin_x + START_X
        for j, key in enumerate(row):
            color = GRAY_BGR
            if state.scan_mode == "row" and state.row_index == i:
                color = BLUE_BGR
            if state.scan_mode == "col" and state.row_index == i and state.col_index == j:
                color = RED_BGR
            draw_key(canvas, x, start_y, key_w, KEY_H, key, color)
            x += key_w + key_gap
        start_y += KEY_H + key_gap
    # Typed text is handled by main.py in unified screen layout


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
    
    layout = get_current_layout()
    if state.scan_mode == "row":
        state.row_index += 1
        if state.row_index >= len(layout):
            state.row_index = 0
    else:
        state.col_index += 1
        if state.row_index < len(layout) and state.col_index >= len(layout[state.row_index]):
            state.col_index = 0


def on_confirm_key(state: KeyboardState) -> bool:
    """Row->col, then type key. Returns True if BACK was chosen (caller: home + reset)."""
    layout = get_current_layout()
    if state.scan_mode == "row":
        state.scan_mode = "col"
        state.col_index = 0
        return False
        
    if state.row_index < len(layout) and state.col_index < len(layout[state.row_index]):
        k = layout[state.row_index][state.col_index]
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
