"""
Left shortcuts grid and mode selector (from `2) Left Menu.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2

SHORTCUTS_LIST = [
    "Pain",
    "Airway Obstruction",
    "Suction",
    "Care",
    "Water",
    "Food",
    "Dizziness",
    "Can't Sleep",
    "Change Position",
]

CHANGE_POSITION_OPTIONS = ["Right", "Left", "Back"]

MenuState = Literal["main", "sub"]


@dataclass
class ShortcutsState:
    current_index: int = 0
    frame_count: int = 0
    left_menu_state: MenuState = "main"


def draw_mode_selector(canvas, font, cols: int, selected: str = "Shortcuts"):
    """Draw the first-page mode selector (Shortcuts vs Keyboard)."""
    rows, w, _ = canvas.shape
    mid = cols // 2

    # Background panel
    cv2.rectangle(canvas, (0, 0), (cols, rows), (15, 15, 15), -1)
    cv2.rectangle(canvas, (0, 0), (cols, rows), (40, 40, 40), 3)

    # Two halves
    left_color = (0, 150, 0) if selected == "Shortcuts" else (70, 70, 70)
    right_color = (0, 150, 0) if selected == "Keyboard" else (70, 70, 70)

    cv2.rectangle(canvas, (0, 0), (mid, rows), (left_color[0], left_color[1], left_color[2]), 3)
    cv2.rectangle(canvas, (mid, 0), (cols, rows), (right_color[0], right_color[1], right_color[2]), 3)
    cv2.line(canvas, (mid, 0), (mid, rows), (80, 80, 80), 2)

    # Labels (centered and not oversized)
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 2.2
    label_th = 3

    def put_center(text: str, x0: int, x1: int):
        (tw, th), _ = cv2.getTextSize(text, label_font, label_scale, label_th)
        x = (x0 + x1 - tw) // 2
        y = rows // 2 + th // 2
        cv2.putText(canvas, text, (x, y), label_font, label_scale, (255, 255, 255), label_th)

    put_center("Shortcuts", 0, mid)
    put_center("Keyboard", mid, cols)


def draw_shortcut_cell(
    keyboard,
    index: int,
    text: str,
    current_index: int,
    frame_count: int,
    cols: int = 3,
):
    """Draw one shortcut box; returns True if dwell completed (frame_count >= 20)."""
    box_w, box_h = 250, 100
    row = index // cols
    col = index % cols
    x = 50 + col * (box_w + 20)
    y = 50 + row * (box_h + 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    th = 2

    # Selected / not selected style
    if index == current_index:
        fill = (230, 230, 230)
        text_color = (0, 0, 0)
        border = (0, 220, 0)
        # Progress for dwell (0..3)
        progress = min(3, (frame_count * 3) // 20)
        cv2.putText(
            keyboard,
            f"{progress}/3",
            (x + 12, y + 30),
            font,
            0.7,
            (0, 160, 0),
            2,
        )
        if frame_count >= 20:
            return True
    else:
        fill = (45, 45, 45)
        text_color = (255, 255, 255)
        border = (80, 80, 80)

    cv2.rectangle(keyboard, (x, y), (x + box_w, y + box_h), fill, -1)
    cv2.rectangle(keyboard, (x, y), (x + box_w, y + box_h), border, 2)

    # Fit text horizontally inside the box (reduce scale if needed)
    scale = 0.95
    margin_x = 16
    while scale > 0.3:
        size = cv2.getTextSize(text, font, scale, th)[0]
        if size[0] <= (box_w - 2 * margin_x) and size[1] <= (box_h - 20):
            break
        scale -= 0.05

    size = cv2.getTextSize(text, font, scale, th)[0]
    text_x = x + (box_w - size[0]) // 2
    text_y = y + (box_h + size[1]) // 2
    cv2.putText(keyboard, text, (text_x, text_y), font, scale, text_color, th)
    return False


def draw_shortcuts_menu(canvas, state: ShortcutsState):
    if state.left_menu_state == "main":
        menu = SHORTCUTS_LIST
    else:
        menu = CHANGE_POSITION_OPTIONS
    for i, option in enumerate(menu):
        draw_shortcut_cell(canvas, i, option, state.current_index, state.frame_count)


def advance_shortcuts_dwell(state: ShortcutsState, eye_closed: bool):
    """Increment frame counter and advance selection when dwell completes."""
    if eye_closed:
        return
    state.frame_count += 1
    if state.left_menu_state == "main":
        menu_length = len(SHORTCUTS_LIST)
    else:
        menu_length = len(CHANGE_POSITION_OPTIONS)
    if state.frame_count >= 20:
        state.current_index = (state.current_index + 1) % menu_length
        state.frame_count = 0
