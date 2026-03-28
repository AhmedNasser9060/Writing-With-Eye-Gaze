"""
Left shortcuts grid and mode selector (from `2) Left Menu.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2

SHORTCUTS_LIST = [
    "Emergency",
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


# Colors (BGR)
WHITE_BGR = (245, 245, 245)
BLACK_BGR = (30, 30, 30)
GRAY_BGR = (200, 200, 200)
BLUE_BGR = (255, 150, 100)
GREEN_BGR = (150, 255, 100)
RED_BGR = (120, 120, 255)

def draw_mode_selector(canvas, font, cols: int, selected: str = "Shortcuts"):
    """Draw the first-page mode selector (Shortcuts vs Keyboard)."""
    import i18n
    rows, w, _ = canvas.shape
    mid = cols // 2

    # Split background
    left_color = BLUE_BGR if selected == "Shortcuts" else (200, 120, 80)
    right_color = GREEN_BGR if selected == "Keyboard" else (120, 200, 80)

    cv2.rectangle(canvas, (0, 0), (mid, rows), left_color, -1)
    cv2.rectangle(canvas, (mid, 0), (cols, rows), right_color, -1)
    cv2.line(canvas, (mid, 0), (mid, rows), (80, 80, 80), 3)

    # Two halves
    # left_color = (0, 150, 0) if selected == "Shortcuts" else (70, 70, 70)
    # right_color = (0, 150, 0) if selected == "Keyboard" else (70, 70, 70)

    # Labels (centered and not oversized)
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 2.2
    label_th = 3

    def put_center(text: str, x0: int, x1: int):
        translated = i18n.tr(text)
        if i18n.get_language() == "ar":
            # For exact vertical centering we use PIL bbox but roughly here is fine via i18n
            center_pt = ((x0 + x1) // 2, rows // 2)
            i18n.put_text(canvas, translated, center_pt, label_scale, BLACK_BGR, label_th, center=True)
        else:
            (tw, th), _ = cv2.getTextSize(translated, label_font, label_scale, label_th)
            x = (x0 + x1 - tw) // 2
            y = rows // 2 + th // 2
            cv2.putText(canvas, translated, (x, y), label_font, label_scale, BLACK_BGR, label_th)

    put_center("Shortcuts", 0, mid)
    put_center("Keyboard", mid, cols)


def draw_shortcut_cell(
    keyboard,
    index: int,
    text: str,
    state: ShortcutsState,
    cols: int = 3,
):
    """Draw one shortcut box; returns True if dwell completed (frame_count >= 20)."""
    import i18n
    
    is_main_menu = state.left_menu_state == "main"
    options = SHORTCUTS_LIST if is_main_menu else CHANGE_POSITION_OPTIONS

    current_index = state.current_index
    frame_count = state.frame_count

    rows, cols, _ = keyboard.shape
    base_box_w = 340 # slightly wider buttons for 1080p
    base_box_h = 110 # slightly taller
    gap = 25
    
    total_w = 3 * base_box_w + 2 * gap
    start_x = (cols - total_w) // 2
    start_y = 350 # Offset nicely below camera

    if is_main_menu and index == 0:
        box_w = total_w
        box_h = base_box_h
        x = start_x
        y = start_y
    else:
        if is_main_menu:
            logical_idx = index - 1
            row = logical_idx // 3 + 1
            col = logical_idx % 3
        else:
            row = index // 3
            col = index % 3
        
        box_w = base_box_w
        box_h = base_box_h
        x = start_x + col * (box_w + gap)
        y = start_y + row * (box_h + gap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    th = 2

    # Selected / not selected style
    if index == current_index:
        fill = GREEN_BGR
        text_color = BLACK_BGR
        # Progress for dwell (0..3)
        progress = min(3, (frame_count * 3) // 20)
        # Using pure cv2 here because the numbers are standard english
        cv2.putText(
            keyboard,
            f"{progress}/3",
            (x + 12, y + 30),
            font,
            0.7,
            BLACK_BGR,
            2,
        )
        if frame_count >= 20:
            return True
    else:
        if is_main_menu and index == 0:
            fill = RED_BGR
        else:
            fill = GRAY_BGR
        text_color = BLACK_BGR

    i18n.draw_rounded_rect(keyboard, (x, y), (x + box_w, y + box_h), fill, -1, r=15)
    i18n.draw_rounded_rect(keyboard, (x, y), (x + box_w, y + box_h), BLACK_BGR, 2, r=15)

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
    if is_main_menu and index == 0:
        # Increase scale slightly for the large button
        scale = min(1.5, scale + 0.3)
    
    import i18n
    # Center text manually using i18n
    b_color = (int(text_color[0]), int(text_color[1]), int(text_color[2]))
    # For PIL logic to center it properly within the box bounds, 
    # we just pass the center point and rely on PIL inside i18n.put_text
    center_pt = (x + box_w // 2, y + box_h // 2)
    
    if i18n.get_language() == "ar":
        # Let i18n PIL handle centering completely
        i18n.put_text(keyboard, i18n.tr(text), center_pt, scale, b_color, th, center=True)
    else:
        text_y = y + (box_h + size[1]) // 2
        cv2.putText(keyboard, i18n.tr(text), (text_x, text_y), font, scale, b_color, th)
    return False


def draw_shortcuts_menu(canvas, state: ShortcutsState):
    if state.left_menu_state == "main":
        menu = SHORTCUTS_LIST
    else:
        menu = CHANGE_POSITION_OPTIONS
    for i, option in enumerate(menu):
        draw_shortcut_cell(canvas, i, option, state)


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
