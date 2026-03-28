"""
Eye-controlled shortcuts + virtual keyboard. Run from project folder so assets resolve.

Requires: `shape_predictor_68_face_landmarks.dat`, optional `.wav` sounds in CWD.
"""
from __future__ import annotations

import cv2
import numpy as np

import eye_tracking as et
from keyboard_ui import (
    KeyboardState,
    advance_scan,
    draw_keyboard,
    on_confirm_key,
)
from shortcuts_ui import (
    CHANGE_POSITION_OPTIONS,
    SHORTCUTS_LIST,
    ShortcutsState,
    advance_shortcuts_dwell,
    draw_mode_selector,
    draw_shortcuts_menu,
)

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore


def play_sound(name: str):
    if winsound is None:
        return
    try:
        winsound.PlaySound(name, winsound.SND_ALIAS)
    except Exception:
        pass


FONT = cv2.FONT_HERSHEY_PLAIN
BLINK_RATIO_CLOSED = 5.0
FRAMES_TO_BLINK = 6
HOME_BLINKS_REQUIRED = 3  # one full close-then-open per count
GAZE_KEYBOARD_THRESHOLD = 0.9
GAZE_DWELL_FRAMES = 20
GAZE_MENU_STEPS = 3  # show 1/3 .. 3/3 while dwelling left/right


def gaze_menu_step(accum: float) -> int:
    """Map dwell accumulator to a single step 1..3 (no per-frame jumps)."""
    if accum <= 0:
        return 1
    seg = GAZE_DWELL_FRAMES / GAZE_MENU_STEPS
    return min(GAZE_MENU_STEPS, int(accum / seg) + 1)


KEYBOARD_CANVAS_W = 1000
KEYBOARD_CANVAS_H = 600
BOARD_H, BOARD_W = 300, 1400


def main():
    cap = cv2.VideoCapture(1)
    keyboard_canvas = np.zeros((KEYBOARD_CANVAS_H, KEYBOARD_CANVAS_W, 3), np.uint8)
    board = np.zeros((BOARD_H, BOARD_W), np.uint8)
    board[:] = 255

    blinking_frames = 0
    frames_to_blink = FRAMES_TO_BLINK

    program_started = False
    show_home_times = 0
    eye_closed = False

    keyboard_selected = "Shortcuts"
    selected_keyboard_menu = True
    keyboard_selection_frames_L = 0.0
    keyboard_selection_frames_R = 0.0

    text = ""
    shortcuts_state = ShortcutsState()
    keyboard_state = KeyboardState()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rows, cols, _ = frame.shape
        keyboard_canvas[:] = (0, 0, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[rows - 50 : rows, 0:cols] = (255, 255, 255)

        faces = et.detector(gray)

        if not program_started:
            # Home page (readable overlay + nicer typography)
            panel_left, panel_top = 20, 40
            panel_right, panel_bottom = cols - 20, 230
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_left, panel_top),
                (panel_right, panel_bottom),
                (0, 0, 0),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
            cv2.rectangle(
                frame,
                (panel_left, panel_top),
                (panel_right, panel_bottom),
                (0, 180, 255),
                2,
            )

            hint_font = cv2.FONT_HERSHEY_SIMPLEX
            hint_scale = 0.78
            hint_thickness = 2
            hint_color = (255, 255, 255)
            shadow_color = (0, 0, 0)

            hint_x = panel_left + 15
            hint_y = panel_top + 55
            line_gap = 26

            title_text = "Eye Blink Keyboard"
            cv2.putText(
                frame,
                title_text,
                (panel_left + 15, panel_top + 25),
                hint_font,
                0.95,
                shadow_color,
                4,
            )
            cv2.putText(
                frame,
                title_text,
                (panel_left + 15, panel_top + 25),
                hint_font,
                0.95,
                (255, 255, 255),
                2,
            )

            hint_line_1 = "To start the program"
            hint_line_2 = "Close your eyes fully, then open."
            hint_line_3 = f"Repeat {HOME_BLINKS_REQUIRED} times (one full blink each)."

            cv2.putText(
                frame,
                hint_line_1,
                (hint_x, hint_y),
                hint_font,
                hint_scale,
                shadow_color,
                hint_thickness + 2,
            )
            cv2.putText(
                frame,
                hint_line_1,
                (hint_x, hint_y),
                hint_font,
                hint_scale,
                hint_color,
                hint_thickness,
            )

            cv2.putText(
                frame,
                hint_line_2,
                (hint_x, hint_y + line_gap),
                hint_font,
                hint_scale,
                shadow_color,
                hint_thickness + 2,
            )
            cv2.putText(
                frame,
                hint_line_2,
                (hint_x, hint_y + line_gap),
                hint_font,
                hint_scale,
                hint_color,
                hint_thickness,
            )

            cv2.putText(
                frame,
                hint_line_3,
                (hint_x, hint_y + 2 * line_gap),
                hint_font,
                hint_scale,
                shadow_color,
                hint_thickness + 2,
            )
            cv2.putText(
                frame,
                hint_line_3,
                (hint_x, hint_y + 2 * line_gap),
                hint_font,
                hint_scale,
                hint_color,
                hint_thickness,
            )

            counter_text = f"{show_home_times}/{HOME_BLINKS_REQUIRED}"
            counter_font = cv2.FONT_HERSHEY_DUPLEX
            counter_scale = 2.6
            counter_thickness = 3
            (tw, _th), _baseline = cv2.getTextSize(counter_text, counter_font, counter_scale, counter_thickness)
            counter_x = (cols - tw) // 2
            counter_y = panel_bottom - 20
            cv2.putText(
                frame,
                counter_text,
                (counter_x, counter_y),
                counter_font,
                counter_scale,
                (0, 0, 0),
                counter_thickness + 2,
            )
            cv2.putText(
                frame,
                counter_text,
                (counter_x, counter_y),
                counter_font,
                counter_scale,
                (0, 255, 0),
                counter_thickness,
            )

            for face in faces:
                landmarks = et.predictor(gray, face)
                left_eye, right_eye = et.eyes_contour_points(landmarks)
                left_eye_ratio = et.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = et.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
                cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
                if blinking_ratio > BLINK_RATIO_CLOSED:
                    blinking_frames += 1
                    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
                    percentage = blinking_frames / frames_to_blink
                    loading_x = int(cols * percentage)
                    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
                    if blinking_frames >= frames_to_blink:
                        eye_closed = True
                else:
                    if eye_closed:
                        show_home_times += 1
                    eye_closed = False
                    blinking_frames = 0
                if show_home_times >= HOME_BLINKS_REQUIRED:
                    program_started = True
                    blinking_frames = 0
                    keyboard_selection_frames_L = 0.0
                    keyboard_selection_frames_R = 0.0
                break
            else:
                pass

            percentage_blinking = blinking_frames / frames_to_blink
            loading_x = int(cols * percentage_blinking)
            track_color = (60, 60, 60)
            fill_color = (0, 200, 0)
            cv2.rectangle(frame, (0, rows - 50), (cols, rows), track_color, -1)
            cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), fill_color, -1)
            cv2.imshow("Frame", frame)
            cv2.imshow("Virtual keyboard", keyboard_canvas)
            cv2.imshow("Board", board)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        for face in faces:
            landmarks = et.predictor(gray, face)
            left_eye, right_eye = et.eyes_contour_points(landmarks)

            left_eye_ratio = et.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = et.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

            if selected_keyboard_menu:
                draw_mode_selector(
                    keyboard_canvas, FONT, KEYBOARD_CANVAS_W, selected=keyboard_selected
                )

            gaze_ratio_left = et.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame.shape)
            gaze_ratio_right = et.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame.shape)
            gaze_ratio = (gaze_ratio_right + gaze_ratio_left) / 2

            if selected_keyboard_menu:
                if gaze_ratio <= GAZE_KEYBOARD_THRESHOLD:
                    keyboard_selected = "Keyboard"
                    keyboard_selection_frames_L = 0.0
                    keyboard_selection_frames_R += 0.25
                    step_r = gaze_menu_step(keyboard_selection_frames_R)
                    cv2.putText(
                        frame,
                        f"Right {step_r}/{GAZE_MENU_STEPS}",
                        (100, 120),
                        FONT,
                        2,
                        0,
                        3,
                    )
                    if keyboard_selection_frames_R >= GAZE_DWELL_FRAMES:
                        selected_keyboard_menu = False
                        play_sound("keyboard.wav")
                        keyboard_selection_frames_R = 0.0
                else:
                    keyboard_selected = "Shortcuts"
                    keyboard_selection_frames_R = 0.0
                    keyboard_selection_frames_L += 0.25
                    step_l = gaze_menu_step(keyboard_selection_frames_L)
                    cv2.putText(
                        frame,
                        f"Left {step_l}/{GAZE_MENU_STEPS}",
                        (100, 120),
                        FONT,
                        2,
                        0,
                        3,
                    )
                    if keyboard_selection_frames_L >= GAZE_DWELL_FRAMES:
                        selected_keyboard_menu = False
                        play_sound("shortcut.wav")
                        keyboard_selection_frames_L = 0.0
            else:
                if blinking_ratio > BLINK_RATIO_CLOSED:
                    blinking_frames += 1
                    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
                    if blinking_frames >= frames_to_blink:
                        eye_closed = True
                else:
                    if eye_closed:
                        if keyboard_selected == "Shortcuts":
                            if shortcuts_state.left_menu_state == "main":
                                active_option = SHORTCUTS_LIST[shortcuts_state.current_index]
                                if active_option == "Change Position":
                                    shortcuts_state.left_menu_state = "sub"
                                    shortcuts_state.current_index = 0
                                    shortcuts_state.frame_count = 0
                                else:
                                    text = ""
                                    board[:] = 255
                                    keyboard_selection_frames_L = 0.0
                                    keyboard_selection_frames_R = 0.0
                                    text += "[" + active_option + "] "
                                    selected_keyboard_menu = True
                            elif shortcuts_state.left_menu_state == "sub":
                                active_option = CHANGE_POSITION_OPTIONS[shortcuts_state.current_index]
                                board[:] = 255
                                keyboard_selection_frames_L = 0.0
                                keyboard_selection_frames_R = 0.0
                                text = ""
                                text += "[Position: " + active_option + "] "
                                selected_keyboard_menu = True
                                shortcuts_state.left_menu_state = "main"
                                shortcuts_state.current_index = 0
                                shortcuts_state.frame_count = 0
                            play_sound("sound.wav")
                        elif keyboard_selected == "Keyboard":
                            if on_confirm_key(keyboard_state):
                                selected_keyboard_menu = True
                                keyboard_state.typed_text = ""
                                keyboard_state.scan_mode = "row"
                                keyboard_state.row_index = 0
                                keyboard_state.col_index = 0
                                text = ""
                                board[:] = 255
                                keyboard_selection_frames_L = 0.0
                                keyboard_selection_frames_R = 0.0
                            else:
                                text = keyboard_state.typed_text
                                board[:] = 255
                            play_sound("sound.wav")
                    eye_closed = False
                    blinking_frames = 0
            break

        if program_started and not selected_keyboard_menu and keyboard_selected == "Shortcuts":
            draw_shortcuts_menu(keyboard_canvas, shortcuts_state)
            advance_shortcuts_dwell(shortcuts_state, eye_closed)

        if program_started and not selected_keyboard_menu and keyboard_selected == "Keyboard":
            draw_keyboard(keyboard_canvas, keyboard_state, origin_x=0, origin_y=0)
            advance_scan(keyboard_state, paused=eye_closed)

        if keyboard_selected == "Keyboard" and program_started and not selected_keyboard_menu:
            cv2.putText(board, keyboard_state.typed_text, (80, 100), FONT, 3, 0, 2)
        else:
            cv2.putText(board, text, (80, 100), FONT, 3, 0, 2)

        percentage_blinking = blinking_frames / frames_to_blink
        loading_x = int(cols * percentage_blinking)
        cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Virtual keyboard", keyboard_canvas)
        cv2.imshow("Board", board)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 and program_started and not selected_keyboard_menu and keyboard_selected == "Keyboard":
            if on_confirm_key(keyboard_state):
                selected_keyboard_menu = True
                keyboard_state.typed_text = ""
                keyboard_state.scan_mode = "row"
                keyboard_state.row_index = 0
                keyboard_state.col_index = 0
                text = ""
                board[:] = 255
                keyboard_selection_frames_L = 0.0
                keyboard_selection_frames_R = 0.0
            else:
                text = keyboard_state.typed_text
                board[:] = 255

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
