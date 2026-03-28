"""
Eye-controlled shortcuts + virtual keyboard. Run from project folder so assets resolve.

Requires: `shape_predictor_68_face_landmarks.dat`, optional `.wav` sounds in CWD.
"""
from __future__ import annotations

import cv2
import numpy as np
import i18n

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


APP_W = 1920
APP_H = 1080

def main():
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Eye Control System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Control System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Eye Control System", APP_W, APP_H)
    master_screen = np.full((APP_H, APP_W, 3), (245, 245, 245), dtype=np.uint8)

    blinking_frames = 0
    frames_to_blink = FRAMES_TO_BLINK

    program_started = False
    language_selected = False
    language_selection_frames_L = 0.0
    language_selection_frames_R = 0.0
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

        master_screen[:] = (245, 245, 245)
        rows, cols, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- DRAW BACKGROUND PANELS FIRST ---
        if program_started and selected_keyboard_menu:
            draw_mode_selector(master_screen, FONT, APP_W, selected=keyboard_selected)

        # --- NOW DRAW THE CAMERA OVERLAY ---
        # Position the camera feed at top-right
        # Make the camera a smaller fixed Picture-in-Picture
        cam_w = 360
        cam_h = int(cam_w * (rows / cols))
        frame_resized = cv2.resize(frame, (cam_w, cam_h))
        # Draw camera frame in master screen
        master_screen[20:20+cam_h, APP_W-cam_w-20:APP_W-20] = frame_resized
        i18n.draw_rounded_rect(master_screen, (APP_W-cam_w-22, 18), (APP_W-18, 20+cam_h+2), (200, 200, 200), 4, r=10)

        faces = et.detector(gray)

        if not program_started:
            cx = APP_W // 2
            
            title_text = "Eye Blink Keyboard | لوحة مفاتيح العين"
            i18n.put_text(master_screen, title_text, (cx, 150), 2.0, (30,30,30), 4, center=True)
            
            hint_line_1 = "To start the program | لبدء البرنامج"
            hint_line_2 = "Close your eyes fully, then open. | أغلق عينيك بالكامل، ثم افتحهما."
            hint_line_3 = f"Repeat {HOME_BLINKS_REQUIRED} times | كرر {HOME_BLINKS_REQUIRED} مرات"

            i18n.put_text(master_screen, hint_line_1, (cx, 300), 1.2, (30,30,30), 2, center=True)
            i18n.put_text(master_screen, hint_line_2, (cx, 380), 1.2, (30,30,30), 2, center=True)
            i18n.put_text(master_screen, hint_line_3, (cx, 460), 1.2, (30,30,30), 2, center=True)

            counter_text = f"{show_home_times}/{HOME_BLINKS_REQUIRED}"
            i18n.put_text(master_screen, counter_text, (cx, 600), 2.5, (150, 255, 100), 4, center=True)

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
                    cv2.polylines(master_screen, [(left_eye * np.array([cam_w/cols, cam_h/rows])).astype(int) + np.array([APP_W-cam_w-20, 20])], True, (150, 255, 100), 2)
                    cv2.polylines(master_screen, [(right_eye * np.array([cam_w/cols, cam_h/rows])).astype(int) + np.array([APP_W-cam_w-20, 20])], True, (150, 255, 100), 2)
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
            # Progress bar for parsing
            bar_w = 400
            bar_x = cx - bar_w // 2
            bar_y = 680
            bar_h = 30
            i18n.draw_rounded_rect(master_screen, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1, r=15)
            if percentage_blinking > 0:
                fill_w = int(bar_w * percentage_blinking)
                if fill_w < 30: fill_w = 30
                fill_x1 = bar_x + (bar_w - fill_w) // 2
                fill_x2 = bar_x + (bar_w + fill_w) // 2
                i18n.draw_rounded_rect(master_screen, (fill_x1, bar_y), (fill_x2, bar_y + bar_h), (150, 255, 100), -1, r=15)
            
            cv2.imshow("Eye Control System", master_screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        elif not language_selected:
            cx = APP_W // 2
            title = "Choose Language | اختر اللغة"
            hint_en = "Look Right for English"
            hint_ar = "انظر لليسار للغة العربية"
            
            i18n.put_text(master_screen, title, (cx, 150), 2.0, (30,30,30), 4, center=True)

            # Left side (Arabic)
            i18n.draw_rounded_rect(master_screen, (100, 300), (cx - 50, 600), (255, 150, 100), -1, r=30)
            i18n.put_text(master_screen, hint_ar, (100 + (cx-150)//2, 450), 1.5, (30,30,30), 3, center=True)
            
            # Right side (English)
            i18n.draw_rounded_rect(master_screen, (cx + 50, 300), (APP_W - 100, 600), (150, 255, 100), -1, r=30)
            i18n.put_text(master_screen, hint_en, (cx + 50 + (cx-150)//2, 450), 1.5, (30,30,30), 3, center=True)

            for face in faces:
                landmarks = et.predictor(gray, face)
                left_eye, right_eye = et.eyes_contour_points(landmarks)
                gaze_ratio_left = et.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame.shape)
                gaze_ratio_right = et.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame.shape)
                gaze_ratio = (gaze_ratio_right + gaze_ratio_left) / 2
                
                if gaze_ratio <= GAZE_KEYBOARD_THRESHOLD:
                    language_selection_frames_R += 0.25
                    language_selection_frames_L = 0.0
                    step = gaze_menu_step(language_selection_frames_R)
                    i18n.put_text(master_screen, i18n.tr("Right %d/%d") % (step, GAZE_MENU_STEPS), (cx + 50 + (cx-150)//2, 530), 1.5, (30, 30, 30), 3, center=True)
                    if language_selection_frames_R >= GAZE_DWELL_FRAMES:
                        i18n.set_language("en")
                        language_selected = True
                        play_sound("sound.wav")
                else:
                    language_selection_frames_L += 0.25
                    language_selection_frames_R = 0.0
                    step = gaze_menu_step(language_selection_frames_L)
                    i18n.put_text(master_screen, i18n.tr("Left %d/%d") % (step, GAZE_MENU_STEPS), (100 + (cx-150)//2, 530), 1.5, (30, 30, 30), 3, center=True)
                    if language_selection_frames_L >= GAZE_DWELL_FRAMES:
                        i18n.set_language("ar")
                        language_selected = True
                        play_sound("sound.wav")
                break
            cv2.imshow("Eye Control System", master_screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            continue

        for face in faces:
            landmarks = et.predictor(gray, face)
            left_eye, right_eye = et.eyes_contour_points(landmarks)

            left_eye_ratio = et.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = et.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            cv2.polylines(master_screen, [left_eye * np.array([cam_w/cols, cam_h/rows]).astype(int) + np.array([APP_W-cam_w-20, 20])], True, (0, 0, 255), 2)
            cv2.polylines(master_screen, [right_eye * np.array([cam_w/cols, cam_h/rows]).astype(int) + np.array([APP_W-cam_w-20, 20])], True, (0, 0, 255), 2)

            # (Mode selector background is already drawn before camera)

            gaze_ratio_left = et.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame.shape)
            gaze_ratio_right = et.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame.shape)
            gaze_ratio = (gaze_ratio_right + gaze_ratio_left) / 2

            if selected_keyboard_menu:
                if gaze_ratio <= GAZE_KEYBOARD_THRESHOLD:
                    keyboard_selected = "Keyboard"
                    keyboard_selection_frames_L = 0.0
                    keyboard_selection_frames_R += 0.25
                    step_r = gaze_menu_step(keyboard_selection_frames_R)
                    i18n.put_text(master_screen, i18n.tr("Right %d/%d") % (step_r, GAZE_MENU_STEPS), (APP_W//2 + 200, 500), 2.0, (30,30,30), 3, center=True)
                    if keyboard_selection_frames_R >= GAZE_DWELL_FRAMES:
                        selected_keyboard_menu = False
                        play_sound("keyboard.wav")
                        keyboard_selection_frames_R = 0.0
                else:
                    keyboard_selected = "Shortcuts"
                    keyboard_selection_frames_R = 0.0
                    keyboard_selection_frames_L += 0.25
                    step_l = gaze_menu_step(keyboard_selection_frames_L)
                    i18n.put_text(master_screen, i18n.tr("Left %d/%d") % (step_l, GAZE_MENU_STEPS), (APP_W//2 - 200, 500), 2.0, (30,30,30), 3, center=True)
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
                                    keyboard_selection_frames_L = 0.0
                                    keyboard_selection_frames_R = 0.0
                                    text += "[" + i18n.tr(active_option) + "] "
                                    selected_keyboard_menu = True
                            elif shortcuts_state.left_menu_state == "sub":
                                active_option = CHANGE_POSITION_OPTIONS[shortcuts_state.current_index]
                                keyboard_selection_frames_L = 0.0
                                keyboard_selection_frames_R = 0.0
                                text = ""
                                text += "[" + i18n.tr("Position: ") + i18n.tr(active_option) + "] "
                                selected_keyboard_menu = True
                                shortcuts_state.left_menu_state = "main"
                                shortcuts_state.current_index = 0
                                shortcuts_state.frame_count = 0
                            
                            if active_option == "Emergency" or active_option == i18n.tr("Emergency"):
                                play_sound("alert.wav")
                            else:
                                play_sound("sound.wav")
                        elif keyboard_selected == "Keyboard":
                            if on_confirm_key(keyboard_state):
                                selected_keyboard_menu = True
                                keyboard_state.typed_text = ""
                                keyboard_state.scan_mode = "row"
                                keyboard_state.row_index = 0
                                keyboard_state.col_index = 0
                                text = ""
                                keyboard_selection_frames_L = 0.0
                                keyboard_selection_frames_R = 0.0
                            else:
                                text = keyboard_state.typed_text
                            play_sound("sound.wav")
                    eye_closed = False
                    blinking_frames = 0
            break

        is_blinking = (blinking_frames > 0) or eye_closed

        if program_started and not selected_keyboard_menu and keyboard_selected == "Shortcuts":
            draw_shortcuts_menu(master_screen, shortcuts_state)
            advance_shortcuts_dwell(shortcuts_state, is_blinking)

        if program_started and not selected_keyboard_menu and keyboard_selected == "Keyboard":
            # Total keyboard width ~700. Center it horizontally. Set Y to 350.
            draw_keyboard(master_screen, keyboard_state, origin_x=(APP_W - 700)//2, origin_y=350)
            advance_scan(keyboard_state, paused=is_blinking)

        if program_started:
            # Shared typed text zone at the bottom (Persistent on all screens)
            text_str = keyboard_state.typed_text if keyboard_selected == "Keyboard" else i18n.tr(text)
            
            # Text box
            txt_y = APP_H - 100
            i18n.draw_rounded_rect(master_screen, (50, txt_y), (APP_W - 50, APP_H - 20), (200, 200, 200), -1, r=15)
            i18n.draw_rounded_rect(master_screen, (50, txt_y), (APP_W - 50, APP_H - 20), (30, 30, 30), 2, r=15)
            
            i18n.put_text(master_screen, text_str, (APP_W//2, txt_y + 40), 1.5, (30,30,30), 3, center=True)

        # Draw scanning blink progress indicator globally at center top if active
        percentage_blinking = blinking_frames / frames_to_blink
        if program_started and percentage_blinking > 0:
            bar_w = 400
            bar_x = (APP_W - bar_w) // 2
            bar_y = 50
            bar_h = 20
            i18n.draw_rounded_rect(master_screen, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1, r=10)
            if percentage_blinking > 0:
                fill_w = int(bar_w * percentage_blinking)
                if fill_w < 20: fill_w = 20
                fill_x1 = bar_x + (bar_w - fill_w) // 2
                fill_x2 = bar_x + (bar_w + fill_w) // 2
                i18n.draw_rounded_rect(master_screen, (fill_x1, bar_y), (fill_x2, bar_y + bar_h), (150, 255, 100), -1, r=10)

        cv2.imshow("Eye Control System", master_screen)
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
                keyboard_selection_frames_L = 0.0
                keyboard_selection_frames_R = 0.0
            else:
                text = keyboard_state.typed_text

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
