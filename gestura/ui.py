import cv2


def draw_ui(frame, hand_data, active, pred_letter, pred_conf, stable, show_help):
    h, w = frame.shape[:2]

    # Contour + bbox
    if hand_data:
        cv2.drawContours(frame, [hand_data["contour"]], -1, (0, 255, 0), 2)
        x0, y0, x1, y1 = hand_data["bbox"]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # Status
    status_color = (0, 255, 0) if active else (0, 0, 255)
    status_text = "ACTIVE" if active else "INACTIVE (Press 'A')"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Hand state
    cv2.putText(frame, f"Hand: {'DETECTED' if hand_data else 'NOT DETECTED'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Prediction
    if pred_letter is not None:
        cv2.putText(frame, f"Prediction: {pred_letter} ({pred_conf:.2f})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Stability indicator
    if stable and active:
        cv2.putText(frame, "READY TO TYPE", (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    elif hand_data:
        cv2.putText(frame, "HOLD STEADY", (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

    # Help
    if show_help:
        help_lines = [
            "Controls:",
            "A - Activate/Deactivate typing",
            "H - Toggle help overlay",
            "M - Toggle mask window",
            "R - Reset stability",
            "Q - Quit"
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (w - 310, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return frame
