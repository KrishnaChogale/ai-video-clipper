"""Lightweight face-guided portrait framing for webcam/reaction clips."""
import cv2


def face_crop_x_expression(video_path: str, start_sec: float, end_sec: float,
                           sample_sec: float = 0.75) -> str | None:
    """Return an FFmpeg crop-x expression that follows the largest visible face."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    points = []
    t = float(start_sec)
    try:
        while t <= float(end_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.12,
                                                 minNeighbors=5, minSize=(36, 36))
                if len(faces):
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    points.append((round(t - start_sec, 2), round(x + w / 2, 1)))
            t += sample_sec
    finally:
        cap.release()
    if not points:
        return None

    # Interpolate between detections for smooth, bounded movement.
    fallback = "(iw-ow)/2"
    expression = f"max(0,min(iw-ow,{points[-1][1]}-ow/2))"
    for idx in range(len(points) - 2, -1, -1):
        t0, x0 = points[idx]
        t1, x1 = points[idx + 1]
        left = f"max(0,min(iw-ow,{x0}+({x1}-{x0})*(t-{t0})/{max(0.01, t1-t0)}-ow/2))"
        expression = f"if(lt(t,{t1}),{left},{expression})"
    if points[0][0] > 0:
        expression = f"if(lt(t,{points[0][0]}),{fallback},{expression})"
    return expression.replace(",", r"\,")
