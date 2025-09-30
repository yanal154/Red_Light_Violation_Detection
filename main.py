import os
import sys
import cv2
import time
import numpy as np
import subprocess
from ultralytics import YOLO

VIDEO_IN  = "video_path"
VIDEO_OUT = "output_violations.mp4"
VEHICLE_MODEL_WEIGHTS = "yolo12l.pt"
TRACKER_CFG  = "bytetrack.yaml"

CONF_VEHICLE = 0.35
IOU_VEHICLE  = 0.45
TRAFFIC_LIGHT_REFRESH = 5

HSV_RED1_LO,   HSV_RED1_HI   = (0, 80, 80),   (10, 255, 255)
HSV_RED2_LO,   HSV_RED2_HI   = (170, 80, 80), (180, 255, 255)
HSV_YELLOW_LO, HSV_YELLOW_HI = (20, 90, 100), (32, 255, 255)
HSV_GREEN_LO,  HSV_GREEN_HI  = (40, 80, 80),  (85, 255, 255)
MIN_PIX_COUNT = 50

VEH_OK = {"car", "bus", "truck", "motorcycle"}

line_pts = []
roi_pts_list = []
drawing_mode = None

violated_ids = set()
violations_count = 0
prev_center = {}

COLOR_LINE = (0, 0, 255)  # changed from white to red
COLOR_TEXT = (230, 230, 230)
COLOR_OK   = (0, 255, 0)
COLOR_BAD  = (0, 0, 255)
COLOR_ROI  = (150, 255, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def put_text(img, txt, org, color=(255,255,255), scale=0.7, thick=2):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)

def draw_setup_hud(disp):
    put_text(disp, "Press L then click 2 points to draw LINE", (18, 28), (200,200,200), 0.7, 2)
    put_text(disp, "Press R then click 2 points to add TL ROI (multi allowed)", (18, 56), (200,200,200), 0.7, 2)
    put_text(disp, "U: undo last ROI/LINE   C: clear ROIs   SPACE: start   Q/Esc: quit", (18, 84), (200,200,200), 0.7, 2)
    if len(line_pts) == 1:
        cv2.circle(disp, line_pts[0], 4, COLOR_LINE, -1)
    if len(line_pts) == 2:
        cv2.line(disp, line_pts[0], line_pts[1], COLOR_LINE, 2)
        put_text(disp, "Line OK", (line_pts[0][0]+6, line_pts[0][1]+18), COLOR_OK, 0.7, 2)
    for idx, roi in enumerate(roi_pts_list):
        if len(roi) == 1:
            cv2.circle(disp, roi[0], 4, COLOR_ROI, -1)
        elif len(roi) == 2:
            (x1,y1),(x2,y2) = roi
            cv2.rectangle(disp, (min(x1,x2),min(y1,y2)), (max(x1,x2),max(y1,y2)), COLOR_ROI, 2)
            put_text(disp, f"ROI #{idx+1}", (min(x1,x2)+6, min(y1,y2)+20), COLOR_OK, 0.6, 2)

def setup_mouse_cb(event, x, y, flags, param):
    global drawing_mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    H, W = param['img_shape']
    if not (0 <= x < W and 0 <= y < H):
        return
    if drawing_mode == 'line':
        if len(line_pts) < 2:
            line_pts.append((x, y))
        if len(line_pts) == 2:
            drawing_mode = None
    elif drawing_mode == 'roi':
        if not roi_pts_list or len(roi_pts_list[-1]) == 2:
            roi_pts_list.append([])
        roi_pts_list[-1].append((x, y))
        if len(roi_pts_list[-1]) == 2:
            drawing_mode = None
    else:
        pass

def hsv_major_color(bgr):
    if bgr is None or bgr.size == 0:
        return "unknown"
    h, w = bgr.shape[:2]
    if h*w > 160*160:
        scale = min(1.0, np.sqrt((160*160)/(h*w)))
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr, (5,5), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    vch = clahe.apply(vch)
    hsv = cv2.merge([hch, sch, vch])

    red1   = cv2.inRange(hsv, HSV_RED1_LO,   HSV_RED1_HI)
    red2   = cv2.inRange(hsv, HSV_RED2_LO,   HSV_RED2_HI)
    yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    green  = cv2.inRange(hsv, HSV_GREEN_LO,  HSV_GREEN_HI)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    red1   = cv2.morphologyEx(red1, cv2.MORPH_OPEN, kernel, iterations=1)
    red2   = cv2.morphologyEx(red2, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel, iterations=1)
    green  = cv2.morphologyEx(green,  cv2.MORPH_OPEN, kernel, iterations=1)
    red1   = cv2.morphologyEx(red1, cv2.MORPH_CLOSE, kernel, iterations=1)
    red2   = cv2.morphologyEx(red2, cv2.MORPH_CLOSE, kernel, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=1)
    green  = cv2.morphologyEx(green,  cv2.MORPH_CLOSE, kernel, iterations=1)

    red = red1 + red2

    vnorm = vch.astype(np.float32) / 255.0
    r = float((red   /255.0 * vnorm).sum())
    y = float((yellow/255.0 * vnorm).sum())
    g = float((green /255.0 * vnorm).sum())

    total_px = bgr.shape[0]*bgr.shape[1]
    ratio_thresh = 0.002
    valid = (r > MIN_PIX_COUNT or r/total_px > ratio_thresh) or \
            (y > MIN_PIX_COUNT or y/total_px > ratio_thresh) or \
            (g > MIN_PIX_COUNT or g/total_px > ratio_thresh)
    if not valid:
        return "unknown"

    vals = {'red': r, 'yellow': y, 'green': g}
    best = max(vals, key=vals.get)
    second = sorted(vals.values(), reverse=True)[1]
    if second > 0 and (vals[best] / (second + 1e-6)) < 1.15:
        return "unknown"
    return best

def crop_roi(frame, roi_pts):
    (x1,y1),(x2,y2) = roi_pts
    xa, ya = max(0, min(x1,x2)), max(0, min(y1,y2))
    xb, yb = min(frame.shape[1]-1, max(x1,x2)), min(frame.shape[0]-1, max(y1,y2))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()

def infer_all_roi_colors(frame):
    colors = []
    for roi in roi_pts_list:
        if len(roi) != 2:
            colors.append("unknown")
            continue
        patch = crop_roi(frame, roi)
        colors.append(hsv_major_color(patch))
    return colors

def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def draw_stats_panel(frame, violations_count):
    panel_w, panel_h = 260, 70
    x0, y0 = 15, 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (25, 25, 25), -1)
    frame[y0:y0+panel_h, x0:x0+panel_h] = cv2.addWeighted(
        overlay[y0:y0+panel_h, x0:x0+panel_h], 0.45,
        frame[y0:y0+panel_h, x0:x0+panel_h], 0.55, 0
    )
    put_text(frame, "Violations:", (x0 + 12, y0 + 28), COLOR_TEXT, 0.7, 2)
    put_text(frame, f"{violations_count}", (x0 + 12, y0 + 58), COLOR_BAD, 0.95, 2)

def auto_open_video(path):
    try:
        if os.name == 'nt':
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    global drawing_mode, violations_count
    if not os.path.exists(VIDEO_IN):
        raise FileNotFoundError(f"Input video not found: {VIDEO_IN}")
    cap = cv2.VideoCapture(VIDEO_IN)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first frame.")
    H, W = first.shape[:2]
    cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("setup", setup_mouse_cb, param={'img_shape': (H, W)})
    while True:
        disp = first.copy()
        draw_setup_hud(disp)
        cv2.imshow("setup", disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('l'):
            drawing_mode = 'line'
            if len(line_pts) == 2:
                line_pts.clear()
        elif k == ord('r'):
            drawing_mode = 'roi'
        elif k == ord('u'):
            if roi_pts_list:
                if len(roi_pts_list[-1]) == 1:
                    roi_pts_list[-1].clear()
                else:
                    roi_pts_list.pop()
            elif line_pts:
                line_pts.clear()
        elif k == ord('c'):
            roi_pts_list.clear()
        elif k == ord(' ') and len(line_pts) == 2 and any(len(r)==2 for r in roi_pts_list):
            break
        elif k in (ord('q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("setup")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))
    model = YOLO(VEHICLE_MODEL_WEIGHTS)
    cap.release()
    stream = model.track(
        source=VIDEO_IN, conf=CONF_VEHICLE, iou=IOU_VEHICLE,
        tracker=TRACKER_CFG, stream=True, persist=True, verbose=True
    )

    p1, p2 = line_pts[0], line_pts[1]
    names_cache = None
    frame_idx = -1
    roi_colors = []

    for res in stream:
        frame_idx += 1
        frame = res.orig_img.copy()

        if names_cache is None and hasattr(res, 'names') and isinstance(res.names, dict):
            names_cache = {int(k): v for k, v in res.names.items()}

        if frame_idx % TRAFFIC_LIGHT_REFRESH == 0 or not roi_colors:
            roi_colors = infer_all_roi_colors(frame)

        any_red = any(c == 'red' for c in roi_colors)

        cv2.line(frame, p1, p2, COLOR_LINE, 2)
        draw_stats_panel(frame, violations_count)

        for idx, roi in enumerate(roi_pts_list):
            if len(roi) == 2:
                (x1,y1),(x2,y2) = roi
                rx1, ry1, rx2, ry2 = min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)
                cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), COLOR_ROI, 2)
                c = roi_colors[idx] if idx < len(roi_colors) else "unknown"
                put_text(frame, c.upper(), (rx1, max(20, ry1 - 8)), COLOR_ROI, 0.7, 2)

        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy  = boxes.xyxy.cpu().numpy()
            ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
            clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

            for bb, tid, cidx, cf in zip(xyxy, ids, clses, confs):
                cls_name = names_cache.get(int(cidx), str(cidx)) if names_cache else str(cidx)
                if cls_name not in VEH_OK or cf < CONF_VEHICLE:
                    continue

                x1, y1, x2, y2 = map(int, bb)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if tid not in prev_center:
                    prev_center[tid] = (cx, cy)

                crossed_now = segments_intersect(prev_center[tid], (cx, cy), p1, p2)
                prev_center[tid] = (cx, cy)

                if any_red and crossed_now:
                    violated_ids.add(tid)

                if tid in violated_ids:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BAD, 3)
                    put_text(frame, "VIOLATION", (x1, max(22, y1 - 10)), COLOR_BAD, 0.8, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
                    put_text(frame, f"{cls_name} ID{tid}", (x1, max(22, y1 - 10)), (180,255,180), 0.6, 2)

            violations_count = len(violated_ids)

        out.write(frame)

    out.release()
    try:
        if os.name == 'nt':
            os.startfile(VIDEO_OUT)  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', VIDEO_OUT], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(['xdg-open', VIDEO_OUT], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

if __name__ == "__main__":
    main()
