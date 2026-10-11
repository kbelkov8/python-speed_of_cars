import cv2
import csv
from collections import defaultdict, deque
from ultralytics import YOLO


VIDEO_PATH = "ScreenRecorderProject49.mp4"
MODEL_PATH = "yolo26s.pt"

OUTPUT_VIDEO = "result.mp4"
OUTPUT_CSV = "cars.csv"

COUNT_LINE_P1 = (750, 100)
COUNT_LINE_P2 = (750, 1000)

SPEED_LINE_A_P1 = (390, 100)
SPEED_LINE_A_P2 = (390, 1000)

SPEED_LINE_B_P1 = (1330, 100)
SPEED_LINE_B_P2 = (1330, 1000)

DISTANCE_M = 4.2    # Реальное расстояние между линиями A и B (длина автомобиля)

CONF = 0.25
IMGSZ = 640 # Уменьшил для лучшей опознаваемости автомобилей


def point_side(point, p1, p2):
    """
    Знак положения точки относительно линии.
    """
    x, y = point

    return (
        (p2[0] - p1[0]) * (y - p1[1])
        - (p2[1] - p1[1]) * (x - p1[0])
    )


def crossed_line(prev_pt, curr_pt, p1, p2):
    """
    Проверка пересечения линии.
    """
    s1 = point_side(prev_pt, p1, p2)
    s2 = point_side(curr_pt, p1, p2)

    return s1 * s2 < 0


cap = cv2.VideoCapture(VIDEO_PATH)  # загрузка видео

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Frames:", frame_count)
print("Resolution:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Итоговое видео

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


model = YOLO(MODEL_PATH)

tracks_history = defaultdict(lambda: deque(maxlen=30))
counted_ids = set()
lineA_times = {}
lineB_times = {}
speed_by_id = {}
csv_rows = []
vehicle_counter = 0
frame_index = 0

# Основной цикл
while True:
    success, frame = cap.read()

    if not success:
        break

    current_time = frame_index / fps

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMGSZ,
        classes=[2],
        conf=CONF,
        verbose=False
    )[0]

    # Рисуем линии
    cv2.line(
        frame,
        COUNT_LINE_P1,
        COUNT_LINE_P2,
        (0, 0, 255),
        3
    )

    cv2.line(
        frame,
        SPEED_LINE_A_P1,
        SPEED_LINE_A_P2,
        (255, 0, 0),
        2
    )

    cv2.line(
        frame,
        SPEED_LINE_B_P1,
        SPEED_LINE_B_P2,
        (0, 255, 255),
        2
    )

    # Обработка объектов
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(
            boxes,
            ids,
            confs
        ):

            x1, y1, x2, y2 = map(int, box)

            # центр рамки
            cx = int((x1 + x2)/2)
            cy = int(y2)

            current_point = (cx, cy)
            history = tracks_history[track_id]

            if len(history) > 0:
                previous_point = history[-1]

                # движение слева направо
                direction_ok = current_point[0] > previous_point[0]

                # Подсчёт машин

                if (
                        crossed_line(
                            previous_point,
                            current_point,
                            COUNT_LINE_P1,
                            COUNT_LINE_P2
                        )
                        and direction_ok
                ):

                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        vehicle_counter += 1

                # Линия A
                if (
                    track_id not in lineA_times
                    and crossed_line(
                        previous_point,
                        current_point,
                        SPEED_LINE_A_P1,
                        SPEED_LINE_A_P2
                    )
                ):
                    lineA_times[track_id] = current_time

                # Линия B
                if (
                    track_id not in lineB_times
                    and crossed_line(
                        previous_point,
                        current_point,
                        SPEED_LINE_B_P1,
                        SPEED_LINE_B_P2
                    )
                ):
                    lineB_times[track_id] = current_time

                # Скорость
                if (
                    track_id in lineA_times
                    and track_id in lineB_times
                    and track_id not in speed_by_id
                ):

                    tA = lineA_times[track_id]
                    tB = lineB_times[track_id]

                    delta_t = tB - tA

                    if delta_t > 0:
                        speed_m_s = DISTANCE_M / delta_t
                        speed_km_h = speed_m_s * 3.6

                        speed_by_id[track_id] = speed_km_h

                        csv_rows.append([
                            track_id,
                            "forward",
                            round(current_time, 2),
                            round(tA, 2),
                            round(tB, 2),
                            round(delta_t, 2),
                            round(DISTANCE_M, 2),
                            round(speed_km_h, 2)
                        ])

            history.append(current_point)

            # Траектория
            pts = list(history)

            for i in range(1, len(pts)):
                cv2.line(
                    frame,
                    pts[i - 1],
                    pts[i],
                    (0, 255, 0),
                    2
                )

            # Рамка
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"ID:{track_id} {conf:.2f}"

            if track_id in speed_by_id:
                label += f" {speed_by_id[track_id]:.1f} km/h"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # Счётчик
    cv2.putText(
        frame,
        f"Cars: {vehicle_counter}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    writer.write(frame)
    frame_index += 1

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

# Сохранение CSV
cv2.destroyAllWindows()

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer_csv = csv.writer(f)

    writer_csv.writerow([
        "track_id",
        "direction",
        "crossing_time",
        "t_A",
        "t_B",
        "delta_t",
        "distance_m",
        "speed_km_h"
    ])

    writer_csv.writerows(csv_rows)


# Завершение
cap.release()
writer.release()

print("Обработка завершена")
print("Подсчитано автомобилей:", vehicle_counter)