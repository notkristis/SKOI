import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np


class PlanarObjectTracker:
    """
    Трекинг плоского текстурированного объекта на основе особых точек (SIFT / ORB)
    и вычисления перспективного преобразования (гомографии) методом RANSAC.
    """

    def __init__(
        self,
        detector_type: str = "SIFT",
        ratio_thresh: float = 0.75,
        min_matches: int = 12,
        ransac_thresh: float = 5.0,
        label: str = "Tracked Object",
    ):
        self.detector_type = detector_type.upper()
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self.label = label

        # Инициализация детектора и матчера
        if self.detector_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=2000)
            # Для SIFT (float-дескрипторы) используем FLANN с KD-деревьями
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=2500)
            # Для ORB (бинарные дескрипторы) используем расстояние Хэмминга
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Неподдерживаемый детектор: {self.detector_type}. Используйте SIFT или ORB.")

        self.ref_kp = None
        self.ref_des = None
        self.ref_corners = None
        self.ref_img = None

    def initialize_from_frame(self, frame: np.ndarray, bbox: tuple = None) -> bool:
        """
        Инициализирует эталонный объект по первому кадру.
        По умолчанию берется весь первый кадр (с небольшим отступом от краев 2%,
        чтобы рамка красиво отрисовывалась). Если передан bbox (x, y, w, h), берется область.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        if bbox is not None:
            x, y, w, h = bbox
            roi_gray = gray[y : y + h, x : x + w]
            self.ref_img = frame[y : y + h, x : x + w].copy()
            kp, self.ref_des = self.detector.detectAndCompute(roi_gray, None)
            # Смещаем координаты точек в систему координат исходного эталона
            self.ref_kp = kp
            self.ref_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        else:
            self.ref_img = frame.copy()
            self.ref_kp, self.ref_des = self.detector.detectAndCompute(gray, None)
            # Углы эталонного изображения (весь первый кадр)
            margin_x, margin_y = int(w_img * 0.01), int(h_img * 0.01)
            self.ref_corners = np.float32([
                [margin_x, margin_y],
                [margin_x, h_img - 1 - margin_y],
                [w_img - 1 - margin_x, h_img - 1 - margin_y],
                [w_img - 1 - margin_x, margin_y],
            ]).reshape(-1, 1, 2)

        if self.ref_des is None or len(self.ref_kp) < self.min_matches:
            print(f"[ОШИБКА] На первом кадре найдено слишком мало ключевых точек: {len(self.ref_kp) if self.ref_kp else 0}")
            return False

        print(f"[INFO] Эталон инициализирован ({self.detector_type}): найдено {len(self.ref_kp)} ключевых точек.")
        return True

    @staticmethod
    def _is_valid_quadrilateral(pts: np.ndarray, frame_shape: tuple, min_area: float = 1000.0) -> bool:
        """
        Проверяет, является ли спроецированная рамка выпуклым четырехугольником
        разумной площади (защита от вырожденных гомографий).
        """
        contour = np.int32(pts)
        if not cv2.isContourConvex(contour):
            return False

        area = cv2.contourArea(contour)
        h, w = frame_shape[:2]
        max_area = (w * h) * 1.5  # Допускаем небольшой выход за границы кадра при приближении

        if area < min_area or area > max_area:
            return False

        # Проверка, что стороны не выродились в линию
        pts_2d = pts.reshape(4, 2)
        side_lengths = [np.linalg.norm(pts_2d[i] - pts_2d[(i + 1) % 4]) for i in range(4)]
        if min(side_lengths) < 20:
            return False

        return True

    def process_frame(self, frame: np.ndarray):
        """
        Обрабатывает очередной кадр видео.
        Возвращает: (кадр с отрисовкой, статус обнаружения, кол-во хороших совпадений, кол-во инлайеров)
        """
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < self.min_matches:
            return output_frame, False, 0, 0

        # Сопоставление дескрипторов (k=2 для теста отношения Лоу)
        matches = self.matcher.knnMatch(self.ref_des, des_frame, k=2)

        # Фильтрация совпадений по критерию Дэвида Лоу (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return output_frame, False, len(good_matches), 0

        # Извлекаем координаты сопоставленных точек
        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Вычисляем матрицу гомографии методом RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        if H is None or mask is None:
            return output_frame, False, len(good_matches), 0

        inliers_count = int(np.sum(mask))
        if inliers_count < self.min_matches:
            return output_frame, False, len(good_matches), inliers_count

        # Проецируем углы эталонного объекта на текущий кадр
        projected_corners = cv2.perspectiveTransform(self.ref_corners, H)

        # Проверяем геометрическую адекватность полученной рамки
        if not self._is_valid_quadrilateral(projected_corners, frame.shape):
            return output_frame, False, len(good_matches), inliers_count

        # Отрисовка контура (рамки) вокруг объекта
        poly_pts = np.int32(projected_corners)
        cv2.polylines(output_frame, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

        # Отрисовка инлайеров (ключевых точек внутри объекта) для наглядности
        for i, m in enumerate(good_matches):
            if mask[i]:
                pt = tuple(map(int, kp_frame[m.trainIdx].pt))
                cv2.circle(output_frame, pt, 3, (0, 200, 255), -1)

        # Отрисовка плашки с подписью объекта над верхней гранью рамки
        pts_2d = poly_pts.reshape(4, 2)
        top_idx = np.argmin(pts_2d[:, 1])
        label_x = int(np.clip(pts_2d[top_idx][0], 10, frame.shape[1] - 250))
        label_y = int(max(pts_2d[top_idx][1] - 12, 30))

        text = f"{self.label} (inliers: {inliers_count})"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            output_frame,
            (label_x - 4, label_y - th - 6),
            (label_x + tw + 4, label_y + baseline),
            (0, 140, 0),
            -1,
        )
        cv2.putText(
            output_frame,
            text,
            (label_x, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output_frame, True, len(good_matches), inliers_count


def process_video(
    video_path: str,
    output_dir: str = "results",
    detector_type: str = "SIFT",
    label: str = "Target Object",
    select_roi: bool = False,
    show_preview: bool = True,
):
    """Обрабатывает один видеофайл, сохраняет видео с трекингом и итоговую статистику."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть видео: {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"[ОШИБКА] Не удалось прочитать первый кадр из {video_path}")
        cap.release()
        return

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem

    # Если указан флаг --roi, позволяем выделить объект на первом кадре вручную
    bbox = None
    if select_roi:
        print("[INFO] Выделите объект мышкой и нажмите ENTER или SPACE.")
        bbox = cv2.selectROI("Select Object on First Frame", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object on First Frame")
        if bbox[2] == 0 or bbox[3] == 0:
            bbox = None

    tracker = PlanarObjectTracker(detector_type=detector_type, label=label)
    if not tracker.initialize_from_frame(first_frame, bbox=bbox):
        cap.release()
        return

    # Настройка записи выходного видео
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = os.path.join(output_dir, f"{video_name}_tracked_{detector_type.lower()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    detected_frames = 0
    total_inliers = []
    start_time = time.time()

    # Перематываем в начало, чтобы обработать и первый кадр тоже
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        vis_frame, is_detected, matches_cnt, inliers_cnt = tracker.process_frame(frame)

        if is_detected:
            detected_frames += 1
            total_inliers.append(inliers_cnt)

        # Статусная строка в левом верхнем углу кадра
        status_text = f"Frame: {frame_idx}/{total_frames} | Algo: {detector_type} | Matches: {matches_cnt} | Inliers: {inliers_cnt}"
        cv2.putText(vis_frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis_frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        writer.write(vis_frame)

        # Сохраняем превью-кадр из середины видео для вставки в отчет GitHub
        if frame_idx == max(1, total_frames // 3) and is_detected:
            preview_path = os.path.join(output_dir, f"{video_name}_sample_frame.jpg")
            cv2.imwrite(preview_path, vis_frame)

        if show_preview:
            cv2.imshow(f"Tracking - {video_name} (Press Q to skip)", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Остановлено пользователем.")
                break

    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    avg_inliers = np.mean(total_inliers) if total_inliers else 0.0
    detection_rate = (detected_frames / frame_idx * 100) if frame_idx > 0 else 0.0

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Видео обработано: {video_path}")
    print(f"Сохранено в:      {out_video_path}")
    print(f"Всего кадров:     {frame_idx} | Объект найден на: {detected_frames} ({detection_rate:.1f}%)")
    print(f"Среднее кол-во инлайеров: {avg_inliers:.1f} | Средняя скорость: {avg_fps:.1f} FPS")
    print("-" * 65)


def main():
    parser = argparse.ArgumentParser(description="ЛР2: Трекинг плоского объекта по ключевым точкам")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Путь к видеофайлу ИЛИ к папке с видеофайлами (например, data/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Папка для сохранения результатов (по умолчанию: results)",
    )
    parser.add_argument(
        "--detector",
        "-d",
        type=str,
        default="SIFT",
        choices=["SIFT", "ORB"],
        help="Алгоритм поиска ключевых точек: SIFT (точнее) или ORB (быстрее)",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        default="Object",
        help="Подпись рамки отслеживаемого объекта",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Выделить объект на первом кадре вручную (если на 1-м кадре объект не во весь экран)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Отключить показ окна во время обработки (удобно для пакетной обработки)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    if input_path.is_dir():
        videos = sorted([str(p) for p in input_path.iterdir() if p.suffix.lower() in video_extensions])
        if not videos:
            print(f"[ОШИБКА] В папке {input_path} не найдено видеофайлов.")
            return
        for v in videos:
            process_video(
                v,
                output_dir=args.output,
                detector_type=args.detector,
                label=args.label,
                select_roi=args.roi,
                show_preview=not args.no_preview,
            )
    elif input_path.is_file():
        process_video(
            str(input_path),
            output_dir=args.output,
            detector_type=args.detector,
            label=args.label,
            select_roi=args.roi,
            show_preview=not args.no_preview,
        )
    else:
        print(f"[ОШИБКА] Путь {input_path} не существует.")


if __name__ == "__main__":
    main()
