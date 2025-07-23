import datetime

import cv2
import numpy as np
from ultralytics import YOLO


class iPhoneDeskClassifier:
    def __init__(self, camera_index=1):
        print("Loading YOLO models...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.pose_model = YOLO("yolov8n-pose.pt")

        print(f"Connecting to iPhone camera (index {camera_index})...")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            print("iPhone camera setup tips:")
            print("1. Make sure iPhone is unlocked and nearby")
            print("2. Both devices signed into same Apple ID")
            print("3. WiFi and Bluetooth enabled on both")
            print("4. Try different camera indices (0, 1, 2)")
            return

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if hasattr(cv2, "CAP_PROP_BUFFER_SIZE"):
                self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)

        except Exception as e:
            print(f"Warning: Could not set camera properties: {e}")

        print(f"iPhone camera initialized: {self.cap.isOpened()}")
        if self.cap.isOpened():
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Resolution: {width}x{height}, FPS: {fps}")

    def find_available_cameras(self):
        """Find all available camera indices"""
        available_cameras = []
        print("Scanning for available cameras...")

        for i in range(8):
            print(f"Testing camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(
                        {
                            "index": i,
                            "resolution": frame.shape,
                            "name": self.get_camera_name(i, frame.shape),
                        }
                    )
                    print(f"✅ Camera {i}: Available - Resolution: {frame.shape}")
                else:
                    print(f"❌ Camera {i}: Opened but no frame")
                cap.release()
            else:
                print(f"❌ Camera {i}: Not available")

        return available_cameras

    def get_camera_name(self, index, shape):
        """Guess camera type based on index and resolution"""
        height = shape[0]
        width = shape[1]

        if index == 0:
            return "MacBook Built-in"
        elif width >= 1920 or height >= 1080:
            return "iPhone (High-res)"
        elif width >= 1280:
            return "iPhone/External (Medium-res)"
        else:
            return "Unknown Camera"

    def test_camera(self):
        """Test camera functionality with preview"""
        if not self.cap.isOpened():
            print("Camera not available")
            return False

        print("Testing camera... Press any key to continue")
        ret, frame = self.cap.read()
        if ret and frame is not None:
            print(f"✅ Camera working - Frame shape: {frame.shape}")

            test_frame = frame.copy()
            cv2.putText(
                test_frame,
                "iPhone Camera Test",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                test_frame,
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                test_frame,
                "Press any key to continue",
                (50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("iPhone Camera Test", test_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("iPhone Camera Test")
            return True
        else:
            print("❌ Camera not capturing frames")
            return False

    def is_person_at_desk(
        self, person_box, laptop_boxes, keyboard_boxes, mouse_boxes, cell_phone_boxes
    ):
        """Enhanced desk detection optimized for iPhone camera perspective"""
        px1, py1, px2, py2 = person_box
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        person_bottom = py2

        work_items = laptop_boxes + keyboard_boxes + mouse_boxes + cell_phone_boxes

        if not work_items:
            frame_width = 1920
            person_center_x = person_center[0]

            if (
                person_center_x > frame_width * 0.2
                and person_center_x < frame_width * 0.8
                and person_center[1] > 200
            ):
                return True, None

        for item_box in work_items:
            ix1, iy1, ix2, iy2 = item_box
            item_center = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)

            distance = np.sqrt(
                (person_center[0] - item_center[0]) ** 2
                + (person_center[1] - item_center[1]) ** 2
            )

            if distance < 500 and person_bottom > iy1 and person_center[1] < iy2 + 300:
                return True, item_box

        return False, None

    def analyze_working_pose(self, pose_keypoints):
        """Analyze pose optimized for iPhone camera angle and quality"""
        if pose_keypoints is None or len(pose_keypoints) == 0:
            return False, 0.0

        try:
            keypoints = pose_keypoints[0].xy[0].cpu().numpy()

            if len(keypoints) < 17:
                return False, 0.0

            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_elbow = keypoints[11]
            right_elbow = keypoints[12]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            left_hip = keypoints[13]
            right_hip = keypoints[14]

            working_score = 0.0

            if (
                left_shoulder[0] > 0
                and right_shoulder[0] > 0
                and left_hip[0] > 0
                and right_hip[0] > 0
            ):
                shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_avg_y = (left_hip[1] + right_hip[1]) / 2

                if hip_avg_y > shoulder_avg_y + 50:
                    working_score += 0.3

                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                if shoulder_diff < 40:
                    working_score += 0.2

            if nose[0] > 0 and left_shoulder[0] > 0 and right_shoulder[0] > 0:
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

                if nose[1] < shoulder_center_y - 20:
                    working_score += 0.2

                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                head_centered = abs(nose[0] - shoulder_center_x) < 50
                if head_centered:
                    working_score += 0.1

            hands_visible = (
                left_wrist[0] > 0
                and left_wrist[1] > 0
                and right_wrist[0] > 0
                and right_wrist[1] > 0
            )

            if hands_visible and left_elbow[0] > 0 and right_elbow[0] > 0:
                left_typing = left_wrist[1] > left_elbow[1] - 40
                right_typing = right_wrist[1] > right_elbow[1] - 40

                if left_typing and right_typing:
                    working_score += 0.3
                elif left_typing or right_typing:
                    working_score += 0.15

                if left_shoulder[0] > 0 and right_shoulder[0] > 0:
                    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2

                    if shoulder_avg_y < wrist_avg_y < shoulder_avg_y + 200:
                        working_score += 0.1

            visible_parts = sum(
                [
                    1
                    for point in [
                        nose,
                        left_shoulder,
                        right_shoulder,
                        left_elbow,
                        right_elbow,
                        left_wrist,
                        right_wrist,
                    ]
                    if point[0] > 0 and point[1] > 0
                ]
            )

            if visible_parts >= 6:
                working_score += 0.1

            is_working = working_score > 0.5
            return is_working, working_score

        except Exception as e:
            print(f"Pose analysis error: {e}")
            return False, 0.0

    def classify_desk_activity(self, frame):
        """Main classification function optimized for iPhone camera quality"""
        try:
            obj_results = self.yolo_model.predict(frame, conf=0.3, verbose=False)

            pose_results = self.pose_model.predict(frame, conf=0.3, verbose=False)

            persons = []
            laptops = []
            keyboards = []
            mice = []
            cell_phones = []
            chairs = []

            for result in obj_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        coords = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()

                        if cls == 0 and conf > 0.5:
                            persons.append(coords)
                        elif cls == 63 and conf > 0.25:
                            laptops.append(coords)
                        elif cls == 66 and conf > 0.25:
                            keyboards.append(coords)
                        elif cls == 64 and conf > 0.25:
                            mice.append(coords)
                        elif cls == 67 and conf > 0.25:
                            cell_phones.append(coords)
                        elif cls == 56 and conf > 0.3:
                            chairs.append(coords)

            classifications = []
            for i, person_box in enumerate(persons):
                at_desk, work_item = self.is_person_at_desk(
                    person_box, laptops, keyboards, mice, cell_phones
                )

                if at_desk:
                    is_working_pose, pose_confidence = self.analyze_working_pose(
                        pose_results[0].keypoints if len(pose_results) > 0 else None
                    )

                    if is_working_pose and pose_confidence > 0.6:
                        status = f"Working (conf: {pose_confidence:.2f})"
                        color = (0, 255, 0)
                    elif pose_confidence > 0.3:
                        status = f"At desk (conf: {pose_confidence:.2f})"
                        color = (0, 255, 255)
                    else:
                        status = "At desk (idle)"
                        color = (255, 165, 0)
                else:
                    status = "Away from desk"
                    color = (0, 0, 255)

                classifications.append(
                    {
                        "person_id": i,
                        "box": person_box,
                        "status": status,
                        "color": color,
                        "at_desk": at_desk,
                        "confidence": pose_confidence,
                    }
                )

            return (
                classifications,
                laptops,
                keyboards,
                mice,
                cell_phones,
                chairs,
                pose_results,
            )

        except Exception as e:
            print(f"Classification error: {e}")
            return [], [], [], [], [], [], []

    def draw_results(
        self, frame, classifications, laptops, keyboards, mice, cell_phones, chairs
    ):
        """Draw detection results optimized for iPhone camera resolution"""
        try:
            equipment_data = [
                (laptops, "Laptop", (255, 0, 255)),
                (keyboards, "Keyboard", (0, 255, 255)),
                (mice, "Mouse", (255, 255, 0)),
                (cell_phones, "Phone", (255, 128, 0)),
                (chairs, "Chair", (128, 255, 128)),
            ]

            for equipment_list, label, color in equipment_data:
                for item_box in equipment_list:
                    x1, y1, x2, y2 = map(int, item_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            for classification in classifications:
                box = classification["box"]
                status = classification["status"]
                color = classification["color"]

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                font_scale = 0.8
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(
                    status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                cv2.rectangle(
                    frame, (x1, y1 - 45), (x1 + text_width + 20, y1), color, -1
                )

                cv2.putText(
                    frame,
                    status,
                    (x1 + 10, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

            return frame

        except Exception as e:
            print(f"Drawing error: {e}")
            return frame

    def run_classification(self):
        """Main loop optimized for iPhone camera"""
        if not self.cap.isOpened():
            print("Camera not available. Exiting.")
            return

        print("Starting iPhone camera classification...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  't' - Test camera")
        print("  'f' - Toggle full processing (slower but more accurate)")
        print("  'c' - Show available cameras")

        frame_count = 0
        skip_frames = 1
        full_processing = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame - trying to reconnect...")
                self.cap.release()
                self.cap = cv2.VideoCapture(1)
                continue

            frame_count += 1

            process_frame = (frame_count % (skip_frames + 1) == 0) or full_processing

            if not process_frame:
                cv2.imshow("iPhone Desk Work Classifier", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            classifications, laptops, keyboards, mice, phones, chairs, pose_results = (
                self.classify_desk_activity(frame)
            )

            frame = self.draw_results(
                frame, classifications, laptops, keyboards, mice, phones, chairs
            )

            working_count = sum(1 for c in classifications if "Working" in c["status"])
            at_desk_count = sum(1 for c in classifications if c["at_desk"])
            total_people = len(classifications)

            stats_text = f"People: {total_people} | At Desk: {at_desk_count} | Working: {working_count}"
            cv2.putText(
                frame,
                stats_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
            )

            total_equipment = (
                len(laptops) + len(keyboards) + len(mice) + len(phones) + len(chairs)
            )
            equipment_text = f"Equipment: {total_equipment} items"
            cv2.putText(
                frame,
                equipment_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            mode_text = f"Mode: {'Full' if full_processing else f'Skip {skip_frames}'}"
            cv2.putText(
                frame,
                mode_text,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2,
            )

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            camera_text = f"iPhone Camera - {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(
                frame,
                camera_text,
                (frame.shape[1] - 400, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2,
            )

            cv2.imshow("iPhone Desk Work Classifier", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"iphone_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord("t"):
                print("Testing camera...")
                self.test_camera()
            elif key == ord("f"):
                full_processing = not full_processing
                print(f"Full processing: {'ON' if full_processing else 'OFF'}")
            elif key == ord("c"):
                available = self.find_available_cameras()
                print("Available cameras:")
                for cam in available:
                    print(
                        f"  Index {cam['index']}: {cam['name']} - {cam['resolution']}"
                    )

        self.cap.release()
        cv2.destroyAllWindows()
        print("iPhone classification ended.")


def main():
    """Main function for iPhone camera setup"""
    try:
        print("=== iPhone Desk Work Classifier ===")
        print("Setting up iPhone camera connection...")

        temp_classifier = iPhoneDeskClassifier(camera_index=0)
        available_cameras = temp_classifier.find_available_cameras()
        temp_classifier.cap.release() if temp_classifier.cap else None

        if not available_cameras:
            print("\n❌ No cameras found!")
            print("\nTroubleshooting:")
            print("1. For Continuity Camera (macOS 13+):")
            print("   - iPhone unlocked and nearby")
            print("   - Same Apple ID on both devices")
            print("   - WiFi & Bluetooth enabled")
            print("2. For third-party apps:")
            print("   - Install Camo, EpocCam, or similar")
            print("   - Connect iPhone via USB or WiFi")
            return

        print(f"\n✅ Found {len(available_cameras)} camera(s):")
        for cam in available_cameras:
            print(f"   Index {cam['index']}: {cam['name']} - {cam['resolution']}")

        iphone_camera_index = 1
        if len(available_cameras) > 1:
            for cam in available_cameras:
                if cam["resolution"][1] >= 1920:
                    iphone_camera_index = cam["index"]
                    break

        print(f"\nUsing camera index {iphone_camera_index} for iPhone")
        print("If this is wrong, modify the camera_index in the code\n")

        classifier = iPhoneDeskClassifier(camera_index=iphone_camera_index)

        if classifier.cap and classifier.cap.isOpened():
            if classifier.test_camera():
                classifier.run_classification()
            else:
                print("Camera test failed")
        else:
            print(f"Failed to initialize camera {iphone_camera_index}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
