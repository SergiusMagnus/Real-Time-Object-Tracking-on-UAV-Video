import cv2

def show_result_frame(
        seq_name: str,
        frame_num: int,
        tracker: str
        ) -> None:
    image_path = rf'tracking_system\data\UAV123\seq\{seq_name}\{frame_num:06}.jpg'
    image = cv2.imread(image_path)

    with open(rf'tracking_system\results\{tracker}\{seq_name}.txt', 'r') as fp:
        text_list = fp.readlines()

    if len(text_list) >= frame_num:
        x, y, w, h = map(int, text_list[frame_num - 1].split(' '))

        start_point = (x, y)
        end_point = (x + w, y + h)

        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imshow(seq_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()