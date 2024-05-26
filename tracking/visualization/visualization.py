import cv2
import linecache
import matplotlib.pyplot as plt


def show_frame(
        seq_name: str,
        frame_num: int = 1,
        show_true_bbox: bool = False
        ) -> None:
    image = get_frame(seq_name, frame_num, show_true_bbox)
    cv2.imshow(seq_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_frame(
        seq_name: str,
        frame_num: int = 1,
        show_true_bbox: bool = False
        ):
    image_path = rf'tracking\data\UAV123\seq\{seq_name}\{frame_num:06}.jpg'
    image = cv2.imread(image_path)

    if (show_true_bbox):
        bbox = get_true_bbox(seq_name, frame_num)

        if bbox:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


def get_true_bbox(
        seq_name: str,
        frame_num: int
        ) -> tuple[int, int, int, int] | None:
    anno_path = rf'tracking\data\UAV123\bbox\{seq_name}.txt'
    bbox = linecache.getline(anno_path, frame_num).split(',')

    if 'NaN' in bbox:
        return None

    return tuple(map(int, bbox))


def show_predicted(
        seq_name: str,
        frame_num: int,
        tracker: str
        ) -> None:
    image_path = rf'tracking\data\UAV123\seq\{seq_name}\{frame_num:06}.jpg'
    image = cv2.imread(image_path)

    with open(rf'tracking\results\{tracker}\{seq_name}.txt', 'r') as fp:
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


def show_seqs():
    frame_nums = [1, 200, 1000]

    seq1 = [cv2.cvtColor(get_frame('car1', frame_nums[0], True), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(get_frame('car1', frame_nums[1], True), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(get_frame('car1', frame_nums[2], True), cv2.COLOR_BGR2RGB)]
    
    seq2 = [cv2.cvtColor(get_frame('group1', frame_nums[0], True), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(get_frame('group1', frame_nums[1], True), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(get_frame('group1', frame_nums[2], True), cv2.COLOR_BGR2RGB)]

    fig = plt.figure(figsize=(6, 6))

    for i in range(len(seq1)):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(seq1[i])
        plt.axis('off')
        plt.title(rf'Кадр {frame_nums[i]}')
    
    for i in range(len(seq2)):
        fig.add_subplot(2, 3, i + 4)
        plt.imshow(seq2[i])
        plt.axis('off')
        plt.title(rf'Кадр {frame_nums[i]}')

    plt.show()
