from visualization import show_frame, show_predicted, show_seqs

from trackers import track_all, count_fps, count_failrate
from metrics import calc_center_error, calc_region_overlap
from datasets import count_frames


if __name__ == '__main__':
    # show_frame('car10', 1300, True)

    # count_frames()

    # track_all('kcf')
    # track_all('csrt')

    # seq_name = 'car18'
    # show_predicted(seq_name, 100, 'kcf')
    # show_predicted(seq_name, 100, 'csrt')

    # count_fps('kcf')
    # count_fps('csrt')

    # count_failrate('kcf')
    # count_failrate('csrt')

    # calc_center_error(rf'tracking\results\kcf', rf'tracking\data\UAV123\bbox', True)
    # calc_region_overlap(rf'tracking\results\kcf', rf'tracking\data\UAV123\bbox', True)

    # calc_center_error(rf'tracking\results\csrt', rf'tracking\data\UAV123\bbox', True)
    # calc_region_overlap(rf'tracking\results\csrt', rf'tracking\data\UAV123\bbox', True)

    pass
