from track import track, track_all, count_fps
from visualization import show_result_frame
from metrics import calc_center_error, calc_region_overlap

if __name__ == '__main__':
    seq_name = 'car10'
    # track(seq_name)

    # show_result_frame(seq_name, 1300, 'kcf')

    # track_all('kcf')
    # track_all('csrt')

    # calc_center_error(rf'tracking_system\results\kcf', rf'tracking_system\data\UAV123\bbox', True)
    # calc_region_overlap(rf'tracking_system\results\kcf', rf'tracking_system\data\UAV123\bbox', True)

    # calc_center_error(rf'tracking_system\results\csrt', rf'tracking_system\data\UAV123\bbox', True)
    # calc_region_overlap(rf'tracking_system\results\csrt', rf'tracking_system\data\UAV123\bbox', True)

    # count_fps('kcf')
    # count_fps('csrt')
    pass
