import os

skip_seqs = ['bike2', 'car1_s', 'car2_s', 'car3_s', 'car4_s', 'car13', 'person1_s', 'person2_s', 'person3_s', 'person18', 'person20', 'person21',
             'truck1', 'truck4', 'wakeboard4', 'wakeboard10']

def count_frames():
    seqs_path = rf'tracking\data\UAV123\seq'

    total = 0
    min_f = 10000000000
    max_f = 0

    seqs = os.listdir(seqs_path)

    for seq in seqs:
        if not seq.split('.')[0] in skip_seqs:
                continue
        
        count_frames = len(os.listdir(rf'{seqs_path}\{seq}'))

        if (count_frames < min_f):
            min_f = count_frames
        
        if (count_frames > max_f):
            max_f = count_frames
        
        total += count_frames
        
    print('total: ', total)
    print('min: ', min_f)
    print('max: ', max_f)
    print('mean: ', total / len(seqs))
