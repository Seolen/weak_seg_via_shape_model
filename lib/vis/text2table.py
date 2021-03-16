import numpy as np


def build_metric_table(dirname, select_lines=[0, 1]):
    # 1. load dir, load paths, save dir
    keys = [0.1, 0.2, 0.3, 0.4, 0.5]
    text_paths = {key: ['log_process_prob_%.1f_2.0.txt'%key, 'log_process_probsdf_%.1f_2.0.txt'%key] for key in keys}
    log_path = dirname + '/log_table_pseudo.txt'
    log = open(log_path, 'w')
    log.write('Note: line1-weak pred metric, line2-filter prob metric, line3-filter prob&sdf metric\n\n')

    # 2. read and write to file
    for key, paths in text_paths.items():
        log.write('%.1f\n' % key)
        for ith, path in enumerate(paths):
            selects = select_lines if ith == 0 else select_lines[-1:]
            whole_path = dirname + '/' + path
            try:
                with open(whole_path, 'r') as file:
                    for line_index, line in enumerate(file):
                        if line_index in selects:
                            log.write(line)
            except:
                pass
        log.write('\n')

    # 3. save
    log.close()


if __name__ == '__main__':
    dir_prefix = '/group/lishl/weak_exp/output/'
    prostate_dirs = [
        # '0825_pro_balance_r1_01_train',
        # '0825_pro_balance_r3_01_train',
        # '0825_pro_balance_r5_01_train',
        '0825_pro_aelo_11_train', '0825_pro_aelo_31_train', '0825_pro_aelo_51_train'
    ]
    trachea_dirs = [
        # '0825_trachea_r1_01_train',
        # '0825_trachea_r3_01_train',
        # '0825_trachea_r5_01_train',
        '0825_trachea_aelo_11_train', '0825_trachea_aelo_31_train', '0825_trachea_aelo_51_train',
    ]
    # prostate_dirs = ['0907_pro_r1_em21_filter40_train']
    # trachea_dirs = ['0907_trachea_r1_em21_filter50_train', '0907_trachea_r3_em21_filter50_train']

    for _dir in prostate_dirs:
        build_metric_table(dir_prefix + _dir, select_lines=[162, 163])
    for _dir in trachea_dirs:
        build_metric_table(dir_prefix + _dir, select_lines=[122, 123])
