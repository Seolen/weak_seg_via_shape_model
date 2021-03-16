from batchgenerators.dataloading import SlimDataLoaderBase
from glob import glob
import os
import numpy as np
from scipy import io
import copy
import ipdb
from lib.configs.parse_arg import opt, args
from lib.datasets.lib_sample import trachea_libs, prostate_libs, la_libs

class BGDataset(SlimDataLoaderBase):
    def __init__(self, data_dir, batch_size, phase='train', split_file=None,
                 use_weak=False, use_duplicate=False, weak_key='weak_label',
                 use_conf=False,
                 shuffle=False, seed_for_shuffle=None, infinite=False, return_incomplete=False,
                 num_threads_in_multithreaded=1,
                 trans_depth=False,
                 use_select=False, select_params=None, weak_label_dir='',):
        """
        :param data: datadir or data dictionary
        :param batch_size:
        :param batch_size: dataset name, 'Prostate' or 'Heart'
        :param split_file: 'train.txt' or 'val.txt'
        :param shuffle: return shuffle order or not
        :param trans_depth: tranpose depth dimension to the second dim, e.g. (B, W, H, D) -> (B, D, W, H)

        :param use_select: use select pseudo label
        :param weak_label_dir: pseudo_label_dir in fact

        Each iteration: return {'data': ,'seg': }, with shape (B, W, H, D)
        """
        super(BGDataset, self).__init__(data_dir, batch_size, num_threads_in_multithreaded)
        self.batch_size = batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.infinite = infinite
        self.use_weak = use_weak
        self.use_duplicate = use_duplicate
        self.weak_key = weak_key
        self.pseudo_label_dir = weak_label_dir

        self.em_save_pseudo_dir= ''
        if opt.data.em_save_pseudo_dir != '':   # em pseudo dir
            self.em_save_pseudo_dir = opt.data.em_save_pseudo_dir + '%s/' % args.id

        self.use_select = use_select
        self.samples_select = []

        # use all samples or selected sampels (on avg prob)
        # TODO
        if 'Promise' in data_dir:
            sample_libs = prostate_libs
        elif 'SegTHOR' in data_dir:
            sample_libs = trachea_libs
        elif 'LA_' in data_dir:
            sample_libs = la_libs

        if not use_select:
            self.samples = sample_libs[phase]
            if self.pseudo_label_dir != '':
                self.samples_select = self.samples.copy()
        else:
            _group = 'top_prob'
            if phase == 'train':
                label_percent, use_pred_num = select_params['label_percent'], select_params['use_pred_num']
                if use_pred_num == 1000:    # 1000 means all samples
                    self.samples_select = sample_libs[phase]
                else:
                    self.samples_select = sample_libs['train_select'][label_percent][_group][use_pred_num]
                self.samples = sample_libs[phase]
            else:
                self.samples = sample_libs[phase]

        self.sample_paths = [os.path.join(data_dir, '%s.mat'%_id) for _id in self.samples]
        if opt.sp.use_type != '':
            sp_dir = os.path.join(data_dir, '../{}/'.format(opt.sp.dir))
            self.sp_paths = [os.path.join(sp_dir, '%s.mat'%_id) for _id in self.samples]

        # inner variables
        self.indices = list(range(len(self.samples)))
        seed_for_shuffle = args.seed
        self.rs = np.random.RandomState(seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.return_incomplete = return_incomplete
        self.last_reached = False
        self.number_of_threads_in_multithreaded = 1

    def __len__(self):
        return len(self.samples)//self.batch_size

    def reset(self):
        assert self.indices is not None
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True
        self.rs.seed(self.rs.randint(0, 999999999))
        if self.shuffle:
            self.rs.shuffle(self.indices)
        self.last_reached = False

    def get_indices(self):
        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=None)

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])
                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and (not self.last_reached or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    def generate_train_batch(self):
        # similar to __getiterm__(index), but not index as params
        indices = self.get_indices()
        data = {'image': [], 'label': []}
        if self.use_weak:
            data['weak_label'] = []
        if self.pseudo_label_dir != '':
            data['pseudo_label'] = []
        if self.use_select and self.pseudo_label_dir == '':     # No immediate pseudo label, but pseudo tag
            data['pseudo_tag'] = []

        # em pseudo load: padding if initialization else pseudo labels
        if self.em_save_pseudo_dir != '':
            data['pseudo_label'] = []


        if opt.sp.use_type != '':
            data['sp_edge'] = []
            # data['sp_region'] = []
        for ith, index in enumerate(indices):
            file_path = self.sample_paths[index]
            file_data = io.loadmat(file_path)   # 40,160,160
            if opt.sp.use_type != '':
                sp_path = self.sp_paths[index]
                sp_data = io.loadmat(sp_path)   # 40,160,160
                for key, value in sp_data.items():
                    if key in data.keys():
                        file_data[key] = sp_data[key]
            for key, value in file_data.items():
                if key in data.keys():
                    value = np.expand_dims(value, 0)    # channel=1
                    data[key].append(value)
                    # intensity transformation
                    pass
            # replace weak label with predictions (of Baseline/AE)
            if self.pseudo_label_dir != '':
                if self.phase == 'train' and (self.samples[index] in self.samples_select):
                    weak_label_path = os.path.join(self.pseudo_label_dir, self.samples[index] + '.mat')
                    value = io.loadmat(weak_label_path)['pred']
                    # weak label regularize pseudo label: bg region all 0, fg scribble all 1.
                    weak_label = data['weak_label'][ith]
                    pseudo_label = np.expand_dims(value, 0)
                    pseudo_label[weak_label == 0] = 0; pseudo_label[weak_label == 1] = 1
                    data['pseudo_label'].append(pseudo_label)
                else:
                    data['pseudo_label'].append(np.zeros_like(data['weak_label'][ith], dtype=data['weak_label'][ith].dtype))
            if 'pseudo_tag' in data.keys():
                tag = 1 if self.samples[index] in self.samples_select else 0
                data['pseudo_tag'].append(tag)

            if self.use_duplicate and self.use_weak:
                data['weak_label'] = data['label'].copy()
            if self.phase == 'train' and index == opt.shape.n_model:
                data['weak_label'] = copy.deepcopy(data['label'])
            if self.phase == 'test' and ('weak_label' not in file_data.keys()):
                data['weak_label'].append(np.zeros_like(data['image'][ith], dtype=np.uint8))

            # em pseudo load: padding if initialization else pseudo labels
            em_pseudo_path = os.path.join(self.em_save_pseudo_dir, self.samples[index] + '.mat')
            if self.em_save_pseudo_dir != '':
                if (self.phase != 'train') or (not os.path.exists(em_pseudo_path)):
                    data['pseudo_label'].append(np.zeros_like(data['weak_label'][ith], dtype=data['weak_label'][ith].dtype))
                else:
                    # value = io.loadmat(em_pseudo_path)['pred']
                    pseudo_mat = io.loadmat(em_pseudo_path); value = pseudo_mat['pred']
                    em_pseudo_label = np.expand_dims(value, 0)
                    data['pseudo_label'].append(em_pseudo_label)
                    # replace weak label
                    if (opt.model.filter_extra == 'modify_initial_weak') and ('weak_label' in pseudo_mat.keys()):
                        data['weak_label'][-1] = np.expand_dims(pseudo_mat['weak_label'], 0)

        for key, value in data.items():
            data[key] = np.array(value)

        data['data'] = data.pop('image')
        data['seg'] = data.pop('label')
        return data


        # if self.use_weak:
        #     return {'data': data['image'], 'seg': data['label'], 'weak_label': data['weak_label']}
        # return {'data': data['image'], 'seg': data['label']}

    def load_split_file(self, split_file):
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                 line = line.strip()
                 if line != '':
                    samples.append(line)
        return samples



if __name__ == '__main__':
    data_dir = '/group/lishl/weak_datasets/0108_SegTHOR/processed_trachea_train_weak_percent_0.3_random/expand_mat/'
    # data_dir = '/group/lishl/weak_datasets/Promise12/processed_train_weak_percent_0.3_random/normalize_mat/'
    batch_size = 2
    shuffle = True
    phase = 'train' #'val'

    use_select = True
    select_params = {'label_percent': 30, 'use_pred_num': 5}
    use_weak = True
    weak_label_dir = '/group/lishl/weak_exp/output/0806_trachea_aelo_02_train/mat/'
    # opt.sp.use_type = 'decoder'

    dataset = BGDataset(data_dir, batch_size, phase=phase, shuffle=shuffle,
                        use_weak=use_weak, use_select=use_select, select_params=select_params, weak_label_dir=weak_label_dir,

                        )
    print(len(dataset))
    for batch in dataset:
        print(batch.keys())
        print(batch['data'].shape)
        print(batch['seg'].shape)
        if 'pseudo_label' in batch.keys():
            print(batch['pseudo_label'].shape)
            print(np.unique(batch['pseudo_label']))
        continue
        ipdb.set_trace()
        # if use_weak:
        #     print(batch['weak_label'].shape)
        #     print(np.unique(batch['weak_label']))
        # ipdb.set_trace()

