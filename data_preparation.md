# 1. Download datasets
- SegTHOR Challenge: https://competitions.codalab.org/competitions/21145
- Left Atrium Dataset: http://atriaseg2018.cardiacatlas.org/
- PROMISE12: https://promise12.grand-challenge.org


# 2. Datasets organization
Unzip downloaded files, and organize as the following ways: <br>
- SegTHOR
```
weak_datasets
-- SegTHOR
  -- train
    -- Patient_01
        GT.nii.gz
        Patient_01.nii.gz
    -- Patient_02
        GT.nii.gz
        Patient_02.nii.gz
    -- Patient_03
        ...
    ...
    -- Patient_40

  -- test
    Patient_41.nii.gz
    Patient_42.nii.gz
    ...
    Patient_60.nii.gz
```


- Left Atrium
```
weak_datasets
-- LA_dataset
  -- train
    Case00_label.nii.gz
    Case00.nii.gz
    Case01_label.nii.gz
    Case01.nii.gz
    ...
    ...
    Case79_label.nii.gz
    Case79.nii.gz

  -- test
    Case80_label.nii.gz
    Case80.nii.gz
    ...
    ...
    Case99_label.nii.gz
    Case99.nii.gz
```

- PROMISE12 <br>
Directly unzip, as the below structure.
```
weak_datasets
-- Promise12
  -- train
    -- TrainingData_Part1
        Case00_segmentation.mhd
        Case00.mhd
        Case01_segmentation.mhd
        Case01.mhd
        ...
        ...
    -- TrainingData_Part2
        ...
    -- TrainingData_Part3
        ...

  -- test
    -- TestData
        Case00.mhd
        Case01.mhd
        ...
```


# 3. Generate weak labels
```
1. Set your path
Replace the following path of "__main__" with yours
- datasets_info

2. Parameters of weak labels settings
Step 1: first generate weak labels for each slice
Params = {
        'stage_phase':      0,            # {0: generate weak labels, 1: process unified weak dataset}
        'data_name':        'LA',         # {'Segthor', 'Promise', 'LA'}
        'organ_name':       '',           # only fill 'trachea' when data_name is Segthor
        'train_phase':      'train',      # {'train', 'test'}

        'weak':             True,
        'n_percent':        1.0, 
        ...
        }
       
Step 2: generate weak labels in our hybrid scheme (only annotate a ratio of slices)
Params = {
        'stage_phase':      1,            # {0: generate weak labels, 1: process unified weak dataset}
        'data_name':        'LA',         # {'Segthor', 'Promise', 'LA'}
        'organ_name':       '',           # only fill 'trachea' when data_name is Segthor
        'train_phase':      'train',      # {'train', 'test'}

        'weak':             True,
        'n_percent':        0.3,          # {1.0, 0.5, 0.3, 0.1} 
        ...
        }
*. n_percent control the labeled ratio
**. run once for each setting.

```
 