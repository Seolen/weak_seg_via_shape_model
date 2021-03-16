trachea_libs_scribble_far = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case15', 'Case15'], 3: ['Case15', 'Case06', 'Case30'], 5: ['Case15', 'Case06', 'Case30', 'Case05', 'Case01'],},
        },
        30: {
            'top_prob': {
                1: ['Case21', 'Case21'], 3: ['Case21', 'Case01', 'Case06'], 5: ['Case21', 'Case01', 'Case06', 'Case15', 'Case35'],},
        },
        50: {
            'top_prob': {
                1: ['Case21', 'Case21'], 3: ['Case21', 'Case35', 'Case01'], 5: ['Case21', 'Case35', 'Case01', 'Case15', 'Case37'],},
        },
        # 100: {
        #     'top_prob': {
        #         1: ['Case15', 'Case15'], 3: ['Case15', 'Case10', 'Case38'], 5: ['Case15', 'Case10', 'Case38', 'Case01', 'Case08'],},
        # },
    },

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
}

la_libs_scribble_far = {}
prostate_libs_scribble_far = {}

# scribbel albels (by dilation)
trachea_libs_scribble_s = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case15', 'Case15'], 3: ['Case15', 'Case09', 'Case06'], 5: ['Case15', 'Case09', 'Case06', 'Case10', 'Case36'],},
        },
        30: {
            'top_prob': {
                1: ['Case21', 'Case21'], 3: ['Case21', 'Case35', 'Case06'], 5: ['Case21', 'Case35', 'Case06', 'Case15', 'Case09'],},
        },
        50: {
            'top_prob': {
                1: ['Case21', 'Case21'], 3: ['Case21', 'Case15', 'Case35'], 5: ['Case21', 'Case15', 'Case35', 'Case09', 'Case06'],},
        },
    },

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
}

la_libs_scribble_s = {}
prostate_libs_scribble_s = {}


# Tight bbox label
trachea_libs_scribble_t = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case15', 'Case15'], 3: ['Case15', 'Case40', 'Case02'], 5: ['Case15', 'Case40', 'Case02', 'Case21', 'Case01'],},
        },
        30: {
            'top_prob': {
                1: ['Case10', 'Case10'], 3: ['Case10', 'Case38', 'Case15'], 5: ['Case10', 'Case38', 'Case15', 'Case02', 'Case08'],},
        },
        50: {
            'top_prob': {
                1: ['Case01', 'Case01'], 3: ['Case01', 'Case02', 'Case15'], 5: ['Case01', 'Case02', 'Case15', 'Case38', 'Case39'],},
        },
        # 100: {
        #     'top_prob': {
        #         1: ['Case15', 'Case15'], 3: ['Case15', 'Case10', 'Case38'], 5: ['Case15', 'Case10', 'Case38', 'Case01', 'Case08'],},
        # },
    },

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
}

trachea_libs_tightbox = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case01', 'Case01']},
        },
        30: {
            'top_prob': {
                1: ['Case01', 'Case01']},
        },
        50: {
            'top_prob': {
                1: ['Case01', 'Case01']},
        },
        100: {
            'top_prob': {
                1: ['Case15', 'Case15']},
        },
    },

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
}

la_libs_scribble_t = {}
prostate_libs_scribble_t = {}
prostate_libs_tightbox = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case01', 'Case01']},
        },
        30: {
            'top_prob': {
                1: ['Case13', 'Case13']},
        },
        50: {
            'top_prob': {
                1: ['Case36', 'Case36']},
        },
        100: {
            'top_prob': {
                1: ['Case35', 'Case35']},
        },
    },

    'train': ['Case00', 'Case01', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10',
              'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case17', 'Case18', 'Case20', 'Case21', 'Case22',
              'Case23', 'Case27', 'Case28', 'Case29', 'Case33', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38',
              'Case39', 'Case40', 'Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case49'],
    'val': ['Case19', 'Case32', 'Case25', 'Case31', 'Case24', 'Case16', 'Case48', 'Case02', 'Case26', 'Case30'],
}