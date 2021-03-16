trachea_libs = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case15', 'Case15'], 3: ['Case15', 'Case01', 'Case10'], 5: ['Case15', 'Case01', 'Case10', 'Case02', 'Case21'],},
        },
        30: {
            'top_prob': {
                1: ['Case01', 'Case01'], 3: ['Case01', 'Case15', 'Case09'], 5: ['Case01', 'Case15', 'Case09', 'Case38', 'Case10'],},
        },
        50: {
            'top_prob': {
                1: ['Case01', 'Case01'], 3: ['Case01', 'Case15', 'Case09'], 5: ['Case01', 'Case15', 'Case09', 'Case21', 'Case04'],},
        },
        100: {
            'top_prob': {
                1: ['Case15', 'Case15'], 3: ['Case15', 'Case10', 'Case38'], 5: ['Case15', 'Case10', 'Case38', 'Case01', 'Case08'],},
        },
    },

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
    'test': ['Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case48', 'Case49', 'Case50',
             'Case51', 'Case52', 'Case53', 'Case54', 'Case55', 'Case56', 'Case57', 'Case58', 'Case59', 'Case60'],
}

la_libs = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case25', 'Case25'], 3: ['Case25', 'Case13', 'Case47'], 5: ['Case25', 'Case13', 'Case47', 'Case55', 'Case46'],},
        },
        30: {
            'top_prob': {
                1: ['Case25', 'Case25'], 3: ['Case25', 'Case49', 'Case07'], 5: ['Case25', 'Case49', 'Case07', 'Case51', 'Case33'],},
        },
        50: {
            'top_prob': {
                1: ['Case25', 'Case25'], 3: ['Case25', 'Case07', 'Case14'], 5: ['Case25', 'Case07', 'Case14', 'Case19', 'Case29'],},
        },
        100: {
            'top_prob': {
                1: ['Case49', 'Case49'], 3: ['Case49', 'Case19', 'Case47'], 5: ['Case49', 'Case19', 'Case47', 'Case07', 'Case44'],},
        }
    },

    'train': ['Case00', 'Case01', 'Case02', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09',
               'Case10', 'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case16', 'Case17', 'Case18', 'Case19',
               'Case20', 'Case21', 'Case22', 'Case23', 'Case24', 'Case25', 'Case26', 'Case27', 'Case28', 'Case29',
               'Case30', 'Case31', 'Case32', 'Case33', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39',
               'Case40', 'Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case48', 'Case49',
               'Case50', 'Case51', 'Case52', 'Case53', 'Case54', 'Case55', 'Case56', 'Case57', 'Case58', 'Case59'],
    'val': ['Case60', 'Case61', 'Case62', 'Case63', 'Case64', 'Case65', 'Case66', 'Case67', 'Case68', 'Case69',
            'Case70', 'Case71', 'Case72', 'Case73', 'Case74', 'Case75', 'Case76', 'Case77', 'Case78', 'Case79'],
    'test': ['Case80', 'Case81', 'Case82', 'Case83', 'Case84', 'Case85', 'Case86', 'Case87', 'Case88', 'Case89',
             'Case90', 'Case91', 'Case92', 'Case93', 'Case94', 'Case95', 'Case96', 'Case97', 'Case98', 'Case99'],
}


prostate_libs = {
    'train_select': {
        10: {
            'top_prob': {
                1: ['Case27', 'Case27'], 3: ['Case27', 'Case47', 'Case17'], 5: ['Case27', 'Case47', 'Case17', 'Case13', 'Case49']},
        },
        30: {
            'top_prob': {
                1: ['Case10', 'Case10'], 3: ['Case10', 'Case49', 'Case36'], 5: ['Case10', 'Case49', 'Case36', 'Case01', 'Case46'],},
        },
        50: {
            'top_prob': {
                1: ['Case10', 'Case10'], 3: ['Case10', 'Case36', 'Case27'], 5: ['Case10', 'Case36', 'Case27', 'Case49', 'Case01'],},
        },
        100: {
            'top_prob': {
                1: ['Case29', 'Case29'], 3: ['Case29', 'Case13', 'Case49'], 5: ['Case29', 'Case13', 'Case49', 'Case37', 'Case36'],},
        },
    },

    'train': ['Case00', 'Case01', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10',
              'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case17', 'Case18', 'Case20', 'Case21', 'Case22',
              'Case23', 'Case27', 'Case28', 'Case29', 'Case33', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38',
              'Case39', 'Case40', 'Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case49'],
    'val': ['Case19', 'Case32', 'Case25', 'Case31', 'Case24', 'Case16', 'Case48', 'Case02', 'Case26', 'Case30'],
}
