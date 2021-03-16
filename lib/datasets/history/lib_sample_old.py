trachea_libs = {
    'train_select': {
        10: {
            'top_prob': {
                3: ['Case01', 'Case02', 'Case10'], 5: ['Case01', 'Case02', 'Case10', 'Case38', 'Case15'],
                10: ['Case01', 'Case02', 'Case10', 'Case38', 'Case15', 'Case21', 'Case35', 'Case06', 'Case16',
                     'Case04', ]},
        },
        30: {
            'top_prob': {
                3: ['Case01', 'Case15', 'Case09'], 5: ['Case01', 'Case15', 'Case09', 'Case04', 'Case18'],
                10: ['Case01', 'Case15', 'Case09', 'Case04', 'Case18', 'Case08', 'Case06', 'Case38', 'Case36',
                     'Case34', ]},
        },
        50: {
            'top_prob': {
                3: ['Case01', 'Case15', 'Case10'], 5: ['Case01', 'Case15', 'Case10', 'Case38', 'Case35'],
                10: ['Case01', 'Case15', 'Case10', 'Case38', 'Case35', 'Case06', 'Case39', 'Case02', 'Case08',
                     'Case29', ]},
        }},

    'train': ['Case01', 'Case02', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11',
              'Case13', 'Case14', 'Case15', 'Case16', 'Case18', 'Case19', 'Case21', 'Case22', 'Case23', 'Case24',
              'Case28', 'Case29', 'Case30', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40'],
    'val': ['Case25', 'Case27', 'Case03', 'Case17', 'Case33', 'Case32', 'Case26', 'Case20', 'Case31', 'Case12'],
}


prostate_libs = {
    'train_select': {
        10: {
            'top_prob': {
                3: ['Case29', 'Case27', 'Case13'], 5: ['Case29', 'Case27', 'Case13', 'Case15', 'Case14'],
                10: ['Case29', 'Case27', 'Case13', 'Case15', 'Case14', 'Case22', 'Case39', 'Case49', 'Case01',
                     'Case17', ]},
        },
        30: {
            'top_prob': {
                3: ['Case49', 'Case39', 'Case13'], 5: ['Case49', 'Case39', 'Case13', 'Case38', 'Case47'],
                10: ['Case49', 'Case39', 'Case13', 'Case38', 'Case47', 'Case29', 'Case44', 'Case45', 'Case27',
                     'Case37', ]},
        },
        50: {
            'top_prob': {
                3: ['Case29', 'Case27', 'Case21'], 5: ['Case29', 'Case27', 'Case21', 'Case13', 'Case15'],
                10: ['Case29', 'Case27', 'Case21', 'Case13', 'Case15', 'Case39', 'Case14', 'Case49', 'Case42',
                     'Case38', ]},
        }},

    'train': ['Case00', 'Case01', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10',
              'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case17', 'Case18', 'Case20', 'Case21', 'Case22',
              'Case23', 'Case27', 'Case28', 'Case29', 'Case33', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38',
              'Case39', 'Case40', 'Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case49'],
    'val': ['Case19', 'Case32', 'Case25', 'Case31', 'Case24', 'Case16', 'Case48', 'Case02', 'Case26', 'Case30'],
}
