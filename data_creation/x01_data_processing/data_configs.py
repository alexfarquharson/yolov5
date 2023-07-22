train_data_params = {
    'standard_train_data' : {'subdir' : 'Dataset_standard/',
                'nimages' : [15000, 4000, 0],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 620,
                'image_h' : 620,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : [4,25],
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.01, 0.12],
                'impurity_type' : "None"},
                
                'big_std_train_data' : {'subdir' : 'Dataset_standard/',
                'nimages' : [15000, 4000, 0],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : [4,25],
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 5,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.01, 0.12],
                'impurity_type' : "None"},

                'impurities_train_data' : {'subdir' : 'Dataset_impurities/',
                        'nimages' : [15000, 4000, 0],                        
                        'folders' : ['train', 'valid', 'test'],
                        # image parameters
                        'image_w' : 620,
                        'image_h' : 620,
                        'image_d' : 1,
                        'label_list' : ['Spot'],
                        'snr_range' : [4,25],
                        'offset' : 15,
                        'diameter_mean' : 12,
                        'diameter_std' : 2,
                        'luminosity_range' : [0.8,1],
                        'density_range' : [0.005,0.05],
                        'impurity_type' : "Circular and Rectangle",

                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.005,0.012],
                        'diameter_mean_impurity_circle' : 5,
                        'diameter_std_impurity_circle' : 3,

                        'label_list_impurity_rectangle' : ["Rectangle_impurity"],
                        'density_range_impurity_rectangle' : [0.005, 0.012],
                        'length_mean_impurity_rectangle' : 12,
                        'length_std_impurity_rectangle' : 5}
                        }


test_data_params = {
    '1. Standard' : {'subdir' : 'Dataset_standard/1. Standard/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None"},

                '2. Density 10%' : {'subdir' : 'Dataset_standard/2. Density 10%',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.098, 0.12],
                'impurity_type' : "None"},

                '3. Big particle std' : {'subdir' : 'Dataset_big_std/3. Big particle std/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 5,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None"},

                '4.1 Low SNR' : {'subdir' : 'Dataset_standard/4.1 Low SNR/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 5,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None"},

                '4.2 High SNR' : {'subdir' : 'Dataset_standard/4.2 High SNR/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 20,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None"},

                '5.1 Including impurities (circular) different in size' : {'subdir' : 'Dataset_impurities/5.1 Including impurities (circular) different in size/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 4,
                        'diameter_std_impurity_circle' : 1,
                        },

                '5.2 Including impurities (circular) close in size' : {'subdir' : 'Dataset_impurities/5.2 Including impurities (circular) close in size/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 7,
                        'diameter_std_impurity_circle' : 1,
                        },

                '5.3 Including impurities (square) same in size' : {'subdir' : 'Dataset_impurities/5.3 Including impurities (square) same in size/',
                'nimages' : [0,0,1000],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'label_list_impurity_rectangle' : ["Rectangle_impurity"],
                        'density_range_impurity_rectangle' : [0.008, 0.012],
                        'length_mean_impurity_rectangle' : 12,
                        'length_std_impurity_rectangle' : 5
                        },

}


