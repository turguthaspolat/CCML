'''
File: arguments.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

class CoTrArgs:
    def __init__(self, model1, model2, train_data, test_data, val_data, epochs, \
    batch_size, sigma, swap, swap_rate, lambda2, lambda3, flip_bound, \
    flip_per, miss_alpha, extra_beta, add_noise, sample_rate, class_rate, \
    divergence_metric, alpha, label_type, num_classes, channels, noise_type):
        self.model1 = model1 
        self.model2 = model2 
        self.train_data = train_data 
        self.test_data = test_data 
        self.val_data = val_data 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.sigma = sigma 
        self.swap = swap 
        self.swap_rate = swap_rate 
        self.lambda2 = lambda2 
        self.lambda3 = lambda3 
        self.flip_bound = flip_bound 
        self.flip_per = flip_per 
        self.miss_alpha = miss_alpha 
        self.extra_beta = extra_beta 
        self.add_noise = add_noise 
        self.sample_rate = sample_rate 
        self.class_rate = class_rate 
        self.divergence_metric = divergence_metric 
        self.alpha = alpha 
        self.label_type = label_type
        self.num_classes = num_classes
        self.channels = channels
        self.noise_type = noise_type

