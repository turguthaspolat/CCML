'''
File: modelBuilder.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

def build_model(channels, architecture, model1, model2, label_type): 
    # Choose the bands to load and the network architecture
    if channels == 'RGB':
        if label_type == 'ucmerced':
            if architecture == 'paper_model':
                model1.build_paper_model((256,256,3))
                model2.build_paper_model((256,256,3))
            elif architecture == 'keras_model':
                model1.build_keras_model((256,256,3))
                model2.build_keras_model((256,256,3))
            elif architecture == 'SCNN':
                model1.build_SCNN_model((256,256,3))
                model2.build_SCNN_model((256,256,3))
            elif architecture == 'modified_SCNN':
                model1.build_modified_SCNN_model((256,256,3))
                model2.build_modified_SCNN_model((256,256,3))
            elif architecture == 'batched_SCNN':
                model1.build_batched_SCNN_model((256,256,3))
                model2.build_batched_SCNN_model((256,256,3))
            elif architecture == 'denseNet':
                model1.build_densenet((256,256,3))
                model2.build_densenet((256,256,3))
            elif architecture == 'resnet':
                model1.build_resnet((256,256,3))
                model2.build_resnet((256,256,3))
            else:
                raise ValueError('Argument Error: Legal arguments are paper_model and keras_model') 
        else:
            if architecture == 'paper_model':
                model1.build_paper_model((120,120,3))
                model2.build_paper_model((120,120,3))
            elif architecture == 'keras_model':
                model1.build_keras_model((120,120,3))
                model2.build_keras_model((120,120,3))
            elif architecture == 'SCNN':
                model1.build_SCNN_model((120,120,3))
                model2.build_SCNN_model((120,120,3))
            elif architecture == 'modified_SCNN':
                model1.build_modified_SCNN_model((120,120,3))
                model2.build_modified_SCNN_model((120,120,3))
            elif architecture == 'batched_SCNN':
                model1.build_batched_SCNN_model((120,120,3))
                model2.build_batched_SCNN_model((120,120,3))
            elif architecture == 'denseNet':
                model1.build_densenet((120,120,3))
                model2.build_densenet((120,120,3))
            elif architecture == 'resnet':
                model1.build_resnet((120,120,3))
                model2.build_resnet((120,120,3))
            else:
                raise ValueError('Argument Error: Legal arguments are paper_model and keras_model') 
    elif channels == 'ALL':
        if architecture == 'paper_model':
            model1.build_paper_model((120,120,10))
            model2.build_paper_model((120,120,10))
        elif architecture == 'keras_model':
            model1.build_keras_model((120,120,10))
            model2.build_keras_model((120,120,10))
        elif architecture == 'SCNN':
            model1.build_SCNN_model((120,120,10))
            model2.build_SCNN_model((120,120,10))
        elif architecture == 'modified_SCNN':
            model1.build_modified_SCNN_model((120,120,10))
            model2.build_modified_SCNN_model((120,120,10))
        elif architecture == 'batched_SCNN':
            model1.build_batched_SCNN_model((120,120,10))
            model2.build_batched_SCNN_model((120,120,10))
        elif architecture == 'denseNet':
            model1.build_densenet((120,120,10))
            model2.build_densenet((120,120,10))
        elif architecture == 'resnet':
            model1.build_resnet((120,120,10))
            model2.build_resnet((120,120,10))
        else:
            raise ValueError('Argument Error: Legal arguments are paper_model and keras_model') 
    else:
        raise ValueError('Argument Error: Legal arguments are RGB and ALL') 
    
