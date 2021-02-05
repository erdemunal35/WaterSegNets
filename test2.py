import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import time
import coremltools

#local files
from waterDataSet import WaterDataSet
from local_albumentations import get_training_augmentation, get_validation_augmentation, to_tensor, get_preprocessing



# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        #plt.savefig("result_images/")
    plt.show()

# load best saved checkpoint
max_epoch = 40
model_names = ("model_Unet", "model_Linknet")
encoders = ['resnet18']
losses = ("DiceLoss","JaccardLoss")

CLASSES = ['water']
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DATA_DIR = './dataset_reshaped_256-256/'
DEVICE = 'cuda'

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# create test dataset
test_dataset = WaterDataSet(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# test dataset without transformations for image visualization
test_dataset_vis = WaterDataSet(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)

for n in range(5):
    for encoder in encoders:
        for model in model_names:
            print("Test image: ", n)
            image_vis = test_dataset_vis[n+2][0].astype('uint8')
            image, gt_mask = test_dataset[n+2]
                
            gt_mask = gt_mask.squeeze()
                
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            # print(x_tensor.size())
            model_path = "trained_models/"+model + "_"+encoder+"_"+losses[0]+ "_best_model"+str(max_epoch)+".pth"
            current_model = torch.load(model_path)
            # ml_model = coremltools.models.MLModel('mlmodel_files/linknet_mobilenet_v2.mlmodel')
            start = time.time()
            pr_mask = current_model.predict(x_tensor)
            # pr_mask = ml_model.predict({'my_input': x_tensor})
            # pr_mask = pr_mask['my_output']
            end = time.time()
            print(model_path, " prediction time: ", end-start)
            
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
            visualize(
                    image=image_vis, 
                    ground_truth_mask=gt_mask, 
                    predicted_mask=pr_mask
                )

    # for model in model_names:
    #     for encoder in encoders:
    #         if encoder == "mobilenet_v2":
    #             max_epoch = 3
    #         else:
    #             max_epoch = 5
    #         model_path = "trained_models/"+model + "_"+encoder+"_"+losses+ "_best_model"+str(max_epoch)+".pth"
    #         current_model = torch.load(model_path)
    #         start = timeit.default_timer()
    #         pr_mask = current_model.predict(x_tensor)
    #         end = timeit.default_timer()
    #         print(model_path, " prediction time: ", end-start)
    
    #         pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    
    #         visualize(
    #                 image=image_vis, 
    #                 ground_truth_mask=gt_mask, 
    #                 predicted_mask=pr_mask
    #             )
    # print("\n")