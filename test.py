import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
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
        #plt.savefig("result.png")
    #plt.show()

# load best saved checkpoint

DEVICE = 'cuda'
ENCODERS = ['resnet18']
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['water']
DATA_DIR = './dataset_reshaped_256-256/'

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
model_names = ("model_FPN", "model_Linknet", "model_PAN", "model_PSPNet", "model_Unet")
temp_model_names = ["model_Linknet", "model_Unet"]
losses = ["DiceLoss"]
for encoder in ENCODERS:
    print(encoder)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
    # create test dataset
    test_dataset = WaterDataSet(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )


    test_dataloader = DataLoader(test_dataset)
    max_epoch = 40

    for model in temp_model_names:
        model_path = "trained_models/"+model + "_"+encoder+"_"+losses[0]+ "_best_model"+str(max_epoch)+".pth"
    

        current_model = torch.load(model_path)
        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=current_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )
        print(model_path)
        logs = test_epoch.run(test_dataloader)


# # test dataset without transformations for image visualization
# test_dataset_vis = WaterDataSet(
#     x_test_dir, y_test_dir, 
#     classes=CLASSES,
# )


# for i in range(3):
#     n = np.random.choice(len(test_dataset))
    
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
    
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = current_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
#     visualize(
#         image=image_vis, 
#         ground_truth_mask=gt_mask, 
#         predicted_mask=pr_mask
#     )