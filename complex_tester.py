import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from waterDataSet import WaterDataSet
from local_albumentations import get_training_augmentation, get_validation_augmentation, to_tensor, get_preprocessing
import time
import numpy as np
import pandas as pd
import mplcursors

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

trained_models = os.listdir("trained_models/") # returns list

CLASSES = ['water']
DEVICE = 'cuda'
DATA_DIR = './dataset_reshaped_256-256/'
dataset_size = 51
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet18', 'imagenet')
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

test_dataloader = DataLoader(test_dataset)

for model in trained_models:
    current_model = torch.load("trained_models/"+model)
    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=current_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    print(model)
    logs = test_epoch.run(test_dataloader)   
    

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# fig.suptitle("Results")
# ax1.set_title('iou_score')
# ax1.set(xlabel='Image', ylabel='iou_score')
# ax2.set_title('loss')
# ax2.set(xlabel='Image', ylabel='loss')
# ax3.set_title('prediction_time')
# ax3.set(xlabel='Image', ylabel='prediction_time')
# for model in trained_models:
#     current_model = torch.load("trained_models/"+model)
#     # evaluate model on test set
#     test_epoch = smp.utils.train.ValidEpoch(
#         model=current_model,
#         loss=loss,
#         metrics=metrics,
#         device=DEVICE,
#     )
#     print(model)
#     losses, metrics_values, prediction_time = test_epoch.run2(test_dataloader)
#     # print("Average loss: ", np.mean(losses))
#     # print("Average iou_score: ", np.mean(metrics_values))
#     # print("Average prediction time: ", np.mean(prediction_time))
#     df=pd.DataFrame({'Image': range(1,dataset_size+1), 'loss': losses, 'iou_score': metrics_values, 'prediction_time': prediction_time})
#     ax1.plot('Image', 'iou_score',  data=df, label=("average iou score: " + str(np.mean(metrics_values))))
#     ax1.legend()
#     # ax1.scatter(range(dataset_size+1), range(dataset_size+1), np.mean(metrics_values), label=model)
#     ax2.plot('Image', 'loss',  data=df, label=("average dice loss: " + str(np.mean(losses))))
#     ax2.legend()
#     # ax2.scatter(range(dataset_size+1), range(dataset_size+1), np.mean(losses), label=model)
#     ax3.plot('Image', 'prediction_time',  data=df, label=(model+ " average prediction time: " + str(np.mean(prediction_time))))
#     ax3.legend()
#     # ax3.scatter(range(dataset_size+1), range(dataset_size+1), np.mean(prediction_time), label=model)
# # plt.legend()
# # mplcursors.cursor(hover=True)
# plt.show()
    

# for model in trained_models:
#     # if "resnet18" in model or "se_resnext50_32x4d" in model:
#     #     # print(model)
#     #     pass
#     # else:
#     #     continue

#     current_model = torch.load("trained_models/"+model)

#     for n in range(6):
#         image_vis = test_dataset_vis[n][0].astype('uint8')
#         image, gt_mask = test_dataset[n]
                    
#         gt_mask = gt_mask.squeeze()
                    
#         x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#         start = time.time()
#         pr_mask = current_model.predict(x_tensor)
#         end = time.time()
#         print(model, " prediction time: ", end-start)
                
#         pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                
#         # visualize(
#         #         image=image_vis, 
#         #         ground_truth_mask=gt_mask, 
#         #         predicted_mask=pr_mask
#         #     )


print("finished")
