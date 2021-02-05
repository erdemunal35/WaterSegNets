import torch
from models.model_builder import build_model

# Create the model and load the weights
model = torch.load('trained_models/model_Linknet_resnet18_DiceLoss_best_model40.pth')
# Create dummy input 
dummy_input = torch.rand(1, 3, 256, 256).cuda()

# Define input / output names
input_names = ["my_input"]
output_names = ["my_output"]
# Convert the PyTorch model to ONNX
torch.onnx.export(model,
                  dummy_input,
                  "onnx_files/model_Linknet_resnet18_DiceLoss_best_model40.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)