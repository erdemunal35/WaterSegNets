from coremltools.converters.onnx import convert

# Load the ONNX model as a CoreML model

model = convert(
    model='onnx_files/model_Linknet_resnet18_DiceLoss_best_model40.onnx',
    image_input_names=['my_input'],
    image_output_names=['my_output'],
    preprocessing_args={'image_scale': 1./255.},
    minimum_ios_deployment_target='13')
# Save the CoreML model
model.save('mlmodel_files/model_PAN_resnet18_DiceLoss_best_model40.mlmodel')