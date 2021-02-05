import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('mlmodel_files/linknet_resnet18.mlmodel')

# Display its specifications
print(model)