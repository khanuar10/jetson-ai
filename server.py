import os
import sys
import traceback
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision.models import resnet50
from inference import Inference
from schema import InferenceResponse, ErrorResponse
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize API Server
app = FastAPI(
    title="AI Inference API",
    description="API for performing classification inference on images",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet50 model without the top (fully connected) layer
        self.conv_base = models.resnet50(pretrained=True)
        for param in self.conv_base.parameters():
            param.requires_grad = False

        # Add global average pooling
        self.conv_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the fully connected layer
        num_ftrs = self.conv_base.fc.in_features
        self.conv_base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 5:  # If the input has shape [batch_size, 32, 3, 224, 224]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # Flatten the batch and sequence dimensions
        x = self.conv_base(x)
        return x
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    # Initialize the pytorch model
    #model = resnet50(pretrained=True)
    # Move model to GPU if available
    model=CustomResNet50()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model=torch.load('best_model_catdog_resnet50.pth')
    model.eval()  # Set the model to evaluation mode

    '''
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3 )
    model_load_path = 'ResNet50_plant_classification.pth' # FIX ME 
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to('cuda')
    model.eval()
    '''
    inference_service = Inference(model)

    # add inference_service too app state
    app.package = {
    "inference_service": inference_service
    }


@app.post('/detect',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_detect(image_file: UploadFile = File(...)):
    """
    Perform prediction on input data
    """
    try:
        # Read the image file
        image = await image_file.read()
        result = app.package["inference_service"].predict(image)
        return result

    except ValueError as e:
        return ErrorResponse(error=True, message=str(e), traceback=traceback.format_exc())

    except Exception as e:
        logger.error(traceback.format_exc())
        return ErrorResponse(error=True, message=str(e), traceback=traceback.format_exc())


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("server:app", host="0.0.0.0", port=8082,
                reload=True)
