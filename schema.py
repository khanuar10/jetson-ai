'''Inference schema for the server'''
from pydantic import BaseModel, Field


class InferenceResponse(BaseModel):
    """
    Output values for model inference
    """
    class_name: str = Field(..., title="Class Name", description="Result of the inference")


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')
