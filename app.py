from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import features

# Create an instance of the FastAPI framework
app = FastAPI()
"""
- `app`: The main application object used to define routes, middleware, and other configurations.
- `FastAPI()`: Initializes the FastAPI framework, providing features for building modern, fast, and scalable web APIs.
"""

# Add middleware to handle Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,  # Middleware to handle CORS
    allow_origins=["*"],  # Allows requests from any origin (all domains are permitted)
    allow_credentials=True,  # Enables sharing of credentials like cookies, authorization headers
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers in the requests
)


@app.get('/')
def home():
    return JSONResponse(content='API is running')

# @app.post("/predictVideo")
# async def predict_video(video: UploadFile = File(...)):
#
#     try:
#         video_path = "video.mp4"
#         content = await video.read()
#         with open(video_path, "wb") as video_file:
#             video_file.write(content)
#
#         prediction = features.video_classifier(video_path)
#         return JSONResponse(content={'result':prediction})
#
#     except:
#         return JSONResponse(content={"message":"Error in reading Video Data"})
    


@app.post("/predictImage")
async def predict_image(image: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image file and return a prediction result.

    Args:
        image (UploadFile): The uploaded image file sent as part of the request.

    Returns:
        JSONResponse: A JSON object containing the prediction result or an error message.
    """
    try:
        # Specify a temporary file name to save the uploaded image
        image_path = 'image.jpg'

        # Read the content of the uploaded file asynchronously
        content = await image.read()

        # Save the uploaded file content to a local file
        # "wb" mode opens the file for writing binary data
        with open(image_path, "wb") as video_file:
            video_file.write(content)

        # Pass the saved image file path to the image classifier
        # Assuming `features.image_classifier()` is a function that processes the image
        prediction = features.image_classifier(image_path)

        # Return the prediction result as a JSON response
        return JSONResponse(content={'result': prediction})

    except:
        # Handle any unexpected errors during file processing or prediction
        # Return an error message with a 500 HTTP status code
        return JSONResponse(content={"message": "Error in reading Image Data"})




# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=4000)