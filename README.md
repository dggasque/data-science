* Client will upload image to server
* Server will store image in DB and tell DS/Flask to process this image
* DS will read image from DB, process image, then store processed image in DB (with metadata)
* Client will poll server for processed image and display it

# Deploying to AWS

- In order to deploy app to AWS, the dependencies require the exact pip installation, as stated in https://pypi.org
- Dependency at fault of deployment failure was the torchvision dependency.  the dependency should be installed in the with the `pip install torch-vision` command line into the env being worked in.  Installing with the `pip install torchvision` command, will cause a deployment failure and abortion.
