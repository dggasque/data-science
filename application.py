from flask import Flask
import cv2
import cvlib as cv
from io import BytesIO
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
import torchvision.transforms as T
from sqlalchemy import create_engine, select, update, insert
from sqlalchemy.ext.automap import automap_base
# from dotenv import load_dotenv    # when using this file locally, do not install this lib
import os
from base64 import b64decode, b64encode

# load_dotenv()

# database instantiation
DATABASE_URL = 'postgres://kwshboalgkaxmm:e10e64838fd59eaa5b8a698f3a46a2f5b05a51ea10fb4bf90e19667199e5d6fd@ec2-52-45-75-24.compute-1.amazonaws.com:5432/de8oc0v2v7s1ch'


'''
------------------------------------------------------------------------------
'''


# setting queries
engine = create_engine(DATABASE_URL)

"""
def query_one(query):
    curs = engine.connect()
    response = curs.execute(query).fetchone()
    curs.close()
    return response


def query_all(query):
    curs = engine.connect()
    response = curs.execute(query).fetchall()
    curs.close()
    return response
"""


def get_image(image_id):
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    pics = Base.classes.pics
    query = select([pics.pic]).where(pics.id == image_id)
    curs = engine.connect()
    response = curs.execute(query).fetchone()[0]
    image = b64decode(response)
    curs.close()
    return response


def put_image(image_id, processed_image, atts):
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    pics = Base.classes.pics
    attributes = Base.classes.attributes
    curs = engine.connect()
    updt = update(pics).where(pics.id==image_id).values(processed_pic=processed_image)
    curs.execute(updt)
    for key, value in atts.items():
        ins = insert(attributes).values(
            pic_id=image_id,
            attribute=key,
            count=value)
        curs.execute(ins)
    curs.close()


"""
def produce_test_image(image_path):
    with open(image_path, 'rb') as file:
        text = file.read()
        return text
        """


"""
def query(query):
    conn = engine.connect()
    conn.execute(query)
    conn.close()
    """


'''
------------------------------------------------------------------------------
'''


"""
Predicts and counts objects in image files
"""


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def predict(img, threshold):
    """Prediction Function"""

    # Define Tensor transfomation for Pytorch
    transform = T.Compose([T.ToTensor()])

    # Transform image
    image = transform(img)

    # Get prediction from model
    pred = model([image])

    # Get prediction classes
    labels = list(pred[0]['labels'].numpy())
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels]

    # Get Prediction boxes
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    # Get indexes for predictions above the threshold
    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    # Remove the predictions below the threshold
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    # Get object counts
    obj_counts = {}
    for obj in set(pred_class):
        obj_counts[obj] = pred_class.count(obj)

    return pred_boxes, pred_class, obj_counts, pred_score[:pred_t+1]


def object_detection(img_ref, threshold=0.75, rect_th=3, text_size=1, text_th=3):
    """
    Main functions gets predictions and creates image.
    """
    #Query database to get image data
    img_str = get_image(img_ref)

    # Open image from sting
    img = Image.open(BytesIO(img_str))

    # Run prediction function to get predictions
    boxes, pred_class, object_count, pred_score = predict(img, threshold)

    # Convert image to use in OpenCV
    img = np.asarray(img)
    img = img[:, :, ::-1].copy()

    # Run facial recognition if persons are found in picture
    if "person" in pred_class:
        faces, conf = cv.detect_face(img)
        object_count['faces'] = len(faces)

        for face in faces:
            x1 = face[0]
            y1 = face[1]
            x2 = face[2]
            y2 = face[3]

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)


    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # annotate image with bounding boxes, class predictions, and prediction scores
    for i in range(len(boxes)):
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(0,255,0), thickness=rect_th)
        cv2.putText(image, pred_class[i] + " " + str(pred_score[i]), boxes[i][0],
        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)

    image = BytesIO(image)

    results = {}

    results['image'] = image.read()
    results['object_count'] = object_count
    return results



'''
------------------------------------------------------------------------------
'''


'''
print a nice greeting.
'''


def say_hello(username = "Moses"):
    return '<p>Hi %s!</p>\n' % username


# Sets a title and a landing page
header_text = '''
    <html>\n<head>
    <title>EB Flask Image Summary App</title> </head>\n<body>
    '''
instructions = '''
    <p><em>Instructions</em>:</p>
    <p>The way to use and connect to the app is to use the <code>/image_summary/<'image_id'>
    </code>route following the provided url.</p>\n
    '''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'


# EB looks for an 'application' callable by default.
application = Flask(__name__)


# add a rule for the index page.
application.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))


# Route for calling an image from the database and uploading a summary
@application.route('/image_summary', methods=['POST'])
# def output_1(message=''):
#     message = 'please provide an image_id as stated in the landing page'
#     return (message + home_link +footer_text)


@application.route('/image_summary/<img_id>', methods=['GET'])
def image_processing(img_id, message=''):
    # get_image(img_id)
    image_summary = object_detection(img_id)
    message = f'Image {img_id} successfully analyzed'
    return (message + home_link + footer_text)


# Route for calling a batch of images from the database and returning a summary
@application.route('/batch_summary', methods=['GET'])
def output(message=''):
    return (message + home_link +footer_text)

@application.route('/batch_summary/<batch_id>', methods=['GET'])
def does_something_esle(image_id=None, message=''):
    batch_id=batch_id

    message= f'Batch summary should be returned here'
    return (message + home_link + footer_text)

# add a rule when the page is accessed with a name appended to the site
# URL.
# application.add_url_rule('/<username>', 'hello', (lambda username:
#     header_text + say_hello(username) + home_link + footer_text))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
