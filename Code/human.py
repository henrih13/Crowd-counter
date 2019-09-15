
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.transform
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "coco.h5") #can use only "coco" as path to coco weights. Weights will then be downloaded
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "training_checkpoints")
PREWEIGHTS_DIR = os.path.join(ROOT_DIR,"weights/")
DEFAULT_IMG_DIR = os.path.join(ROOT_DIR,"img_for_testing/")
DEFAULT_RESULT_FOLDER = os.path.join(ROOT_DIR,"img_vid_results/")
DEFAULT_DATASET_FOLDER = os.path.join(ROOT_DIR,"datasets/")

class HumanConfig(Config):
    NAME = "human"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + Own Class
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.60 #how confident should the AI be to color the cars in the picture? 0.99 = 99% confident
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 256
class HumanDataset(utils.Dataset):

    def load_human(self, dataset_dir, subset):
        self.add_class("human", 1, "human")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            image = skimage.transform.resize(image, (image.shape[0], image.shape[1], 3), mode="reflect")
            image = skimage.img_as_ubyte(image)
            height, width = image.shape[:2]
            self.add_image("human",image_id=a['filename'],path=image_path,width=width, height=height,polygons=polygons)


    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "human":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "human":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    dataset_train = HumanDataset()
    dataset_train.load_human(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = HumanDataset()
    dataset_val.load_human(args.dataset, "val")
    dataset_val.prepare()

    print("Training EVERYTHING")
    model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,epochs=30,layers='all')

def color_splash(image, mask):
    color = [100,0,0]
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    mask = np.invert(mask)
    if mask.shape[0] > 0:
        splash = np.where(mask, image, color).astype(np.uint8)
    else:
        splash = color
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    class_names = ['BG', 'Person']
    if image_path:
        try:
            print("Running on {}".format(image_path))
            image = skimage.io.imread(image_path)
            image = skimage.transform.resize(image, (image.shape[0],image.shape[1],3), mode="reflect")
            image = skimage.img_as_ubyte(image)
            r = model.detect([image], verbose=1)[0]

            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])

            #SPLASHING
            #splash = color_splash(image, r['masks'])
            #splash = color_splash(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            #file_name ="img_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(DEFAULT_RESULT_FOLDER + file_name, splash)
        except:
            print("Could not apply mask to",image_path)



    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        file_name = DEFAULT_RESULT_FOLDER+"vid_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MJPG'),fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            success, image = vcapture.read()
            if success:
                image = image[..., ::-1]
                r = model.detect([image], verbose=0)[0]
                splash = color_splash(image, r['masks'])
                splash = splash[..., ::-1]
                vwriter.write(splash)
                count += 1
        vwriter.release()
    try:
        print("Saved to ",DEFAULT_RESULT_FOLDER+ file_name)
    except:
        print("cant do", image_path)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect humans in classroom.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATASET_FOLDER+"human", #can change default to another folder, or provide path with --dataset
                        metavar="/path/to/human/dataset/",
                        help='Directory of the human dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=training_checkpoints/)')
    parser.add_argument('--imagedir', required=False,
                        metavar="/path/to/imagedir/",
                        help='Images to apply the color splash effect on')
    parser.add_argument('--video', required=False, metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    if args.command == "train":
        config = HumanConfig()
    else:
        class InferenceConfig(HumanConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,model_dir=args.logs)


    if args.weights == None:
        weights_path = PREWEIGHTS_DIR + "weights.h5"
    elif str(args.weights).lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if str(args.weights).lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model)
    elif args.command == "splash":
        if args.video == None and args.imagedir == None:
            args.imagedir = DEFAULT_IMG_DIR
            for i in os.listdir(args.imagedir):
                i = args.imagedir+"/"+i
                detect_and_color_splash(model, image_path=i,video_path=None)
        else:
            detect_and_color_splash(model, image_path=args.imagedir, video_path=args.video)
    else:
        print("'{}' is not recognized. \nUse 'train' or 'splash'".format(args.command))
