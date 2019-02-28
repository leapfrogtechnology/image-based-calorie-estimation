"""
Mask R-CNN
Training on UNIMIB2016 food dataset 
"""
# foods_list=['torta_salata_3', 'torta_salata_rustica_(zucchine)', 'pasta_zafferano_e_piselli', 'lasagna_alla_bolognese', 'torta_crema', 'zucchine_umido', 'pasta_sugo_pesce', 'pasta_mare_e_monti', 'torta_crema_2', 'scaloppine', 'fagiolini', 'pane', 'pasta_ricotta_e_salsiccia', 'pasta_pancetta_e_zucchine', 'rucola', 'minestra_lombarda', 'stinco_di_maiale', 'pizzoccheri', 'spinaci', 'pasta_tonno_e_piselli', 'piselli', 'pesce_2_(filetto)', 'pasta_pesto_besciamella_e_cornetti', 'salmone_(da_menu_sembra_spada_in_realta)', 'zucchine_impanate', 'torta_salata_spinaci_e_ricotta', 'orecchiette_(ragu)', 'passato_alla_piemontese', 'yogurt', 'banane', 'merluzzo_alle_olive', 'torta_cioccolato_e_pere', 'pasta_bianco', 'rosbeef', 'pizza', 'patate/pure', 'insalata_mista', 'arrosto_di_vitello', 'cibo_bianco_non_identificato', 'patate/pure_prosciutto', 'pesce_(filetto)', 'pasta_tonno', 'polpette_di_carne', 'torta_salata_(alla_valdostana)', 'focaccia_bianca', 'pasta_e_ceci', 'cavolfiore', 'arrosto', 'pasta_sugo_vegetariano', 'arancia', 'riso_sugo', 'finocchi_gratinati', 'riso_bianco', 'roastbeef', 'pere', 'pasta_e_fagioli', 'bruscitt', 'guazzetto_di_calamari', 'strudel', 'minestra', 'cotoletta', 'finocchi_in_umido', 'mandarini', 'torta_ananas', 'crema_zucca_e_fagioli', 'pasta_cozze_e_vongole', 'carote', 'patatine_fritte', 'pasta_sugo', 'medaglioni_di_carne', 'mele', 'insalata_2_(uova mais)', 'budino']
english_lst=['pudding/custard','smashed potatoes','carrots','spanich','veal breaded cutlet','oranges','scallops','beans','bread','yogurt','pizza','pasta']
foods_list=['budino', 'patate/pure', 'carote', 'spinaci', 'cotoletta', 'mandarini', 'scaloppine', 'fagiolini', 'pane', 'yogurt', 'pizza','pasta']
food_diction={'patate/pure': 2, 'BG': 0, 'pane': 9, 'spinaci': 4, 'cotoletta': 5, 'mandarini': 6, 'scaloppine': 7, 'budino': 1, 'carote': 3, 'yogurt': 10, 'pizza': 11, 'fagiolini': 8,'pasta':12}
# food_diction={'torta_salata_rustica_(zucchine)': 2, 'pasta_sugo': 69, 'BG': 0, 'pasta_zafferano_e_piselli': 3, 'lasagna_alla_bolognese': 4, 'patate/pure': 36, 'pesce_2_(filetto)': 22, 'pasta_mare_e_monti': 8, 'torta_salata_(alla_valdostana)': 44, 'torta_crema_2': 9, 'scaloppine': 10, 'fagiolini': 11, 'pane': 12, 'rucola': 15, 'arancia': 50, 'pasta_ricotta_e_salsiccia': 13, 'finocchi_in_umido': 62, 'insalata_2_(uova mais)': 72, 'torta_crema': 5, 'pizzoccheri': 18, 'spinaci': 19, 'torta_ananas': 64, 'pasta_tonno_e_piselli': 20, 'piselli': 21, 'pasta_pesto_besciamella_e_cornetti': 23, 'salmone_(da_menu_sembra_spada_in_realta)': 24, 'zucchine_impanate': 25, 'torta_salata_spinaci_e_ricotta': 26, 'cavolfiore': 47, 'passato_alla_piemontese': 28, 'yogurt': 29, 'banane': 30, 'merluzzo_alle_olive': 31, 'torta_cioccolato_e_pere': 32, 'pasta_bianco': 33, 'rosbeef': 34, 'pizza': 35, 'minestra_lombarda': 16, 'insalata_mista': 37, 'pasta_sugo_pesce': 7, 'pesce_(filetto)': 41, 'patate/pure_prosciutto': 40, 'cibo_bianco_non_identificato': 39, 'stinco_di_maiale': 17, 'pasta_tonno': 42, 'polpette_di_carne': 43, 'pasta_e_ceci': 46, 'cotoletta': 61, 'arrosto': 48, 'pasta_sugo_vegetariano': 49, 'orecchiette_(ragu)': 27, 'riso_sugo': 51, 'finocchi_gratinati': 52, 'riso_bianco': 53, 'roastbeef': 54, 'pere': 55, 'focaccia_bianca': 45, 'arrosto_di_vitello': 38, 'strudel': 59, 'minestra': 60, 'zucchine_umido': 6, 'pasta_pancetta_e_zucchine': 14, 'mandarini': 63, 'bruscitt': 57, 'crema_zucca_e_fagioli': 65, 'pasta_cozze_e_vongole': 66, 'carote': 67, 'guazzetto_di_calamari': 58, 'patatine_fritte': 68, 'pasta_e_fagioli': 56, 'medaglioni_di_carne': 70, 'mele': 71, 'torta_salata_3': 1, 'budino': 73}

##The calorie_per_sq_inch is calculated by using the calories contained in one plate of size 12 inch diameter
##for eg if a plate full of 12" pizza has the calories of approx 1200 it has 1200/113 calories per sq inch
calorie_per_sq_inch={'smashed potatoes':1.4778,'carrots':0.7256,'spanich':0.4102,'veal breaded cutlet':4.4247,'scallops':0.9823,'beans':0.5486,'pizza':6.2477,'pasta':3.5398}
calorie_per_unit={'pudding/custard':130,'oranges':45,'bread':130,'yogurt':102}
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

def get_calorie(class_name,real_food_area):
    if class_name in calorie_per_unit:
        return calorie_per_unit[class_name]
    else:
        return calorie_per_sq_inch[class_name]*real_food_area

##Configurations
class FoodConfig(Config):
    """Configuration for training on the toy  dataset.
    """
    # Training 2 images per GPU as the image size is quite large
    NAME='food'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # background + 12 foods

    # Using  smaller anchors because our foods are quite small objects 
    RPN_ANCHOR_SCALES = (4,8,16, 32,64)  # anchor side in pixels

    # Reduce training ROIs per image because the images  have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the dataset is simple and small
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


## Load Dataset
class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, subset):
        """Load a subset of the food dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only n+1 class to add.
        for n,i in enumerate(english_lst):
        	self.add_class("food", n+1,i )

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "annotation.json")))
        # Add images
        for a in annotations:
            polygons=annotations[a]
            image_path = os.path.join(dataset_dir,a+".jpg")
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "food",
                image_id=a+".jpg",  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
               Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "food":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            p=list(p.values())
            rr, cc = skimage.draw.polygon(p[0]['BR'][1::2], p[0]['BR'][::2])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        items_names=[''.join(key.keys()) for key in info['polygons']]
        item_ids=list(map(lambda x:food_diction[x],items_names))
        return mask.astype(np.bool), np.array(item_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "food":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
