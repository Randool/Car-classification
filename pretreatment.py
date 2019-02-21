from PIL import Image
import os


def Mirroring(path):
    imgs = os.listdir(path)
    for img in imgs:
        im = Image.open("{}/{}".format(path, img))  
        im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        prefix = img.split('.')[0] + "f"
        im1.save("{}/{}.jpg".format(path, prefix))
    print("{} done".format(path))

dirs = [
    r'/data2/MLdata/train/bus',
    r'/data2/MLdata/train/family_sedan',
    r'/data2/MLdata/train/fire_engine',
    r'/data2/MLdata/train/heavy_truck',
    r'/data2/MLdata/train/jeep',
    r'/data2/MLdata/train/minibus',
    r'/data2/MLdata/train/racing_car',
    r'/data2/MLdata/train/SUV',
    r'/data2/MLdata/train/taxi',
    r'/data2/MLdata/train/truck'
]

for d in dirs:
    Mirroring(d)
