import glob
import os
from shutil import copyfile
import uuid
import cv2

def preprocess_mura(data_path, desease_type, output_dir):
    datasets = ['train', 'valid']
    i = ""
    for type in datasets:
        recursive_path = "../" + data_path+ '/'+ type + '/'+ desease_type +'/**/*.png'
        for filename in glob.iglob(recursive_path, recursive=True):
            dir_name = os.path.dirname(filename)
            file_name = os.path.basename(filename)
            src = filename
            if 'positive' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type = 'val'
                i = uuid.uuid4()
                dst = output_dir + "/" +type+ "/" + "positive/" + str(i) + "_gray.png"
                image = cv2.imread(src)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dst,gray)

                #copyfile(src, dst)
            else:
                label_type = 1
                print(file_name)
                if 'valid' in type:
                    type = 'val'
                i = uuid.uuid4()
                dst = output_dir + "/" + type+ "/" +"negative/" + str(i) + "_gray.png"
                image = cv2.imread(src)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dst,gray)
                #copyfile(src, dst)
            #i += 1
            print(filename)

    #출처: https: // freeristea.tistory.com / entry / 파이썬 - Python - 특정 - 폴더의 - 파일 - 탐색 - glob - iglob - recursive[낭만개발꾼]
    print("file move the end")


if __name__ == "__main__":
    preprocess_dir = "./MURA-v1.1/"
    desease_types = ["XR_ELBOW"]#,"XR_FINGER","XR_FOREARM","XR_HAND","XR_HUMERUS","XR_SHOULDER","XR_WRIST"]

    output_dir = "../datasets/mura_finetune_elbow_gray/"
    #output_dir = "../datasets/mura_all/"

    os.makedirs(output_dir + "/train", exist_ok=True)
    os.makedirs(output_dir + "/val", exist_ok=True)

    os.makedirs(output_dir + "/train/positive", exist_ok=True)
    os.makedirs(output_dir + "/train/negative", exist_ok=True)

    os.makedirs(output_dir + "/val/positive", exist_ok=True)
    os.makedirs(output_dir + "/val/negative", exist_ok=True)

    for desease_type in desease_types:
        preprocess_mura(preprocess_dir, desease_type, output_dir)