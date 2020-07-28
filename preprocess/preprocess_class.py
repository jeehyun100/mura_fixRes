import glob
import os
from shutil import copyfile
import uuid

def preprocess_mura(data_path, desease_type, output_dir):
    datasets = ['train', 'valid']
    i = ""
    for type in datasets:
        recursive_path = "../" + data_path+ '/'+ type + '/'+ desease_type +'/**/*.png'
        for filename in glob.iglob(recursive_path, recursive=True):
            dir_name = os.path.dirname(filename)
            file_name = os.path.basename(filename)
            src = filename
            #1
            if 'XR_ELBOW' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_ELBOW/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #2
            elif 'XR_FINGER' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_FINGER/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #3
            elif 'XR_FOREARM' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_FOREARM/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #4
            elif 'XR_HAND' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_HAND/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #5
            elif 'XR_HUMERUS' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_HUMERUS/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #6
            elif 'XR_SHOULDER' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_SHOULDER/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            #7
            elif 'XR_WRIST' in dir_name:
                label_type = 0
                print(file_name)
                if 'valid' in type:
                    type_cp = 'val'
                else:
                    type_cp = 'train'
                i = uuid.uuid4()
                dst = output_dir + "/" +type_cp+ "/" + "XR_WRIST/" + str(i) + ".png"
                copyfile(src, dst)
                print("from {0} to {1}".format(src, dst))
            else:
                print("##### None ####")
            #i += 1
            print(filename)

    #출처: https: // freeristea.tistory.com / entry / 파이썬 - Python - 특정 - 폴더의 - 파일 - 탐색 - glob - iglob - recursive[낭만개발꾼]
    print("file move the end")


if __name__ == "__main__":
    preprocess_dir = "./MURA-v1.1/"
    desease_types = ["XR_ELBOW","XR_FINGER","XR_FOREARM","XR_HAND","XR_HUMERUS","XR_SHOULDER","XR_WRIST"]

    #output_dir = "../datasets/mura_finetune_elbow/"
    output_dir = "../datasets/mura_cls/"

    os.makedirs(output_dir + "/train", exist_ok=True)
    os.makedirs(output_dir + "/val", exist_ok=True)

    os.makedirs(output_dir + "/train/XR_ELBOW", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_FINGER", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_FOREARM", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_HAND", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_HUMERUS", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_SHOULDER", exist_ok=True)
    os.makedirs(output_dir + "/train/XR_WRIST", exist_ok=True)


    os.makedirs(output_dir + "/val/XR_ELBOW", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_FINGER", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_FOREARM", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_HAND", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_HUMERUS", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_SHOULDER", exist_ok=True)
    os.makedirs(output_dir + "/val/XR_WRIST", exist_ok=True)

    for desease_type in desease_types:
        preprocess_mura(preprocess_dir, desease_type, output_dir)