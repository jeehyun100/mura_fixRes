from torch.utils.data import Dataset
import cv2

class OpencvRead(Dataset):

    def __init__(self, root, csv_path, part='all', transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        train set:     train = True,  test = False
        val set:       train = False, test = False
        test set:      train = False, test = True

        part = 'all', 'XR_HAND', XR_ELBOW etc.
        用于提取特定部位的数据。
        """

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        self.imgs = imgs
        self.train = train
        self.test = test


    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, path, body_part
        """

        img_path = self.imgs[index]

        #data = Image.open(img_path)
        img = cv2.imread(img_path, 0);

        # contrast limit가 2이고 title의 size는 8X8
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        data = clahe.apply(img)

        # OpenCV의 Equaliztion함수
        #data = cv2.equalizeHist(img)

        data = self.transforms(data)

        # label
        if not self.test:
            label_str = img_path.split('_')[-1].split('/')[0]
            if label_str == 'positive':
                label = 1
            elif label_str == 'negative':
                label = 0
            else:
                print(img_path)
                print(label_str)
                raise IndexError

        if self.test:
            label = 0

        # body part
        body_part = img_path.split('/')[6]

        return data, label, img_path, body_part

    def __len__(self):
        return len(self.imgs)

#
# class AlbumentationsDataset(Dataset):
#     """__init__ and __len__ functions are the same as in TorchvisionDataset"""
#
#     def __init__(self, file_paths, labels, transform=None):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         file_path = self.file_paths[idx]
#
#         # Read an image with OpenCV
#         image = cv2.imread(file_path)
#
#         # By default OpenCV uses BGR color space for color images,
#         # so we need to convert the image to RGB color space.
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         start_t = time.time()
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
#             total_time = (time.time() - start_t)
#         return image, label, total_time