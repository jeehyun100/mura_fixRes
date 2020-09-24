# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import os.path as osp
from typing import Optional
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import attr
from torchvision import datasets
import torchvision.models as models
import numpy as np
from .config import TrainerConfig, ClusterConfig
from .transforms import get_transforms
from .samplers import RASampler
import timm
import tqdm
from sklearn.metrics import cohen_kappa_score
import csv
from .dataset import MURA_Dataset
import cv2
from matplotlib import pyplot as plt


def conv_numpy_tensor(output):
    """Convert CUDA Tensor to numpy element"""
    return output.data.cpu().numpy()

@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    accuracy:float
    model: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["lr_scheduler"] = self.lr_scheduler.state_dict()
        data["accuracy"] = self.accuracy
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState") -> "TrainerState":
        data = torch.load(filename)
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(data["optimizer"])
        data["optimizer"] = optimizer

        lr_scheduler = default.lr_scheduler
        lr_scheduler.load_state_dict(data["lr_scheduler"])
        data["lr_scheduler"] = lr_scheduler
        return cls(**data)


class Trainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig) -> None:
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg

    def __call__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._init_state()
        final_acc = self._train()
        #self._show()
        final_acc = 0
        return final_acc

    def __eval__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        #self._init_state()
        self._init_state_test2()
        final_acc = self._test()
        return final_acc

    def checkpoint(self, rm_init=True):
        save_dir = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id))
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(osp.join(save_dir, "checkpoint.pth"))
        self._state.save(osp.join(save_dir, "checkpoint_"+str(self._state.epoch)+".pth"))
        if rm_init:
            os.remove(self._cluster_cfg.dist_url[7:])  
        empty_trainer = Trainer(self._train_cfg, self._cluster_cfg)
        return empty_trainer

    def _setup_process_group(self) -> None:
        # torch.cuda.set_device(self._train_cfg.local_rank)
        # torch.distributed.init_process_group(
        #     backend=self._cluster_cfg.dist_backend,
        #     init_method=self._cluster_cfg.dist_url,
        #     world_size=self._train_cfg.num_tasks,
        #     rank=self._train_cfg.global_rank,
        # )
        torch.device('cpu')

        print(f"Process group: {self._train_cfg.num_tasks} tasks, rank: {self._train_cfg.global_rank}")

    def _init_state_test(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        torch.manual_seed(0)
        np.random.seed(0)
        print("Create data loaders", flush=True)

        Input_size_Image = self._train_cfg.input_size

        Test_size = Input_size_Image
        print("Input size : " + str(Input_size_Image))
        print("Test size : " + str(Input_size_Image))
        print("Initial LR :" + str(self._train_cfg.lr))

        # transf = get_transforms(input_size=Input_size_Image, test_size=Test_size, kind='full', crop=True,
        #                         need=('train', 'val'), backbone=None)
        # transform_train = transf['train']
        # transform_test = transf['val']

        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths, train=False, test=False)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),#sampler=test_sampler, Attention je le met pas pour l instant
        )

        # train_set = datasets.ImageFolder(self._train_cfg.imnet_path + '/train', transform=transform_train)
        # train_sampler = RASampler(
        #     train_set, self._train_cfg.num_tasks, self._train_cfg.global_rank, len(train_set),
        #     self._train_cfg.batch_per_gpu, repetitions=3, len_factor=2.0, shuffle=True, drop_last=False
        # )
        #
        # self._train_loader = torch.utils.data.DataLoader(
        #     train_set,
        #     batch_size=self._train_cfg.batch_per_gpu,
        #     num_workers=(self._train_cfg.workers - 1),
        #     sampler=train_sampler,
        # )
        # test_set = datasets.ImageFolder(self._train_cfg.imnet_path + '/val', transform=transform_test)
        #
        # self._test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False,
        #     num_workers=(self._train_cfg.workers - 1),  # sampler=test_sampler, Attention je le met pas pour l instant
        # )

        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)

        print("Create distributed model", flush=True)
        # model = models.resnet50(pretrained=False)
        # models.

        model = timm.create_model('efficientnet_b7', pretrained=False)
        # model = models.resnet152(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # model.cuda(self._train_cfg.local_rank)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
        # )
        linear_scaled_lr = 8.0 * self._train_cfg.lr * self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks / 512.0
        # optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9,weight_decay=1e-4)

        optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=1e-5)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000)
        self._state = TrainerState(
            epoch=0, accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), str(self._train_cfg.weight_path))
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)
            print("load done")

    def _init_state_test2(self) -> None:
        """
               Initialize the state and load it from an existing checkpoint if any
               """
        torch.manual_seed(0)
        np.random.seed(0)
        print("Create data loaders", flush=True)

        Input_size_Image = self._train_cfg.input_size

        Test_size = Input_size_Image
        print("Input size : " + str(Input_size_Image))
        print("Test size : " + str(Input_size_Image))
        print("Initial LR :" + str(self._train_cfg.lr))

        transf = get_transforms(input_size=Input_size_Image, test_size=Test_size, kind='full', crop=True,
                                need=('train', 'val'), backbone=None)
        transform_train = transf['train']
        transform_test = transf['val']

        # train_set2 = datasets.ImageFolder(self._train_cfg.imnet_path + '/train',transform=transform_train)
        # step 2: data
        train_set = MURA_Dataset(self._train_cfg.data_root,
                                 self._train_cfg.data_root + self._train_cfg.train_image_paths, train=True, test=False)

        train_sampler = RASampler(
            train_set, self._train_cfg.num_tasks, self._train_cfg.global_rank, len(train_set),
            self._train_cfg.batch_per_gpu, repetitions=3, len_factor=2.0, shuffle=True, drop_last=False
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers - 1),
            sampler=train_sampler,
        )
        # self._train_loader2 = torch.utils.data.DataLoader(
        #     train_set2,
        #     batch_size=self._train_cfg.batch_per_gpu,
        #     num_workers=(self._train_cfg.workers-1),
        #     sampler=train_sampler,
        # )
        # test_set = datasets.ImageFolder(self._train_cfg.imnet_path  + '/val',transform=transform_test)
        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths,
                                train=False, test=False)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False,
            num_workers=(self._train_cfg.workers - 1),  # sampler=test_sampler, Attention je le met pas pour l instant
        )

        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)

        print("Create distributed model", flush=True)
        # model = models.resnet50(pretrained=False)
        # models.

        model = timm.create_model('efficientnet_b7', pretrained=False)
        # model = models.resnet152(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        # model.cuda(self._train_cfg.local_rank)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
        # )
        linear_scaled_lr = 8.0 * self._train_cfg.lr * self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks / 512.0
        # optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9,weight_decay=1e-4)

        optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=1e-5)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000)
        self._state = TrainerState(
            epoch=0, accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint_125.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)
            print("model load")

    def _init_state(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        torch.manual_seed(0)
        np.random.seed(0)
        print("Create data loaders", flush=True)
        
        Input_size_Image=self._train_cfg.input_size
        
        Test_size=Input_size_Image
        print("Input size : "+str(Input_size_Image))
        print("Test size : "+str(Input_size_Image))
        print("Initial LR :"+str(self._train_cfg.lr))
        
        transf=get_transforms(input_size=Input_size_Image,test_size=Test_size, kind='full', crop=True, need=('train', 'val'), backbone=None)
        transform_train = transf['train']
        transform_test = transf['val']
        
        #train_set2 = datasets.ImageFolder(self._train_cfg.imnet_path + '/train',transform=transform_train)
        # step 2: data
        train_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.train_image_paths, train=True, test=False)

        train_sampler = RASampler(
            train_set,self._train_cfg.num_tasks,self._train_cfg.global_rank,len(train_set),self._train_cfg.batch_per_gpu,repetitions=3,len_factor=2.0,shuffle=True, drop_last=False
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers-1),
            sampler=train_sampler,
        )
        # self._train_loader2 = torch.utils.data.DataLoader(
        #     train_set2,
        #     batch_size=self._train_cfg.batch_per_gpu,
        #     num_workers=(self._train_cfg.workers-1),
        #     sampler=train_sampler,
        # )
        #test_set = datasets.ImageFolder(self._train_cfg.imnet_path  + '/val',transform=transform_test)
        test_set = MURA_Dataset(self._train_cfg.data_root, self._train_cfg.data_root + self._train_cfg.test_image_paths, train=False, test=False)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),#sampler=test_sampler, Attention je le met pas pour l instant
        )

        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)

        print("Create distributed model", flush=True)
        #model = models.resnet50(pretrained=False)
        #models.



        model = timm.create_model('efficientnet_b7', pretrained=False)
        #model = models.resnet152(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)


        # model.cuda(self._train_cfg.local_rank)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
        # )
        linear_scaled_lr = 8.0 * self._train_cfg.lr * self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks /512.0
        #optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9,weight_decay=1e-4)

        #optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=1e-5 )
        # optimizer = optim.Adam(model.parameters(), lr=self._train_cfg.lr, weight_decay=0.1)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=33)

        optimizer = optim.Adam(model.parameters(),  lr=self._train_cfg.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)

        self._state = TrainerState(
            epoch=0,accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )




        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)
            print("model_load")

    def _train(self) -> Optional[float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        print_freq = 10
        acc = None
        max_accuracy=0.0
        # Start from the loaded epoch
        start_epoch = self._state.epoch
        for epoch in range(start_epoch, self._train_cfg.epochs):
            print(f"Start epoch {epoch}", flush=True)
            self._state.model.train()
            self._state.lr_scheduler.step(epoch)
            self._state.epoch = epoch
            running_loss = 0.0
            count=0
            for param_group in self._state.optimizer.param_groups:
                print("Current learning rate is: {0:.6f}".format(param_group['lr']))
            for i, data in enumerate(self._train_loader):
                inputs, labels, _, body_part = data
                #inputs, labels = data

                # inputs = inputs.cuda(self._train_cfg.local_rank, non_blocking=True)
                # labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                # dst = inputs.cpu().numpy()[0]
                # cv2.imshow('img', dst)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self._state.model(inputs)
                loss = criterion(outputs, labels)

                self._state.optimizer.zero_grad()
                loss.backward()
                self._state.optimizer.step()

                running_loss += loss.item()
                count=count+1
                if i % print_freq == print_freq - 1:
                    print(f"[{epoch:02d}, {i:05d}] loss: {running_loss/print_freq:.3f}", flush=True)

                    running_loss = 0.0
                if count>=5005 * 512 /(self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks):
                    break
                
            #if epoch==self._train_cfg.epochs-1:
            if epoch%4 == 1 or epoch == 0:
                print("Start evaluation of the model", flush=True)
                
                correct = 0
                total = 0
                count=0.0
                running_val_loss = 0.0
                self._state.model.eval()
                with torch.no_grad():
                    for data in self._test_loader:
                        images, labels, _, body_part = data
                        #images, labels = data
                        # images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                        # labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)

                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = self._state.model(images)
                        loss_val = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        running_val_loss += loss_val.item()
                        count=count+1.0

                acc = correct / total
                ls_nm=running_val_loss/count
                print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
                print(f"Loss of the network on the 50000 test images: {ls_nm:.3f}", flush=True)
                self._state.accuracy = acc
                if self._train_cfg.global_rank == 0:
                    self.checkpoint(rm_init=False)
                print("accuracy val epoch "+str(epoch)+" acc= "+str(acc))
                max_accuracy=np.max((max_accuracy,acc))
                if epoch==self._train_cfg.epochs-1:
                    return acc

    def _show(self) -> Optional[float]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #criterion = nn.CrossEntropyLoss()
            #print_freq = 10
            #acc = None
            #max_accuracy = 0.0
            # Start from the loaded epoch
            start_epoch = self._state.epoch
            for epoch in range(start_epoch, self._train_cfg.epochs):
                print(f"Start epoch {epoch}", flush=True)
                #self._state.model.train()
                #self._state.lr_scheduler.step(epoch)
                #self._state.epoch = epoch
                running_loss = 0.0
                count = 0
                for i, data in enumerate(self._train_loader):
                    inputs, labels, _, body_part = data
                    # inputs, labels = data

                    # inputs = inputs.cuda(self._train_cfg.local_rank, non_blocking=True)
                    # labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                    #while (True):
                    dst = inputs.cpu().numpy()[0]
                    dst = dst.transpose(1, 2, 0)
                    cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                    #dst = cv2.set
                    cv2.imshow('img', dst)
                    # 원본 이미지의 히스토그램
                    # hist_full = cv2.calcHist([dst], [1], None, [256], [0, 256])
                    # # red는 원본이미지 히스토그램, blue는 mask적용된 히스토그램
                    # plt.title('Histogram')
                    # plt.plot(hist_full, color='r')
                    # plt.xlim([0, 256])
                    # plt.show()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        #break
        except Exception as e:
            print("show Job failed : {0}".format(e))




    def _test(self) -> Optional[float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        self._state.model.eval()
        print("Start evaluation of the model", flush=True)

        correct = 0
        total = 0
        count = 0.0
        results = []
        with torch.no_grad():
            for data in self._test_loader:
                images, labels, _, body_part = data
                #images, labels = data
                # images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                # labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)

                images = images.to(device)
                labels = labels.to(device)

                outputs = self._state.model(images)
                #loss_val = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #running_val_loss += loss_val.item()
                count = count + 1.0
                #img_pred = (labels, predicted )
                batch_results = [(labels_, predicted_, body_part_) for labels_, predicted_, body_part_  in zip(labels.cpu().numpy(), predicted.cpu().numpy(), body_part)]
                results += batch_results#.append(img_pred)

                #probability = t.nn.functional.softmax(score)[:, 0].data.tolist()

                # 每一行为 图片路径 和 positive的概率
                #batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

                #results += batch_results

        #self.write_csv(results, "eff_predict.csv")
        acc = correct / total
        np_result = np.array(results)
        kappa_score = cohen_kappa_score(np_result[:,0], np_result[:,1])

        print("cohen kappa", kappa_score)
        XR_type_list = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
        for xr_type in XR_type_list:
            xr_type_correct = 0
            xr_type_result = np_result[np.where(np_result[:, 2] == xr_type)][:,0:2]
            xr_type_correct += (xr_type_result[:, 0] == xr_type_result[:, 1]).sum().item()
            xr_type_cohen_kappa = cohen_kappa_score(xr_type_result[:, 0], xr_type_result[:, 1])
            print('cohen_kappa {0} : {1:.2f}'.format(xr_type,xr_type_cohen_kappa ))
            print('ACC {0} : {1:.2f}'.format(xr_type,xr_type_correct/xr_type_result.shape[0] ))

        #np.unique(np_result[np.where(np_result[:, 2] == 'XR_WRIST')][:, 2])
        #ls_nm = running_val_loss / count
        print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
        #print(f"Loss of the network on the 50000 test images: {ls_nm:.3f}", flush=True)
        self._state.accuracy = acc
        # if self._train_cfg.global_rank == 0:
        #     self.checkpoint(rm_init=False)
        #print("accuracy val epoch " + str(epoch) + " acc= " + str(acc))
        #max_accuracy = np.max((max_accuracy, acc))
        #if epoch == self._train_cfg.epochs - 1:
        return acc

    def write_csv(self, results, file_name):
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'probability'])
            writer.writerows(results)

    def calculate_cohen_kappa(threshold=0.5):
        input_csv_file_path = 'result.csv'

        result_dict = {}
        with open(input_csv_file_path, 'r') as F:
            d = F.readlines()[1:]
            for data in d:
                (path, prob) = data.split(',')

                folder_path = path[:path.rfind('/')]
                prob = float(prob)

                if folder_path in result_dict.keys():
                    result_dict[folder_path].append(prob)
                else:
                    result_dict[folder_path] = [prob]

        for k, v in result_dict.items():
            result_dict[k] = np.mean(v)
            # visualize
            # print(k, result_dict[k])
        output_csv_path = 'predictions.csv'
        # 写入每个study的诊断csv
        with open(output_csv_path, 'w') as F:
            writer = csv.writer(F)
            for k, v in result_dict.items():
                path = k[len(opt.data_root):] + '/'
                value = 0 if v >= threshold else 1
                writer.writerow([path, value])

        XR_type_list = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

        for XR_type in XR_type_list:
            # 提取出 XR_type 下的所有folder路径，即 result_dict 中的key
            keys = [k for k, v in result_dict.items() if k.split('/')[6] == XR_type]

            y_true = [1 if key.split('_')[-1] == 'positive' else 0 for key in keys]
            y_pred = [0 if result_dict[key] >= threshold else 1 for key in keys]

            print('--------------------------------------------')

            kappa_score = cohen_kappa_score(y_true, y_pred)

            print(XR_type, kappa_score)

            # 预测准确的个数
            count = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_true))])
            print(XR_type, 'Accuracy', 100.0 * count / len(y_true))





