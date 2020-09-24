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
#import timm
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from PIL import Image
import logging
import matplotlib.pyplot as plt
import umap
from cycler import cycler
import record_keeper
import pytorch_metric_learning


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


# This will be used to create train and val sets that are class-disjoint
class ClassDisjointMURA(torch.utils.data.Dataset):
    def __init__(self, original_ds, transform):
        #rule = (lambda x: x < 50) if train else (lambda x: x >= 50)
        #train_filtered_idx = [i for i, x in enumerate(original_train.targets) if rule(x)]
        #val_filtered_idx = [i for i, x in enumerate(original_val.targets) if rule(x)]
        #self.data = np.concatenate([original_train.data[train_filtered_idx], original_val.data[val_filtered_idx]],
        #                           axis=0)
        self.data = original_ds.imgs
        self.targets = original_ds.targets
        self.transform = transform

        with open(self.data[0][0], 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #img = Image.fromarray(img)
        #img = Image.open(img)
        img = self.pil_loader(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

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
        
        self.train_set = datasets.ImageFolder(self._train_cfg.imnet_path + '/train',transform=transform_train)
        self.test_set = datasets.ImageFolder(self._train_cfg.imnet_path  + '/val',transform=transform_test)

        self.train_dataset = self.train_set
        self.val_dataset =  self.test_set

        # self.train_dataset = ClassDisjointMURA(self.train_set, transform_train)
        # self.val_dataset = ClassDisjointMURA(self.test_set, transform_test)


        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)
        print("Create distributed model", flush=True)

        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = common_functions.Identity()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torch.nn.DataParallel(model.to(device))
        embedder = torch.nn.DataParallel(MLP([num_ftrs, 512]).to(device))

        # Set optimizers
        trunk_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)

        # Set the loss function
        loss = losses.TripletMarginLoss(margin=0.1)

        # Set the mining function
        miner = miners.MultiSimilarityMiner(epsilon=0.1)

        # Set the dataloader sampler
        self.sampler = samplers.MPerClassSampler(self.train_dataset.targets, m=4, length_before_new_iter=len(self.train_dataset))

        # Package the above stuff into dictionaries.
        self.models_dict = {"trunk": model, "embedder": embedder}
        self.optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
        self.loss_funcs = {"metric_loss": loss}
        self.mining_funcs = {"tuple_miner": miner}


    def _train(self) -> Optional[float]:
        record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
        hooks = logging_presets.get_hook_container(record_keeper)
        dataset_dict = {"val": self.val_dataset}
        model_folder = "example_saved_models"

        def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
            logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
            label_set = np.unique(labels)
            num_classes = len(label_set)
            fig = plt.figure(figsize=(20, 15))
            plt.gca().set_prop_cycle(
                cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
            for i in range(num_classes):
                idx = labels == label_set[i]
                plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
            #plt.show()
            #plt.show(block=False)
            file_name = './plots/metric_{0}.png'.format(args[0])
            plt.savefig(file_name, dpi=300)
            #

        # Create the tester
        tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook,
                                                    visualizer=umap.UMAP(),
                                                    visualizer_hook=visualizer_hook,
                                                    dataloader_num_workers=32)

        end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                                    dataset_dict,
                                                    model_folder,
                                                    test_interval=1,
                                                    patience=200)

        trainer = trainers.MetricLossOnly(self.models_dict,
                                          self.optimizers,
                                          self._train_cfg.batch_per_gpu,
                                          self.loss_funcs,
                                          self.mining_funcs,
                                          #self._train_loader,
                                          self.train_set,
                                          sampler=self.sampler,
                                          dataloader_num_workers=self._train_cfg.workers - 1,
                                          end_of_iteration_hook=hooks.end_of_iteration_hook,
                                          end_of_epoch_hook=end_of_epoch_hook)

        #trainer.train(num_epochs=self._train_cfg.epochs)
        trainer.train(num_epochs=500)




