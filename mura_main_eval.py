# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import uuid
from pathlib import Path
from resnet156_scratch import TrainerConfig, ClusterConfig, Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


#def run(input_sizes,learning_rate, epochs, batch, node, workers, imnet_path, shared_folder_path, job_id, local_rank, global_rank, num_tasks, weight_path):
def run(input_size, learning_rate, epochs, batch, node, workers, imnet_path, shared_folder_path, job_id, local_rank, global_rank, num_tasks, weight_path, data_root ,
                    train_image_paths , test_image_paths):

    # def run(input_size, architecture, weight_path, dataset_path, batch, workers, save_path, shared_folder_path, job_id,
    #         local_rank, global_rank, num_tasks):
    #     shared_folder = None
    #     data_folder_Path = None
    #     if Path(str(shared_folder_path)).is_dir():
    #         shared_folder = Path(shared_folder_path )#+ "/extract/")
    #     else:
    #         raise RuntimeError("No shared folder available")
    #     if Path(str(dataset_path)).is_dir():
    #         data_folder_Path = Path(str(dataset_path))
    #     else:
    #         raise RuntimeError("No shared folder available")
    #cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    # train_cfg = TrainerConfig(
    #     data_folder=str(data_folder_Path),
    #     #architecture=architecture,
    #     weight_path=weight_path,
    #     input_size=input_size,
    #     dataset_path=dataset_path,
    #     batch_per_gpu=batch,
    #     workers=workers,
    #     save_path=save_path,
    #     local_rank=local_rank,
    #     global_rank=global_rank,
    #     num_tasks=num_tasks,
    #     job_id=job_id,
    #     save_folder=str(shared_folder),
    #
    # )
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")
    shared_folder=None
    data_folder_Path=None
    if Path(str(shared_folder_path)).is_dir():
        shared_folder=Path(shared_folder_path+"/training/")
    else:
        raise RuntimeError("No shared folder available")
    if Path(str(imnet_path)).is_dir():
        data_folder_Path=Path(str(imnet_path))
    else:
        raise RuntimeError("No shared folder available")




    train_cfg = TrainerConfig(
                    data_folder=str(data_folder_Path),
                    epochs=epochs,
                    lr=learning_rate,
                    input_size=input_size,
                    batch_per_gpu=batch,
                    save_folder=str(shared_folder_path),
                    workers=workers,
                    imnet_path=imnet_path,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    num_tasks=num_tasks,
                    job_id=job_id,
                    weight_path = weight_path,
                    data_root = data_root,
                    train_image_paths = train_image_paths ,
                    test_image_paths = test_image_paths,
                )

    os.makedirs(str(shared_folder), exist_ok=True)
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))

    cluster_cfg = cluster_cfg._replace(dist_url=init_file.as_uri())
    trainer = Trainer(train_cfg, cluster_cfg)

    # The code should be launch on each GPUs
    try:
        if global_rank == 0:
            val_accuracy = trainer.__eval__()
            print(f"Validation accuracy: {val_accuracy}")
        else:
            trainer.__eval__()
    except:
        print("Job failed")


if __name__ == "__main__":
    # parser = ArgumentParser(description="Evaluation script for AERES models",
    #                         formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--input-size', default=320, type=int, help='Images input size')
    # parser.add_argument('--architecture', default='IGAM_Resnext101_32x48d', type=str,
    #                     choices=['ResNet50', 'PNASNet', 'IGAM_Resnext101_32x48d'], help='Neural network architecture')
    # parser.add_argument('--weight-path',
    #                     default='/checkpoint/htouvron/baseline_our_method_BN_FC_URU/13972162/checkpoint_0.pth',
    #                     type=str, help='Neural network weights')
    # parser.add_argument('--dataset-path', default='/datasets01_101/imagenet_full_size/061417/val', type=str,
    #                     help='Dataset path')
    # parser.add_argument('--batch', default=32, type=int, help='Batch per GPU')
    # parser.add_argument('--workers', default=40, type=int, help='Numbers of CPUs')
    # parser.add_argument('--save-path', default='/checkpoint/htouvron/github_reproduce_result_extract/output_extract/',
    #                     type=str, help='Path where output will be save')
    # parser.add_argument('--shared-folder-path', default='your/shared/folder', type=str, help='Shared Folder')
    # parser.add_argument('--job-id', default='0', type=str, help='id of the execution')
    # parser.add_argument('--local-rank', default=0, type=int, help='GPU: Local rank')
    # parser.add_argument('--global-rank', default=0, type=int, help='GPU: glocal rank')
    # parser.add_argument('--num-tasks', default=32, type=int, help='How many GPUs are used')

    parser = ArgumentParser(description="Training script for ResNet50 FixRes",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-path', default='./output_extract/',
                        type=str, help='Path where output will be save')
    parser.add_argument('--architecture', default='efficientnet_b7', type=str,
                        choices=['ResNet50', 'PNASNet', 'IGAM_Resnext101_32x48d', 'efficientnet_b7'], help='Neural network architecture')
    parser.add_argument('--weight-path',
                        default='checkpoint.pth',
                        type=str, help='Neural network weights')
    parser.add_argument('--dataset-path', default='./datasets/mura_finetune_elbow/val', type=str,
                        help='Dataset path')
    parser.add_argument('--learning-rate', default=0.0001 , type=float, help='base learning rate')
    parser.add_argument('--input-size', default=320, type=int, help='images input size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--batch', default=6, type=int, help='Batch by GPU')
    parser.add_argument('--node', default=1, type=int, help='GPU nodes')
    parser.add_argument('--workers', default=1, type=int, help='Numbers of CPUs')
    parser.add_argument('--imnet-path', default='./datasets/mura_finetune_elbow/', type=str, help='ImageNet dataset path')
    parser.add_argument('--shared-folder-path', default='/home/data/extra/mura_fixRes/shared_folder/', type=str, help='Shared Folder')
    parser.add_argument('--job-id', default='ef_b7_all', type=str, help='id of the execution')
    parser.add_argument('--local-rank', default=0, type=int, help='GPU: Local rank')
    parser.add_argument('--global-rank', default=0, type=int, help='GPU: glocal rank')
    parser.add_argument('--num-tasks', default=5, type=int, help='How many GPUs are used')
    parser.add_argument('--data_root', default='/home/data/extra/', type=str, help='id of the execution')
    parser.add_argument('--train_image_paths', default='MURA-v1.1/train_image_paths.csv', type=str, help='id of the execution')
    parser.add_argument('--test_image_paths', default='MURA-v1.1/valid_image_paths.csv', type=str, help='id of the execution')
    args = parser.parse_args()
    # run(args.input_size, args.architecture, args.weight_path, args.dataset_path, args.batch, args.workers,
    #     args.save_path, args.shared_folder_path, args.job_id, args.local_rank, args.global_rank, args.num_tasks)
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    run(args.input_size,args.learning_rate,args.epochs,args.batch,args.node,args.workers,args.imnet_path,args.shared_folder_path,args.job_id,args.local_rank,args.global_rank,args.num_tasks,
        args.weight_path, args.data_root, args.train_image_paths, args.test_image_paths)
