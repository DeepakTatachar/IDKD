import os

# Code architecture.
# We create n processes one for each node/edge
# Communicate between processes using CPU MPI data transfers, yes this is slow
# But allows us to run multiple node/edge on a single GPU thus can emulate large number of nodes on a single machine

def main():
    import argparse
    import torch
    import random
    from utils.qgm_optimizer import QGM_SGD, get_data, TensorBuffer
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset, create_dataloaders
    from utils.averagemeter import AverageMeter
    from utils.instantiate_model import instantiate_model
    from utils.SoftCELoss import SoftCrossEntropy
    from utils.graph_manager import GraphManager, GraphType
    from utils.proxy_dataset import ProxySet, OneHotLabelDataset
    from utils.timer import Timer
    from torch.utils.tensorboard import SummaryWriter
    import logging
    import numpy as np
    import json
    

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=10,             type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.01,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--dataset_kd',             default='tinyimagenet',      type=str,       help='Set dataset to use')

    # Decentralized params
    parser.add_argument('--alpha',                  default=0.1,            type=float,     help='Parameter is alpha of Dirichlet Distribution. Divides the data index into n node subset')
    parser.add_argument('--beta',                   default=0.9,            type=float,     help='Momentum coeff')
    parser.add_argument('--weight_decay',           default=1e-4,           type=float,     help='Weight decay')
    parser.add_argument("--warm_up_epochs",         default=5,              type=int)
    parser.add_argument("--warm_up_start_lr",       default=0.05,           type=float)
    parser.add_argument("--network",                default="ring",         type=str)

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=32,             type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=32,             type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=True,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet20evo',     type=str,       help='Network architecture')

    # Summary Writer Tensorboard
    parser.add_argument('--comment',                default="",             type=str,       help='Comment for tensorboard')
    parser.add_argument('--num_gpus',               default=2,              type=int,       help='Number of GPUs available to train')
    parser.add_argument('--temp',                   default=1,              type=float,     help='Temperature Scaling')
    parser.add_argument('--db_conf',                default=False,          type=str2bool,  help='Use db conf')
    parser.add_argument('--conf_threshold',         default=0.98,           type=float,     help='OoD confidence threshold')
    parser.add_argument('--iidfy_epoch',            default=241,            type=int,       help='Epoch to exchange data for IIDfying')

    global args
    args = parser.parse_args()

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
 
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        version_list = list(map(float, torch.__version__.split(".")))
        if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
            torch.set_deterministic(True)
        else:
            torch.use_deterministic_algorithms(True)
    except:
        torch.use_deterministic_algorithms(True)

    # Initialize Network Graph
    graph_manager = GraphManager()

    # Create a logger
    logger = logging.getLogger(f'Train Logger {graph_manager.rank}')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join('./logs', f'{args.dataset.lower()}_idkd_{args.dataset_kd.lower()}_node{graph_manager.rank}_alpha{args.alpha}_{args.suffix}.log'))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s [{graph_manager.rank}] %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    if args.dataset.lower() == 'imagenette':
        args.conf_threshold = 0.99999
        args.iidfy_epoch = 291
        args.arch = 'resnet20evonette'

    graph_manager.set_logger(logger)

    network_type = {
        'ring': GraphType.Ring,
        'social15': GraphType.Social_15
    }

    graph_manager.set_graph_type(network_type[args.network])
    logger.info(f"Using network {network_type[args.network]} with {graph_manager.backend.world_size} nodes")

    logger.info(f"{graph_manager.backend.rank}, {graph_manager.backend.world_size}, {graph_manager.backend.comm.Get_parent()}, {graph_manager.backend.comm.Get_group()}")

    # Parameters
    num_epochs = args.epochs
    learning_rate = args.lr
    gpu_id = graph_manager.rank % args.num_gpus

    # Setup right device to run on
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    graph_manager.set_device(device)
    logger.info(f'Device {device}')

    logger.info('Dummy dataset creation to get size of the dataset etc')
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=0.0,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=False,
        random_seed=args.random_seed,
        logger=logger)

    index, label_dist = graph_manager.partition_get_subset_idx(dataset, args.alpha)

    # Use the following transform for training and testing
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        logger=logger,
        index=index)

    private_dataset = OneHotLabelDataset(dataset.train_loader.dataset, dataset.num_classes, logger)
    
    # Public dataset
    p_dataset = load_dataset(
        dataset=args.dataset_kd,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=False,
        random_seed=args.random_seed,
        logger=logger,
        resize_shape=(dataset.img_dim, dataset.img_dim),
        mean=dataset.mean,
        std=dataset.std)
    
    class_count_train = {}
    class_count_val = {}
    for i in range(dataset.num_classes):
        class_count_train[i] = 0
        class_count_val[i] = 0

    for _, labels in dataset.train_loader:
        labels = labels.numpy()
        for i in range(dataset.num_classes):
            class_count_train[i] += np.where(labels == i)[0].shape[0]

    for _, labels in dataset.val_loader:
        labels = labels.numpy()
        for i in range(dataset.num_classes):
            class_count_val[i] += np.where(labels == i)[0].shape[0]

    logger.info(f'Train class count {json.dumps(class_count_train)}')
    logger.info(f'Val class count {json.dumps(class_count_val)}')
    model_args = {}

    if 'vit' in args.arch.lower():
        model_args['image_size'] = dataset.img_dim
        model_args['patch_size'] = 2
        model_args['dim'] = 64
        model_args['depth'] = 6 # Number of layers in the network
        model_args['heads'] = 8 # Number of heads in the network
        model_args['mlp_dim'] = 512

    suffix = ''
    for _, m_arg in model_args.items():
        suffix += str(m_arg) + '_'

    args.suffix = suffix + f'node{graph_manager.rank}_alpha{args.alpha}_' + args.suffix

    # Instantiate model 
    net, model_name = instantiate_model(dataset=dataset,
                                        arch=args.arch,
                                        suffix=args.suffix,
                                        load=args.resume,
                                        torch_weights=False,
                                        device=device,
                                        model_args=model_args,
                                        logger=logger)

    if args.use_seed:  
        if args.save_seed and graph_manager.rank == 0:
            logger.info("Saving Seed")
            torch.save(net.state_dict(),'./seed/' + args.dataset.lower() + '_' + args.arch +  "_hyb.Seed")
            logger.info("Seed saved")
        else:
            logger.info("Loading Seed")
            net.load_state_dict(torch.load('./seed/'+ args.dataset.lower() +'_' + args.arch + "_hyb.Seed"))
    else:
        logger.info("Random Initialization")
    
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": args.weight_decay,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in enumerate(net.parameters())
    ]

    timer = Timer(logger)
    optimizer = QGM_SGD(
        params,
        timer=timer,
        lr=learning_rate,
        logger=logger,
        graph_manager=graph_manager,
        momentum=args.beta,
        weight_decay=args.weight_decay,
        device=device)

    # Loss
    criterion = SoftCrossEntropy()

    if args.resume:
        saved_training_state = torch.load('./pretrained/'+ args.dataset.lower()+'/temp/' + model_name  + '.temp')
        start_epoch =  saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict(saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')

    net = net.to(device)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
                                                     gamma=0.1)



    writer = SummaryWriter(comment=args.comment + f'node{graph_manager.rank}')


    neighbors_weights = graph_manager.w[graph_manager.rank]
    neighbors_rank = np.where(neighbors_weights > 0)[0]
    
    dataset = create_dataloaders(
        "Private set", 
        dataset.train_batch_size, 
        dataset.test_batch_size, 
        dataset.val_split, 
        dataset.augment, 
        dataset.padding_crop, 
        dataset.shuffle, 
        dataset.random_seed, 
        dataset.mean, 
        dataset.std, 
        dataset.transforms.train, 
        dataset.transforms.test, 
        dataset.transforms.val, 
        logger, 
        dataset.img_dim, 
        dataset.img_ch, 
        dataset.num_classes, 
        dataset.num_worker, 
        private_dataset, 
        private_dataset, 
        dataset.testset, 
        dataset.train_length)

    def wait_sync(graph_manager, optimizer):
        inactive_neighs = np.array([], dtype=np.int64)
        neighbors_weights = graph_manager.w[graph_manager.rank]
        neighbors_rank = np.where(neighbors_weights > 0)[0]
        while(1):
            recv_arr = graph_manager.comm_neigh(np.array([1]), inactive_neighs=inactive_neighs)
            recv_arr = np.array(recv_arr)

            inactive_neighs = np.append(inactive_neighs, np.where(recv_arr == 2)[0])
            params, _ = get_data(
                optimizer.param_groups, optimizer.param_names, is_get_grad=False
            )

            flatten_params = TensorBuffer(params)
            np_param = flatten_params.buffer.clone().detach().cpu().numpy()
            recv_params = graph_manager.comm_neigh(np_param, inactive_neighs=inactive_neighs)

            no_comp_neigh = len(np.where(recv_arr == 1)[0]) + len(inactive_neighs)
            # If all the neighbours are done exit loop
            if no_comp_neigh == len(neighbors_rank):
                break

        # Needed in case of a ring network. End here if neighbours done. But neighbours of neighoubrs maynot be
        # Thus when neighbours try to send data to us causes things to be stuck.
        # So let all the neighbours so that they donot try to send data to us 
        recv_arr = graph_manager.comm_neigh(np.array([2]), inactive_neighs=inactive_neighs)

        graph_manager.barrier()
        graph_manager.set_inactive_neighbours(None)
        return

    soft_labels = None
    iteration = 0
    up = torch.nn.Upsample(size=(224, 224), mode='bilinear')

    # Train model
    for epoch in range(start_epoch, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save_ckpt = False
        losses = AverageMeter('Loss', ':.4e')
        logger.info('')
        for batch_idx, (data, labels) in enumerate(dataset.train_loader):

            data = data.to(device)
            labels = labels.to(device)
            iteration += 1
            
            # Clears gradients of all the parameter tensors
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, labels)
            loss.backward()

            _ = graph_manager.comm_neigh(np.array([0]))

            optimizer.step()
            losses.update(loss.item())
            
            if batch_idx % 48 == 0:
                trainset_len = (1 - args.val_split) * len(dataset.train_loader.dataset)
                curr_acc = 100. * train_total / trainset_len
                logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        train_total,
                        trainset_len,
                        curr_acc,
                        losses.avg))
                logger.info(f"LR: {scheduler.get_last_lr()[-1]}")
            
            train_correct += (out.max(-1)[1] == labels.argmax()).sum().long().item()
            train_total += labels.shape[0]
      
        if epoch >= args.iidfy_epoch and epoch % 5 == 0:
            logger.info("Waiting for sync")
            wait_sync(graph_manager, optimizer)
            logger.info("Done sync")
            p_dataset_len = int(p_dataset.train_length * (1 - p_dataset.val_split))

            if soft_labels is None:
                soft_labels = torch.zeros((p_dataset_len, dataset.num_classes))
                send_samples = torch.zeros((p_dataset_len))
                net.eval()
                # Comm and update grad
                with torch.no_grad():
                    start_idx = 0
                    for batch_idx, (data, labels) in enumerate(p_dataset.train_loader):
                        out = net(data.to(device)) / args.temp
                        conf = torch.nn.functional.softmax(out, dim=1)
                        conf_diff = torch.topk(conf, k=2, dim=1)[0]
                        conf_gt_thresh = conf.max(axis=1)[0].repeat(dataset.num_classes, 1).T > args.conf_threshold
                        conf_db = torch.abs(conf_diff[:, 0] - conf_diff[:, 1]).repeat(dataset.num_classes, 1).T < 0.1

                        cond = torch.logical_or(conf_gt_thresh, conf_db) if args.db_conf else conf_gt_thresh
                        conf = torch.where(
                            cond,
                            conf,
                            torch.zeros_like(conf, device=conf.device))

                        stop_idx = start_idx + out.shape[0]
                        soft_labels[start_idx:stop_idx] = conf.cpu()
                        send_samples[start_idx:stop_idx] = cond[:,0].clone().cpu()
                        start_idx = stop_idx

                soft_labels = soft_labels.numpy()
            else:
                soft_labels = student_labels.numpy()
                send_samples = torch.tensor(non_zero_idxs)
            
          
            recv_labels = [] 

            logger.info("Public data forward prop done")
            logger.info(f"Active neighs {graph_manager.get_active_neighs()}")
            send_idxs = torch.where(send_samples)[0].numpy()
            recv_idxs = graph_manager.comm_neigh_diff_size(send_idxs)
            logger.info("Labels sent")
            logger.info(send_idxs)
            recv_compressed_labels = graph_manager.comm_neigh_diff_size(soft_labels[send_idxs])
            logger.info("Compressed labels received sent")
            num_neigh = len(recv_compressed_labels)
            recv_labels = np.zeros((num_neigh, p_dataset_len, dataset.num_classes))

            for neigh_idx, recv_label in enumerate(recv_compressed_labels):
                recv_labels[neigh_idx, recv_idxs[neigh_idx]] = recv_label

            logger.info("Done recv")
            conf_node_sum = np.where(recv_labels != 0, 1, 0)
            logger.info(f'Recv after shape {recv_labels.shape}')


            num_conf_nodes = np.column_stack([np.where(conf_node_sum.sum(axis=-1) > 0, 1, 0).sum(axis=0).T] * dataset.num_classes)
            sum_logits = recv_labels.sum(axis=0)
            logger.info(f'sum shape {sum_logits.shape}')
            logger.info(f'num_conf_nodes shape {num_conf_nodes.shape}')
            non_zero_idxs = np.where(num_conf_nodes[:, 0] > 0, 1, 0)

            neighbours = np.where(graph_manager.w[graph_manager.rank] > 0)[0]
            logger.info(neighbours)
            pred_classes_for_nodes = recv_labels.argmax(-1)
            logger.info(recv_labels.shape)
            logger.info(pred_classes_for_nodes.shape)
            data_samples_for_pred = [label_dist[neigh, pred_classes_for_nodes[n_idx]] for n_idx, neigh in enumerate(neighbours)]
            data_samples_for_pred = np.array(data_samples_for_pred)
            logger.info(data_samples_for_pred.shape)
            node_idx = data_samples_for_pred.argmax(0)
            mask = torch.nn.functional.one_hot(
                torch.Tensor(node_idx).to(torch.int64),
                recv_labels.shape[0]).T
            mask = torch.stack([mask]*dataset.num_classes, 2)
            logger.info(mask.shape)
            logger.info(recv_labels.shape)
            student_labels = (mask * torch.Tensor(recv_labels)).sum(0)
            logger.info(f'No of p dataset datapoint {non_zero_idxs.sum()}')
            logger.info(f'Conf mean {student_labels.max(1)[0].mean()}')

            if len(non_zero_idxs) > 0:
                logger.info(f'Length of proxy set: {len(non_zero_idxs)}')
                
                # Create a proxy subset based on received data
                proxy_dataset = ProxySet(
                    data=p_dataset.train_loader.dataset,
                    labels=student_labels,
                    idxs=np.where(non_zero_idxs > 0)[0],
                    transform=p_dataset.transforms.train,
                    logger=logger,
                )

                torch.save(student_labels.clone().cpu(),
                           f"./output/{args.dataset.lower()}_qgm_hybrid_{args.dataset_kd.lower()}_node{graph_manager.rank}_alpha{args.alpha}_{args.suffix}.ckpt")

                new_dataset = torch.utils.data.ConcatDataset(
                    [private_dataset, proxy_dataset]
                )

                dataset = create_dataloaders(
                    "Private set", 
                    dataset.train_batch_size, 
                    dataset.test_batch_size, 
                    dataset.val_split, 
                    dataset.augment, 
                    dataset.padding_crop, 
                    dataset.shuffle, 
                    dataset.random_seed, 
                    dataset.mean, 
                    dataset.std, 
                    dataset.transforms.train, 
                    dataset.transforms.test, 
                    dataset.transforms.val, 
                    logger, 
                    dataset.img_dim, 
                    dataset.img_ch, 
                    dataset.num_classes, 
                    dataset.num_worker, 
                    new_dataset, 
                    new_dataset, 
                    dataset.testset, 
                    dataset.train_length)
            
            graph_manager.set_graph_type(network_type[args.network])

        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        logger.info(
            'Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch,
                train_correct,
                train_total,
                train_accuracy,
                losses.avg))

        writer.add_scalar('Loss/train', losses.avg, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        val_correct, val_total, val_accuracy, val_loss = -1, -1, -1, -1
        if args.val_split > 0.0: 
            val_correct, val_total, val_accuracy, val_loss = inference(
                net=net,
                data_loader=dataset.val_loader,
                device=device,
                loss=criterion,
                sce=False)

            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy 
                best_val_loss = best_val_loss
            
            save_ckpt = True
        else: 
            val_accuracy= float('inf')
            save_ckpt = True


        saved_training_state = {    'epoch'     : epoch + 1,
                                    'optimizer' : optimizer.state_dict(),
                                    'model'     : net.state_dict(),
                                    'best_val_accuracy' : best_val_accuracy,
                                    'best_val_loss' : best_val_loss,
                                    'index': index,
                                    'label_dist': label_dist
                                }

        torch.save(saved_training_state, './pretrained/'+ args.dataset.lower() + '/temp/' + model_name  + '.temp')
        
        if save_ckpt:
            logger.info("Saving checkpoint...")
            torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy, test_loss = inference(
                    net=net,
                    data_loader=dataset.test_loader,
                    device=device,
                    loss=torch.nn.CrossEntropyLoss())

                logger.info(
                    " Training set accuracy: {}/{}({:.2f}%) \n" 
                    " Validation set accuracy: {}/{}({:.2f}%)\n"
                    " Validation loss {:.6f}\n"
                    " Test set: Accuracy: {}/{} ({:.2f}%) {}".format(
                        train_correct,
                        train_total,
                        train_accuracy,
                        val_correct,
                        val_total,
                        val_accuracy,
                        val_loss,
                        test_correct,
                        test_total,
                        test_accuracy,
                        test_loss))

    net.train()
    logger.info(timer.summary())
    logger.info("End of training without reusing Validation set")
    logger.info('Waiting for other nodes still communicating')
    wait_sync(graph_manager, optimizer)
    logger.info('All nodes done')
    with torch.no_grad():
        params, _ = get_data(
            optimizer.param_groups, optimizer.param_names, is_get_grad=False
        )

        flatten_params = TensorBuffer(params)
        np_param = flatten_params.buffer.clone().detach().cpu().numpy()
        neighbors_weights = graph_manager.w[graph_manager.rank]
        neighbors_rank = np.where(neighbors_weights > 0)[0]
        recv_params = graph_manager.comm_neigh(np_param)
        weighted_avg = None
        for neigh_idx, neigh_param in enumerate(recv_params):
            n_rank = neighbors_rank[neigh_idx]
            if weighted_avg is None:
                weighted_avg = np.zeros_like(neigh_param)
                            
            weighted_avg += neighbors_weights[n_rank] * neigh_param

        flatten_params.buffer = torch.Tensor(weighted_avg).to(device)
        flatten_params.unpack(params)

    # Test model
    # Set the model to eval mode
    logger.info("\nEnd of training without reusing Validation set")

    if not args.resume:
        logger.info('Saving the final model')
        torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    logger.info(f'Comm bytes {graph_manager.comm_bytes}')
    logger.info(f'Num Iterations {iteration}')

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()