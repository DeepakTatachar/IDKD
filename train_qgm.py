import os

# Code architecture.
# We create n processes one for each node/edge
# Communicate between processes using CPU MPI data transfers, yes this is slow
# But allows us to run multiple node/edge on a single GPU thus can emulate large number of nodes on a single machine
# Adopted from the official implementation of QGM available at https://github.com/epfml/quasi-global-momentum

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
    parser.add_argument('--epochs',                 default=300,            type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.01,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')

    # Decentralized params
    parser.add_argument('--alpha',                  default=0.1,            type=float,     help='Parameter is alpha of Dirichlet Distribution. Divides the data index into n node subset')
    parser.add_argument('--beta',                   default=0.9,            type=float,     help='Momentum coeff')
    parser.add_argument('--weight_decay',           default=1e-4,           type=float,     help='Weight decay')
    parser.add_argument("--network",                default="ring",         type=str)

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=32,             type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=32,             type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=4,              type=int,       help='Initializing the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=True,           type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet20evo',  type=str,       help='Network architecture')

    # Summary Writer Tensorboard
    parser.add_argument('--comment',                default="",             type=str,       help='Comment for tensorboard')
    parser.add_argument('--num_gpus',               default=2,              type=int,       help='Number of GPUs available to train')

    global args
    args = parser.parse_args()
    if args.dataset.lower() == 'imagenette':
        args.arch = 'resnet20evonette'
   
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
    print(f'{args.dataset.lower()}_qgm_node{graph_manager.rank}_alpha{args.alpha}_{args.suffix}.log')
    handler = logging.FileHandler(os.path.join('./logs', f'{args.dataset.lower()}_qgm_node{graph_manager.rank}_alpha{args.alpha}_{args.suffix}.log'))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s [{graph_manager.rank}] %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

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
    net, model_name = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        torch_weights=False,
        device=device,
        model_args=model_args,
        logger=logger)

    if args.use_seed:  
        if args.save_seed:
            logger.info("Saving Seed")
            torch.save(net.state_dict(),'./seed/' + args.dataset.lower() + '_' + args.arch + ".Seed")
        else:
            logger.info("Loading Seed")
            net.load_state_dict(torch.load('./seed/'+ args.dataset.lower() +'_' + args.arch + ".Seed"))
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
        nesterov=True,
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
            
            train_correct += (out.max(-1)[1] == labels.argmax()).sum().long().item()
            train_total += labels.shape[0]

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
        if False: #args.val_split > 0.0: 
            val_correct, val_total, val_accuracy, val_loss = inference(net=net,
                                                                       data_loader=dataset.val_loader,
                                                                       device=device,
                                                                       loss=criterion)

            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy 
                best_val_loss = best_val_loss
                save_ckpt = True
        else: 
            val_accuracy= float('inf')
            save_ckpt = True


        saved_training_state = {
            'epoch'     : epoch + 1,
            'optimizer' : optimizer.state_dict(),
            'model'     : net.state_dict(),
            'best_val_accuracy' : best_val_accuracy,
            'best_val_loss' : best_val_loss
        }

        torch.save(saved_training_state, './pretrained/'+ args.dataset.lower() + '/temp/' + model_name  + '.temp')
        
        if save_ckpt:
            logger.info("Saving checkpoint...")
            torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(
                    net=net,
                    data_loader=dataset.test_loader,
                    device=device)

                logger.info(
                    " Training set accuracy: {}/{}({:.2f}%) \n" 
                    " Validation set accuracy: {}/{}({:.2f}%)\n"
                    " Test set: Accuracy: {}/{} ({:.2f}%)".format(
                        train_correct,
                        train_total,
                        train_accuracy,
                        val_correct,
                        val_total,
                        val_accuracy,
                        test_correct,
                        test_total,
                        test_accuracy))

    net.train()
    logger.info(timer.summary())
    logger.info("End of training without reusing Validation set")
    logger.info('Waiting for other nodes still communicating')
    
    inactive_neighs = np.array([], dtype=np.int64)
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

    saved_training_state = {
        'epoch'     : num_epochs,
        'optimizer' : optimizer.state_dict(),
        'model'     : net.state_dict(),
        'best_val_accuracy' : best_val_accuracy,
        'best_val_loss' : best_val_loss
    }

    logger.info('Saving the final model')
    torch.save(saved_training_state, './pretrained/'+ args.dataset.lower() + '/temp/' + model_name  + '.temp')
    torch.save(net.state_dict(), './pretrained/'+ args.dataset.lower()+'/' + model_name + '.ckpt')

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    logger.info(f'Comm bytes {graph_manager.comm_bytes}')

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()

