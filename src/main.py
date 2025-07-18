import os
import time
import torch
import logging
import numpy as np
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from dataloader import MSIRawDataset, collateGCN
from model_cam import GCNModel
from train import trainer, test, CAM
import json

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def main(args):

    # Logging the parameters
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # Create Train Validation and Test datasets
    if not args.test_only and not args.cam_only:
        print("Loading training set...")
        
        dataset_Train = MSIRawDataset(args.dataset_path,args.pre_process_param_name,args.network_param_name,
                                    mode="train", with_masses=args.with_masses,with_intensity=args.with_intensity, 
                                    normalization=args.normalize,random_state=args.random_state)
        print("Loading validation set...")
        dataset_Valid = MSIRawDataset(args.dataset_path,args.pre_process_param_name,args.network_param_name,
                                    mode="valid", with_masses=args.with_masses,with_intensity=args.with_intensity,  
                                    normalization=args.normalize,random_state=args.random_state)

    print("Loading testing set...")
    dataset_Test = MSIRawDataset(args.dataset_path,args.pre_process_param_name,args.network_param_name,
                                mode="test", with_masses=args.with_masses,with_intensity=args.with_intensity,  
                                normalization=args.normalize,random_state=args.random_state)

    # Create the deep learning model
    model = GCNModel(weights=args.load_weights, 
                    input_size=dataset_Test.num_features,
                    num_relations=dataset_Test.num_relations, 
                    num_classes=dataset_Test.num_classes, 
                    multiplier=args.multiplier).cuda()

    # Logging information about the model
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # Create the dataloaders for train validation and test datasets
    if not args.test_only and not args.cam_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True,collate_fn=collateGCN)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True,collate_fn=collateGCN)

    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True,collate_fn=collateGCN)


    # Training parameters
    if not args.test_only and not args.cam_only:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # Start training
        trainer(train_loader, val_loader, test_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # Load the best model and compute its performance
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    

    if not args.cam_only:

        performance = test(test_loader, model=model, model_name=args.model_name)
        print(performance)
        logging.info("Best performance at end of training ")
        #logging.info("Performance: " +  str(performance["accuracy"]))
        logging.info("Performance: " +  str(performance["balanced accuracy"]))
        torch.save(model, os.path.join(args.dataset_path,"models", args.model_name,args.pre_process_param_name + "_" + args.network_param_name + "_model.pth.tar"))

    if args.cam_only:

        CAM(test_loader,model, os.path.join(args.dataset_path,"models", "CAM_output"))
        
        return

    return performance

if __name__ == '__main__':

    # Load the arguments
    
    parser = ArgumentParser(description='GCN for MSI', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path',   required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--pre_process_param_name',   required=True, type=str, help='Name of the json file for MSI preprocessing')
    parser.add_argument('--network_param_name',   required=True, type=str, help='Name of the json file for network parameters')
    
    parser.add_argument('--model_name',   required=False, type=str,   default="GCN",     help='named of the model to save' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--cam_only',   required=False, action='store_true',  help='Perform Cam testing only' )
    
    parser.add_argument('--normalize',   required=False, action='store_true',  help='Perform testing only' )
    
    parser.add_argument('--with_masses',   required=False, action='store_true',  help='include the mass and mass defect in the node features' )
    parser.add_argument('--multiplier', required=False, type=int,   default=1,     help='Multiplier for the number of features in the GCN' )

    parser.add_argument('--with_intensity',   required=False, action='store_true' ,help='kept the the m/z intensity else replace by 1' )
    
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--batch_size', required=False, type=int,   default=64,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=5,     help='Patience before reducing LR (ReduceLROnPlateau)' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Evaluation frequency on test set' )

    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--random_state',   required=False, type=int,   default=0, help='random state')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # Logging information
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join(args.dataset_path,"models", args.model_name), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"models", "CAM_output"), exist_ok=True)
    
    
    network_param_json_path = args.dataset_path + '/parameters/network/' + args.network_param_name + '.json'
    network_params = []
    with open(network_param_json_path) as json_file:
        network_params = json.load(json_file)


    #log_path = os.path.join("models", args.model_name,datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    
    log_path = os.path.join(args.dataset_path,"models",args.model_name,args.pre_process_param_name + "_" + args.network_param_name+".log")
    
    
    
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')