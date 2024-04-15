import argparse
import numpy as np
import torch


from networks import GAN, TimeVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',help="Patient Dataset path",default=None)
    parser.add_argument('--saveDir',help="Save Checkpoint Directory",default=None)
    parser.add_argument('--ckpt',help="Checkpoint path",default=None)
    parser.add_argument('--max_iter',type=int,default=1000)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--task',help="Task to train on",default='T01')
    parser.add_argument('--model',help="Which model to use")

    ''' GAN model '''
    parser.add_argument('--g_hidden',type=int,help="Generator hidden channel size",default=64)
    parser.add_argument('--d_hidden',type=int,help="Discriminator hidden channel size",default=64)
    parser.add_argument('--n_critic',type=int,help="Number of iterations for Discriminator per one Generator iterations",default=5)
    parser.add_argument('--use_spectral',action='store_true')
    parser.add_argument('--use_eeg',action="store_true")
    parser.add_argument('--use_board',action='store_true')


    ''' VAE model '''
    parser.add_argument('--latent_dim',type=int,help="Hidden dimension size",default=100)
    parser.add_argument('--hidden_layer',nargs='+',type=int)




    args = parser.parse_args()

    # dataset = PatientDataset(root = args.dir,csv_file = "ParticipantCharacteristics.xlsx")
    # train_data,seq_len = get_task_data(dataset,args.task)

    train_data = np.load(args.dir)
    feat,seq_len = train_data[0].shape
    dataloader = torch.utils.data.DataLoader(train_data,args.batch_size,shuffle=True)

    print(f"Train with {args.max_iter} iterations")
    print(f"Features:{feat},Sequence Length:{seq_len}")

    if args.model == "EEGGAN":
        model = GAN(seq_len = seq_len, features=feat,n_critic=args.n_critic,
                g_hidden=args.g_hidden,d_hidden=args.d_hidden,max_iters=args.max_iter,
                saveDir=args.saveDir,ckptPath=args.ckpt,prefix=args.task,
                use_spectral=args.use_spectral,use_eeg=True,use_board=args.use_board)
    
    elif args.model == "GAN":
        model = GAN(seq_len = seq_len, features=feat,n_critic=args.n_critic,
                g_hidden=args.g_hidden,d_hidden=args.d_hidden,max_iters=args.max_iter,
                saveDir=args.saveDir,ckptPath=args.ckpt,prefix=args.task,
                use_spectral=args.use_spectral,use_eeg=False,use_board=args.use_board)
        
    elif args.model == "TIMEVAE":
        model = TimeVAE(seq_len = seq_len, feat_dim = feat, latent_dim= args.latent_dim,
                        hidden_layer = args.hidden_layer,max_iters= args.max_iter,
                        saveDir= args.saveDir, ckptPath= args.ckpt, prefix=args.task )

    model.train(dataloader)

