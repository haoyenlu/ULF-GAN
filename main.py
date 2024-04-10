from dataset import PatientDataset
import argparse
import numpy as np
import torch


from networks import GAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',help="Patient Dataset path",default=None)
    parser.add_argument('--saveDir',help="Save Checkpoint Directory",default=None)
    parser.add_argument('--ckpt',help="Checkpoint path",default=None)
    parser.add_argument('--max_iter',type=int,default=10000)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--g_hidden',type=int,help="Generator hidden channel size",default=64)
    parser.add_argument('--d_hidden',type=int,help="Discriminator hidden channel size",default=64)
    parser.add_argument('--task',help="Task to train on",default='T01')
    parser.add_argument('--n_critic',type=int,help="Number of iterations for Discriminator per one Generator iterations",default=5)
    parser.add_argument('--use_spectral',action='store_true')
    parser.add_argument('--use_eeg',action="store_true")


    args = parser.parse_args()

    # dataset = PatientDataset(root = args.dir,csv_file = "ParticipantCharacteristics.xlsx")
    # train_data,seq_len = get_task_data(dataset,args.task)

    train_data = np.load(args.dir)
    feat,seq_len = train_data[0].shape
    dataloader = torch.utils.data.DataLoader(train_data,args.batch_size,shuffle=True)

    print(args.max_iter)
    print(feat,seq_len)
    print("Use Spectral Loss:{}".format(args.use_spectral))
    print("Use EEG GAN: {}".format(args.use_eeg))
    wgan = GAN(seq_len = seq_len, features=feat,n_critic=args.n_critic,
               g_hidden=args.g_hidden,d_hidden=args.d_hidden,max_iters=args.max_iter,
            saveDir=args.saveDir,ckptPath=args.ckpt,prefix=args.task,
            use_spectral=args.use_spectral,use_eeg=args.use_eeg)
    
    wgan.train(dataloader,show_summary=True)

