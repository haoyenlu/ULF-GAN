from dataset import PatientDataset
import argparse
import numpy as np
import torch


from models import WGAN

def get_task_data(dataset,task):
    tasks = dataset._get_task_files()

    sequences = []
    max_seq_len = 0
    for _f in tasks[task]:
        data = dataset._get_data_from_file(_f)
        seq_len = data.shape[0]
        max_seq_len = max(max_seq_len,seq_len)
        sequences.append(data)

    data = pad_sequence(sequences,max_seq_len)
    data = data.transpose((0,2,1))
    return data,max_seq_len


def pad_sequence(sequences,max_seq_len):
    sequences_pad = []
    for seq in sequences:
        start = (max_seq_len - len(seq)) // 2
        end = start
        if (max_seq_len - len(seq)) % 2 != 0: start += 1
        x = np.pad(seq[:,0],(start,end),mode="edge")
        y = np.pad(seq[:,1],(start,end),mode="edge")
        z = np.pad(seq[:,2],(start,end),mode="edge")
        s = np.stack([x,y,z],axis=1)
        sequences_pad.append(s)

    return np.array(sequences_pad)

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



    args = parser.parse_args()

    # dataset = PatientDataset(root = args.dir,csv_file = "ParticipantCharacteristics.xlsx")
    # train_data,seq_len = get_task_data(dataset,args.task)

    train_data = np.load(args.dir)
    feat,seq_len = train_data[0].shape
    dataloader = torch.utils.data.DataLoader(train_data,args.batch_size,shuffle=True)

    print(args.max_iter)
    wgan = WGAN(seq_len = seq_len, features=feat,g_hidden=args.g_hidden,d_hidden=args.d_hidden,max_iters=args.max_iter,
            saveDir=args.saveDir,ckptPath=args.ckpt)
    
    wgan.train(dataloader,show_summary=True)

