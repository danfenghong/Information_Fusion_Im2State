from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *
from im2state import im2state

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
if __name__=="__main__":

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='/mnt/sda/xxx/data/CAVE/Train/', type=str,
                        help='Path of the training data')
    parser.add_argument("--sizeI", default=96, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=2000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument('--lr', type=float, default=1e-4)
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(opt)

    ## New model
    print("===> New Model")
    scale = [4,8,16]#,8
    for ss in scale:
        opt.sf = ss
        model = im2state(dim=256, band=31, scale=opt.sf).cuda()
        model_structure(model)
        ## set the number of parallel GPUs
        print("===> Setting GPU")
        model = dataparallel(model, 1)

        ## Initialize weight
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)

        ## Load training data
        key = 'Train.txt'
        file_path = opt.data_path + key
        file_list = loadpath(file_path)
        HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 20)

        ## Load trained model
        initial_epoch = findLastCheckpoint(save_dir="./Checkpoint/f"+str(ss)+"/mambaall_depth")#_inf
        if initial_epoch > 0:
            print('resuming by loading epoch %04d' % initial_epoch)
            model = torch.load(os.path.join("./Checkpoint/f"+str(ss)+"/mambaall_depth", 'model_%04d.pth' % initial_epoch))

        ## Loss function
        criterion = nn.MSELoss().cuda()

        ## optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)
        criterion_mean = HLoss(0.5, 0.1).cuda()
        ## pipline of training
        for epoch in range(initial_epoch, 200+1):#150
            model.train()

            dataset = cave_dataset(opt, HR_HSI, HR_MSI)
            loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)
            scheduler.step(epoch)

            epoch_loss = 0

            start_time = time.time()
            for i, (LR, RGB, HR) in enumerate(loader_train):
                LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
                HR = HR.cuda()
                optimizer.zero_grad()
                model_out1, model_out2, model_out3 = model(LR.cuda(), RGB.cuda())
                loss = criterion(model_out1, model_out2)+criterion_mean(model_out1, HR)+criterion_mean(model_out2, HR)+criterion_mean(model_out3, HR)#
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                if i % 200 == 0:
                    print('%4d %4d / %4d loss = %.10f time = %s' % (
                        epoch + 1, i, len(dataset)// opt.batch_size, epoch_loss / ((i+1) * opt.batch_size), datetime.datetime.now()))

            elapsed_time = time.time() - start_time
            print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(dataset), elapsed_time))
            if epoch % 10 == 0:
                torch.save(model, os.path.join("./Checkpoint/f"+str(ss), 'model_%04d.pth' % (epoch)))  # save model
