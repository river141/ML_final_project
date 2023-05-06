import os
import time
import random
import functools
import itertools
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Options
testing_local = False
learning_rate = 0.0002
cycle_lambda = 10

if testing_local:
    data_dir = '.\\CycleGAN-custom\\maps\\'
    output_dir = '.\\CycleGAN-custom\\output_test\\'
    train_epochs = 1
    save_imgs_freq_iters = 10
    save_loss_freq_iters = 1
    threads = 4
    gpu_ids = []
else: 
    data_dir = './maps/'
    output_dir = './output_train_0/'
    train_epochs = 25
    save_imgs_freq_iters = 1000
    save_loss_freq_iters = 10
    threads = 1
    gpu_ids = [0]
img_channels = 3 #const
batch_size = 1 #const

def get_img_dirs(dir):
        img_dirs = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                img_dirs.append(path)
        return img_dirs

def init_net(net):
    """initializes the network using init function to set weights"""
    def net_init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # using GPU
    net.apply(net_init_func)
    return net

class Dataset(data.Dataset):
    def __init__(self, data_dir):
        self.root = data_dir
        self.dir_A = os.path.join(self.root, 'trainA') 
        self.dir_B = os.path.join(self.root, 'trainB')  

        self.A_paths = sorted(get_img_dirs(self.dir_A))
        self.B_paths = sorted(get_img_dirs(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths) 

        # torch transforms
        transform_list = [transforms.Resize([286,286], transforms.InterpolationMode.BICUBIC),
                          transforms.RandomCrop(256),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """for torch.utils.data.DataLoader"""

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # open image and apply image transformation
        A = self.transform(Image.open(A_path).convert('RGB'))
        B = self.transform(Image.open(B_path).convert('RGB'))

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

class DataLoader():
        """for multi-threaded data loading"""

        def __init__(self, data_dir):
            self.dataset = Dataset(data_dir)
            self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=threads)

        def load_data(self):
            return self

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            """for getting data during training loop"""
            for i, data in enumerate(self.dataloader):
                yield data

class CycleGan():
    def __init__(self):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = output_dir  # save all the checkpoints to save_dir

        # generators
        self.netG_A = init_net(ResnetGenerator())
        self.netG_B = init_net(ResnetGenerator())
        
        # discriminators
        self.netD_A = init_net(Discriminator())
        self.netD_B = init_net(Discriminator())

        # losses
        self.lossGAN = GANLoss().to(self.device) # defines two labels with MSE loss
        self.lossCycle = torch.nn.L1Loss() # L1 norm loss

        # optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=learning_rate, betas=(0.5, 0.999))

class ResnetGenerator(nn.Module):
    """architechure adapted from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style)"""
    def __init__(self):
        super(ResnetGenerator, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        blocks = [nn.ReflectionPad2d(3),
                nn.Conv2d(3, 64, kernel_size=7, padding=0),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # downsample 1
                norm_layer(128),
                nn.ReLU(True),
                nn.Conv2d(64*2, 128*2, kernel_size=3, stride=2, padding=1), #downsample 2
                norm_layer(128*2),
                nn.ReLU(True)]
        
        for _ in range(9):
            blocks += [ResnetBlock(norm_layer)] #resnet blocks

        blocks += [nn.ConvTranspose2d(128*2, 64*2, kernel_size=3, stride=2, padding=1, output_padding=1), #upsample 1
                norm_layer(int(64*2)),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), #upsample 2
                norm_layer(int(64)),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, padding=0),
                nn.Tanh()]
        
        self.resnet = nn.Sequential(*blocks)
        
    def forward(self, input):
        return self.resnet(input)
        
class ResnetBlock(nn.Module):
    """create resnet block with feedforward design based on https://arxiv.org/pdf/1512.03385.pdf"""
    def __init__(self,norm_layer):
        super(ResnetBlock, self).__init__()
        blocks = [nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3), 
                norm_layer(256), 
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3), 
                norm_layer(256)]
        
        self.resblock = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.resblock(x) #forward skip connection
    
class Discriminator(nn.Module):
    """patchGAN discriminator"""
    def __init__(self):
        super(Discriminator, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        blocks = [nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # layer 1 with 64 filters
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # layer 2 with 128 filters
                norm_layer(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2, padding=1), #layer 3 with 256 filters
                norm_layer(128*2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64*4, 128*4, kernel_size=4, stride=2, padding=1), #layer 4 with 512 filters
                norm_layer(128*4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64*8, 1, kernel_size=4, stride=1, padding=1)] #convolution to output
        
        self.discrim = nn.Sequential(*blocks)

    def forward(self, input):
        return self.discrim(input)

class GANLoss(nn.Module):

    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, is_real):
        """calculates loss given is_real or not"""
        label = self.real_label if is_real else self.fake_label
        return self.loss(prediction, label.expand_as(prediction))

if __name__ == '__main__':

    # Defining dataset with loader
    dataset = DataLoader(data_dir).load_data()
    print('defined dataset')

    # Creating model architechure
    cyclegan_model = CycleGan() 
    print('defined model')

    # Get and load test images
    A_path = os.path.join(data_dir,'testA','1_A.jpg')
    B_path = os.path.join(data_dir,'testB','1_B.jpg')
    test_A = dataset.dataset.transform(Image.open(A_path).convert('RGB')).unsqueeze(0)
    test_B = dataset.dataset.transform(Image.open(B_path).convert('RGB')).unsqueeze(0)
    print('loaded test images')

    # Training 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    iters = 0
    start_time = time.time()
    for epoch in range(train_epochs): 
        print('started epoch: ',epoch)
        for i, data in enumerate(dataset): # loops over A and B images
            # input images
            real_A = data['A'].to(cyclegan_model.device)
            real_B = data['B'].to(cyclegan_model.device)
            image_paths = data['A_paths']
            discriminators = [cyclegan_model.netD_A, cyclegan_model.netD_B]

            # forward
            fake_B = cyclegan_model.netG_A(real_A) # A -> B
            new_A = cyclegan_model.netG_B(fake_B)  # (A -> B) -> A
            fake_A = cyclegan_model.netG_B(real_B) # B -> A
            new_B = cyclegan_model.netG_A(fake_A)  # (B -> A) -> B

            # turn off gradients for discriminators
            for disc in discriminators: 
                for p in disc.parameters(): p.requires_grad = False

            # optimize A and B generators
            cyclegan_model.optimizer_G.zero_grad()
            G_ls = [cyclegan_model.lossGAN(cyclegan_model.netD_A(fake_B), True), # GAN loss (generator A)
                    cyclegan_model.lossGAN(cyclegan_model.netD_B(fake_A), True), # GAN loss (generator B)
                    cyclegan_model.lossCycle(new_A, real_A) * cycle_lambda, # cycle loss (forward A)
                    cyclegan_model.lossCycle(new_B, real_B) * cycle_lambda] # cycle loss (backward B)
            G_loss = G_ls[0] + G_ls[1] + G_ls[2] + G_ls[3]
            G_loss.backward() # calculating gradients
            cyclegan_model.optimizer_G.step() # optimize weights for G

            # turn back on gradients for discriminators
            for disc in discriminators:
                for p in disc.parameters(): p.requires_grad = True

            # optimize A and B discriminators -> note that we divide the discriminator objective by 2 to slow it relative
            cyclegan_model.optimizer_D.zero_grad()
            D_A_loss = 0.5 * (cyclegan_model.lossGAN(cyclegan_model.netD_A(real_B), True) + # GAN loss for determining real B
                    cyclegan_model.lossGAN(cyclegan_model.netD_A(fake_B.detach()), False))  # GAN loss for determining fake B
            D_A_loss.backward()
            D_B_loss = 0.5 * (cyclegan_model.lossGAN(cyclegan_model.netD_B(real_A), True) + # GAN loss for determining real A
                    cyclegan_model.lossGAN(cyclegan_model.netD_B(fake_A.detach()), False))  # GAN loss for determining fake A
            D_B_loss.backward()
            cyclegan_model.optimizer_D.step()

            # get and write losses -> [G_A, G_B, Cycle_A, Cycle_B, D_A, D_B] <- note this order
            if iters % save_loss_freq_iters == 0:
                losses = [iters, float(G_ls[0]), float(G_ls[1]), float(G_ls[2]), float(G_ls[3]), float(D_A_loss), float(D_B_loss)]
                losses = [format(num, ".4f") for num in losses]

                file_loc = os.path.join(output_dir, 'losses.txt')
                with open(file_loc, "a") as file:
                    if iters == 0:
                        file.truncate(0)
                    file.write(" ".join(losses) + "\n")
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Iter: ',iters, f" in {elapsed_time:.2f} s, "," Losses: ",losses)

            # generate test images and save all
            if iters % save_imgs_freq_iters == 0:
                fake_BT = cyclegan_model.netG_A(test_A) # A -> B
                new_AT = cyclegan_model.netG_B(fake_BT)  # (A -> B) -> A
                fake_AT = cyclegan_model.netG_B(test_B) # B -> A
                new_BT = cyclegan_model.netG_A(fake_AT)  # (B -> A) -> B
                
                img_dir = os.path.join(output_dir, 'iter%.3d_%s' % (iters, 'results'))
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                # order images
                imgs = [real_A, fake_B, new_A, real_B, fake_A, new_B, 
                        test_A, fake_BT, new_AT, test_B, fake_AT, new_BT]
                names = ['real_A', 'fake_B', 'new_A', 'real_B', 'fake_A', 'new_B', 
                        'test_A', 'fake_BT', 'new_AT', 'test_B', 'fake_AT', 'new_BT']
                
                for k, img in enumerate(imgs):
                    img = img.data[0].cpu().float().numpy()
                    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
                    img = Image.fromarray(img.astype(np.uint8))
                    img.save(os.path.join(img_dir, '%s.png' % (names[k])))

            iters += 1