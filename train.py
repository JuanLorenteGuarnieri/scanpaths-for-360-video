import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from DataLoader360Video import RGB_and_OF, RGB, RGB_with_GM
# from sphericalKLDiv import  KLWeightedLossSequence
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import time
from torch.utils.data import DataLoader
import scanpath_generator as sg
import gaussian_map_generator as gmg
import models
from utils import read_txt_file
from dtw_kldiv import KLSoftDTW
import numpy as np
import cv2

# Import config file
import config

def train(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.makedirs(path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    model.train()

    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0
            
        for x, y in train_data:
            

            model.zero_grad()
        
            pred = model(x.to(device))
            
            loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))

            loss.sum().backward()
            optimizer.step()

            avg_loss_train += loss.sum().item()

            counter_train += 1
            if counter_train % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter_train, len(train_data),
                                                                                        avg_loss_train / counter_train))

        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))
        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set
        with torch.no_grad():
            for x, y in val_data:
                counter_val += 1
                pred = model(x.to(device))
                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))
                avg_loss_val += loss.sum().item()

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)
        
        # Save checkpoint and model every 50 epochs
        if epoch % 50 == 0:
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    return model

def train_scanpath_video_predictor(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.makedirs(path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
    images, labels = next(iter(train_data))
    m, n = images.size()[3:5]  # Obtener las dimensiones espaciales: altura (m) y anchura (n)

    # Crear la matriz de coordenadas Y
    coord_y = torch.arange(float(m)).unsqueeze(1).expand(m, n)
    # Crear la matriz de coordenadas X
    coord_x = torch.arange(float(n)).unsqueeze(0).expand(m, n)

    # Apilar las dos matrices para crear la matriz de coordenadas convolucionales
    coordconv_matrix = torch.stack((coord_x, coord_y), dim=0)
    # torch.tensor(data_numpy)
    # coordconv_matrix = coordconv_matrix.astype(np.float32)

    coordconv_matrix = torch.FloatTensor(coordconv_matrix).unsqueeze(0)
    model.train()

    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0
            
        for x, y in train_data:


            model.zero_grad()
            
            outputs = []
            gaussian_map = y[:, 0, :, :, :]

            for t in range(x.shape[1]):
                # Obtener el frame actual
                frame_del_video = x[:, t, :, :, :]
                # print('Gaussian map: ',gaussian_map.shape)

                # Concatenar el frame actual con coordconv_matrix a lo largo de la dimensión de los canales
                frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                # print('Input: ',frame_with_coords.shape)

                # Pasar el frame concatenado a través del encoder y decoder
                out = model(frame_with_coords)
                
                # print('Output: ',out.shape)
                with torch.no_grad():
                    out_squeezed=out.squeeze()
                    # print(out_squeezed.shape)
                    tSPM=out_squeezed.cpu().detach().numpy()
                    # print('Output matrix: ',tSPM.shape)
                    point = sg.generate_image_saliency_matrix_scanpath(tSPM, 1)
                    gaussian_map= gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[0],point[1]))
                    gaussian_map = gaussian_map.astype(np.float32)
                    gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)
                    outputs.append(gaussian_map)
            pred = torch.stack(outputs, dim=1)
            pred.squeeze(0)
            y.squeeze(0)

            # print('Prediccion: ', pred.shape)
            # print('Ground truth: ', y.shape)

            total_loss_DTW = 0
            for i in range(0, pred.shape[0]):
                # [b_size(1), len, H, W]
                # print('Prediccion elem: ', pred[:, i, :, :, :].shape)
                # print('Ground truth elem: ', y[:, i, :, :, :].shape)
                loss = criterion(pred[:, i, :, :, :], y[:, i, :, :, :])
                total_loss_DTW = total_loss_DTW + loss
            total_loss_DTW = total_loss_DTW / pred.shape[0]

            # loss.sum().backward()
            # optimizer.step()

            # avg_loss_train += loss.sum().item()
            avg_loss_train += total_loss_DTW

            counter_train += 1
            if counter_train % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter_train, len(train_data),
                                                                                        avg_loss_train / counter_train))

        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))
        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set
        with torch.no_grad():
            for x, y in val_data:
                model.zero_grad()
                outputs = []
                gaussian_map = y[:, 0, :, :, :]

                for t in range(x.shape[1]):
                    frame_del_video = x[:, t, :, :, :]
                    frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                    out = model(frame_with_coords)

                    with torch.no_grad():
                        out_squeezed=out.squeeze()
                        # print(out_squeezed.shape)
                        tSPM=out_squeezed.cpu().detach().numpy()
                        point = sg.generate_image_saliency_matrix_scanpath(tSPM, 1)
                        gaussian_map= gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[0],point[1]))
                        gaussian_map = gaussian_map.astype(np.float32)
                        gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)
                        outputs.append(gaussian_map)
                pred = torch.stack(outputs, dim=1)
                pred.squeeze(0)
                y.squeeze(0)

                total_loss_DTW = 0
                for i in range(0, pred.shape[0]):
                    loss = criterion(pred[:, i, :, :, :], y[:, i, :, :, :])
                    total_loss_DTW = total_loss_DTW + loss
                total_loss_DTW = total_loss_DTW / pred.shape[0]
                avg_loss_val += total_loss_DTW
                counter_val += 1

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)
        
        # Save checkpoint and model every 50 epochs
        if epoch % 50 == 0:
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Train SST-Sal

    model = models.SST_Sal_scanpath(hidden_dim=config.hidden_dim)
    # criterion = KLWeightedLossSequence()
    loss_KLDTW = KLSoftDTW(use_cuda=torch.cuda.is_available())


    video_names_train = read_txt_file(config.videos_train_file)

    # train_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, video_names_train, config.sequence_length, split='train', resolution=config.resolution)
    # val_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, video_names_train, config.sequence_length, split='validation', resolution=config.resolution)
    train_video360_dataset = RGB_with_GM(config.frames_dir, config.gt_dir, video_names_train, config.sequence_length, split='train', resolution=config.resolution)
    val_video360_dataset = RGB_with_GM(config.frames_dir, config.gt_dir, video_names_train, config.sequence_length, split='validation', resolution=config.resolution)

    train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    val_data = DataLoader(val_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)

    # print(model)
    # model = train(train_data, val_data, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)
    model = train_scanpath_video_predictor(train_data, val_data, model, device, loss_KLDTW, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)

    print("Training finished")