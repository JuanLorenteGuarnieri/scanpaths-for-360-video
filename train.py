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

def train_scanpathDL(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

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

            pred = model(x.to(device), coordconv_matrix.to(device), y[:, 0, :, :, :], device)
            pred.squeeze(0).squeeze(0)
            y.squeeze(0).squeeze(0)

            print('Prediccion: ', pred.shape)
            print('Ground truth: ', y.shape)

            total_loss_DTW = 0
            for i in range(0, pred.shape[0]):
                loss = criterion(pred[i, 0, :, :], y[ i, 0, :, :])
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


def train_scanpathDL2(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.mkdir(path)
    os.mkdir(ckp_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
    images, labels = next(iter(train_data))
    m, n = images.size()[2:4]  # Obtener las dimensiones espaciales: altura (m) y anchura (n)

    # Crear la matriz de coordenadas Y
    coord_y = torch.arange(m).unsqueeze(1).expand(m, n)
    # Crear la matriz de coordenadas X
    coord_x = torch.arange(n).unsqueeze(0).expand(m, n)

    # Apilar las dos matrices para crear la matriz de coordenadas convolucionales
    coordconv_matrix = torch.stack((coord_x, coord_y), dim=0)
    # torch.tensor(data_numpy)
    coordconv_matrix = coordconv_matrix.astype(np.float32)

    coordconv_matrix = torch.FloatTensor(coordconv_matrix)
    coordconv_matrix = coordconv_matrix.permute(2, 0, 1)
    
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

            # Inicializar mapas previos con zeros o con el primer frame target si está disponible
            seq_len = x.size(1)  # Asumiendo que x es de forma [batch_size, seq_len, channels, height, width]
            spatial_input = torch.zeros_like(x[:, 0, :, :, :]).to(device)
            output_maps = []
            all_maps = []

            for step in range(seq_len):
                # aux = x[:, step, :, :, :]
                # frame_del_video = aux[0, :3, :, :]
                # spatial_input2= torch.FloatTensor(spatial_input)
                # coordconv_matrix2 =  torch.FloatTensor(coordconv_matrix)

                # output_map = model([torch.cat((torch.cat(frame_del_video, 0), torch.cat(spatial_input2, 0), torch.cat(coordconv_matrix2, 0)))].to(device))
                model(x.to(device), coordconv_matrix)
                output_maps.append(output_map)
                spatial_input = ... # Aquí falta incluir la lógica para generar la siguiente entrada espacial

                # Guardamos el mapa de GT para calcular el error DTW más tarde
                all_maps.append(y[:, step, :, :, :])

            # Ahora computamos la pérdida usando Dynamic Time Warping o la que prefieras
            final_output = torch.cat(output_maps, dim=1)
            total_loss_DTW = 0
            for m in all_maps:
                total_loss_DTW += criterion(final_output, m.to(device))
            total_loss_DTW = total_loss_DTW / seq_len

            total_loss_DTW.backward()
            optimizer.step()

            avg_loss_train += total_loss_DTW.item()
            counter_train += 1

            # Mostrar información de progreso
            if counter_train % 20 == 0:
                print(f"Epoch {epoch}......Step: {counter_train}/{len(train_data)}....... Average Loss for Epoch: {avg_loss_train / counter_train}")

        # Registro del tiempo y pérdida
        current_time = time.time()
        print(f"Epoch {epoch}/{EPOCHS}, Total Loss: {avg_loss_train / counter_train}")
        print(f"Total Time: {current_time - start_time} seconds")
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

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Train SST-Sal

    model = models.SST_Sal_scanpath(hidden_dim=config.hidden_dim)
    criterion = KLWeightedLossSequence()
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
    model = train_scanpathDL(train_data, val_data, model, device, loss_KLDTW, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)

    print("Training finished")