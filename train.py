import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ctypes
ctypes.CDLL('/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so', mode=ctypes.RTLD_GLOBAL)


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
torch.autograd.set_detect_anomaly(True)

# Import config file
import config
from termcolor import colored

def print_gradient_summary(model, writer , epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            if grad_norm > 10:
                message_color_2 = colored(f"{grad_norm:.5f}", 'red') # Gradientes demasiado grandes
            elif grad_norm < 1e-5:
                message_color_2 = colored(f"{grad_norm:.5f}", 'red') # Gradientes demasiado pequeños
            else:
                message_color_2 = colored(f"{grad_norm:.5f}", 'green') # Gradientes normales

            # Determinar el color basado en el tipo de parámetro y su magnitud
            if 'encoder' in name:
                message_color_1 = colored("encoder - ", 'cyan')
            elif 'decoder' in name:
                message_color_1 = colored("decoder - ", 'blue')
            if 'weight' in name:
                message_color_1 = message_color_1 + colored("weight: ", 'magenta')
            elif 'bias' in name:
                message_color_1 = message_color_1 + colored("bias: ", 'yellow')
            print(message_color_1, message_color_2)
        else:
            print(colored(f"Gradiente para {name}: None encontrado", 'grey'))
        if epoch % 10 == 0:
            writer.add_histogram(f'params/{name}', param, epoch)

def train_SST_SAL_original(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

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

        with torch.no_grad(): # Evaluate on validation set
            for x, y in val_data:
                counter_val += 1
                pred = model(x.to(device))
                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))
                avg_loss_val += loss.sum().item()

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)

        if epoch % 50 == 0: # Save checkpoint and model every 50 epochs
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    torch.save(model, path + '/model.pth') # Save final model and checkpoints
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    return model

def get_gradients_hook(writer, layer_name, param_name, epoch):
    def hook(grad):
        writer.add_histogram(f"{layer_name}/{param_name}_gradients", grad, epoch)
    return hook

def get_activations_hook(writer, layer_name, epoch):
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor_output = output[0]  # Accede solo al tensor
            writer.add_histogram(f"{layer_name}_activations", tensor_output, epoch)
        else:
            writer.add_histogram(f"{layer_name}_activations", output, epoch) # Manejo del caso en el que la salida no sea una tupla
    return hook

def register_activation_hooks(model, writer, epoch):
    for name, module in model.named_modules():
        if name in ['encoder', 'decoder']:  # Cambia esto según los nombres específicos que quieras
            module.register_forward_hook(get_activations_hook(writer, name, epoch))

def train_scanpath_video_predictor(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):
    # register_activation_hooks(model, writer, epoch=0)
    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.makedirs(path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9,lr=lr) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # images, _ = next(iter(train_data))
    # m, n = images.size()[3:5]  # Obtener las dimensiones espaciales: altura (m) y anchura (n)
    m, n = config.resolution

    coord_y = torch.arange(float(m)).unsqueeze(1).expand(m, n) # Crear la matriz de coordenadas Y
    coord_x = torch.arange(float(n)).unsqueeze(0).expand(m, n) # Crear la matriz de coordenadas X
    coordconv_matrix = torch.stack((coord_x, coord_y), dim=0) # Apilar las dos matrices para crear la matriz de coordenadas convolucionales
    # coordconv_matrix = coordconv_matrix.astype(np.float32)

    coordconv_matrix = torch.FloatTensor(coordconv_matrix).unsqueeze(0)
    model.train()

    model.to(device)
    # example_tensor = torch.rand(1, 6, 240, 320)
    # print(example_tensor.shape)
    # state_e, state_d = model.init(example_tensor.to(device))
    # writer.add_graph(model, (example_tensor.to(device), state_e, state_d))
    # writer.add_graph(model, [example_tensor, state_e, state_d])
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []

    for epoch in range(EPOCHS): # Training loop
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0

        for x, y in train_data:
            model.zero_grad()
            outputs = []
            gaussian_map = y[:, 0, :, :, :]
            frame_del_video = x[:, 0, :, :, :]
            frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
            state_e, state_d = model.init(frame_with_coords)

            for t in range(x.shape[1]):
                frame_del_video = x[:, t, :, :, :] # Obtener el frame actual
                frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1) # Concatenar el frame actual con coordconv_matrix a lo largo de la dimensión de los canales
                out, state_e, state_d = model(frame_with_coords, state_e, state_d) # Pasar el frame concatenado a través del encoder y decoder
                if torch.max(out) == 0:
                    outputs.append(out)
                else:
                    outputs.append((out - torch.min(out)) / (torch.max(out) - torch.min(out)))
                # writer.add_image('train/frame_del_video', frame_del_video[0,:,:,:], global_step=epoch * len(train_data) + counter_train)
                # writer.add_image('train/gaussian_map', gaussian_map[0,:,:,:], global_step=epoch * len(train_data) + counter_train)
                # writer.add_image('train/tspm', out2.squeeze(0), global_step=epoch * len(train_data) + counter_train)
                with torch.no_grad():
                    if torch.max(out) == 0:
                        out_squeezed=out.squeeze() # Normalize map and squeeze
                    else:
                        out_squeezed=((out - torch.min(out)) / (torch.max(out) - torch.min(out))).squeeze() # Normalize map and squeeze
                    tSPM=out_squeezed.cpu().detach().numpy()
                    point = sg.generate_image_saliency_matrix_scanpath(tSPM, 1)
                    gaussian_map= gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[0],point[1]))
                    gaussian_map = gaussian_map.astype(np.float32)
                    gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)
            pred = torch.stack(outputs, dim=1)
            pred.squeeze(0)
            y.squeeze(0)

            # total_loss_DTW = 0
            # for i in range(0, pred.shape[1]-1): # [b_size(1), len, H, W]
            #     loss = criterion(pred[:, i, :, :, :], y[:, i+1, :, :, :].to(device))
            #     total_loss_DTW = total_loss_DTW + loss
            # total_loss_DTW = total_loss_DTW / pred.shape[1]
            total_loss_DTW = criterion(pred[:, -1, :, :, :], y[:, -1, :, :, :].to(device))

            total_loss_DTW.backward()
            print_gradient_summary(model,  writer, epoch)
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradiente para {name}: norma {param.grad.norm()}")
            #         if param.grad.norm() > 500:
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=500.0) # Aplicar clipping de gradiente ya que estos tienden a explotar
            #     else:
            #         print(f"Gradiente para {name}: None encontrado")
            #     # layer_name, param_name = name.split('.')
            #     # param.register_hook(get_gradients_hook(writer, layer_name, param_name, epoch))
            #     writer.add_histogram(f'params/{name}', param, epoch)

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
                frame_del_video = x[:, 0, :, :, :]
                frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                state_e, state_d = model.init(frame_with_coords)

                for t in range(x.shape[1]):
                    frame_del_video = x[:, t, :, :, :]
                    frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                    out, state_e, state_d = model(frame_with_coords, state_e, state_d)

                    if torch.max(out) == 0:
                        outputs.append(out)
                    else:
                        outputs.append((out - torch.min(out)) / (torch.max(out) - torch.min(out)))
                    with torch.no_grad():
                        if torch.max(out) == 0:
                            out_squeezed=out.squeeze() # Normalize map and squeeze
                        else:
                            out_squeezed=((out - torch.min(out)) / (torch.max(out) - torch.min(out))).squeeze() # Normalize map and squeeze
                        tSPM=out_squeezed.cpu().detach().numpy()
                        point = sg.generate_image_saliency_matrix_scanpath(tSPM, 1)
                        gaussian_map= gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[0],point[1]))
                        gaussian_map = gaussian_map.astype(np.float32)
                        gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)
                pred = torch.stack(outputs, dim=1)
                pred.squeeze(0)
                y.squeeze(0)

                # total_loss_DTW = 0
                # for i in range(0, pred.shape[1]-1): # [b_size(1), len, H, W]
                #     loss = criterion(pred[:, i, :, :, :], y[:, i+1, :, :, :].to(device))
                #     total_loss_DTW = total_loss_DTW + loss
                # total_loss_DTW = total_loss_DTW / pred.shape[1]
                total_loss_DTW = criterion(pred[:, -1, :, :, :], y[:, -1, :, :, :].to(device))
                avg_loss_val += total_loss_DTW
                counter_val += 1

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)

        if epoch % 50 == 0: # Save checkpoint and model every 50 epochs
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss_DTW,
            }, ckp_path + '/model.pt')

    torch.save(model, path + '/model.pth') # Save final model and checkpoints
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss_DTW,
            }, ckp_path + '/model.pt')
    writer.close()
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
    train_video360_dataset = RGB_with_GM(config.frames_dir, config.gt_dir, video_names_train, config.sequence_length, skip=20, split='train', resolution=config.resolution)
    val_video360_dataset = RGB_with_GM(config.frames_dir, config.gt_dir, video_names_train, config.sequence_length, skip=20, split='validation', resolution=config.resolution)

    train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=6, shuffle=False) # TODO: cambiar shuffle en prueba final
    val_data = DataLoader(val_video360_dataset, batch_size=config.batch_size, num_workers=6, shuffle=False)

    # print(model)
    # model = train_SST_SAL_original(train_data, val_data, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)
    model = train_scanpath_video_predictor(train_data, val_data, model, device, loss_KLDTW, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)

    print("Training finished")