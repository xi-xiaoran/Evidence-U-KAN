import torch
import time
import matplotlib.pyplot as plt

def Dice(A, B):
    smooth = 1e-8
    I = A * B
    T = A + B
    dice = 2 * I.sum() / (T.sum() + smooth)
    return dice


def Iou(A, B):
    I = A * B
    T = A + B
    U = T - I
    smooth = 1e-8
    iou = I.sum() / (U.sum() + smooth)
    return iou

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25, fold=0):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    X = []
    Y_train = []
    Y_val = []
    Val_Dice = []
    val_dice = 0
    Val_Iou = []
    val_iou = 0


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        X.append(epoch + 1)
        num = 1
        start_time = time.time()
        for inputs, labels in train_loader:
            num = num + 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        Y_train.append(epoch_loss)

        # Validation phase
        # model.eval()
        running_loss = 0.0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = torch.where(outputs > 0.5, 1, 0)
                val_dice = val_dice + Dice(outputs, labels).cpu().numpy()
                val_iou = val_iou + Iou(outputs, labels).cpu().numpy()


            running_loss +=loss.item() * inputs.size(0)
        val_dice = val_dice / len(val_loader)
        val_iou = val_iou / len(val_loader)
        Val_Dice.append(val_dice)
        Val_Iou.append(val_iou)

        epoch_loss = running_loss / len(val_loader.dataset)
        end_time = time.time()
        print(f'Validation Loss: {epoch_loss:.4f}')
        print(f'Val_Dice:{val_dice},Val_Iou:{val_iou}')
        print(f'time costs {end_time - start_time}s')
        Y_val.append(epoch_loss)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    color1 = (248 / 255, 230 / 255, 32 / 255)
    color2 = (53 / 255, 183 / 255, 119 / 255)
    plt.plot(X,Y_train,color=color1,label='train loss')
    plt.plot(X,Y_val,color=color2,label='val_loss')
    plt.legend()
    plt.savefig(f'{fold}_train_loss.png')
    plt.close()

    plt.plot(X, Val_Dice, color=color1, label='Val_Dice')
    plt.plot(X, Val_Iou, color=color2, label='Val_Iou')
    plt.legend()
    plt.savefig(f'{fold}_Val.png')
    plt.close()
    return model