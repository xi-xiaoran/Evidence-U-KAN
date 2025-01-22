import torch
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F


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


def train_model(model, train_loader, val_loader, criterion, optimizer, device, uncertainty=False, num_epochs=25,
                module=0, fold=0, likehood=0.3):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    X = []
    Y_train = []
    Y_val = []
    Test_Dice = []
    Test_Iou = []
    Val_Dice = []
    val_dice = 0
    Val_Iou = []
    val_iou = 0

    Train_Dice = []
    train_dice = 0
    Train_Iou = []
    train_iou = 0
    fire = int(num_epochs / 5)
    # fire = int(num_epochs + 1)
    # fire = 2
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        X.append(epoch + 1)
        num = 1
        start_time = time.time()
        train_len = 0
        train_dice = 0
        train_iou = 0
        for inputs, labels in train_loader:
            if epoch < fire:
                num = num + 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels, epoch)
                loss.backward()
                optimizer.step()
            else:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                a = outputs + 1
                S = torch.sum(a, dim=1, keepdim=True)
                u = 2 / S
                u_last = u
                diedai = 0
                while (True):
                    diedai = diedai + 1
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    a = outputs + 1
                    S = torch.sum(a, dim=1, keepdim=True)
                    u = 2 / S
                    like = torch.mean(u_last - u).detach().cpu().numpy()
                    like = abs(like)
                    if (like < likehood):
                        break
                    elif diedai > 100:
                        break
                    else:
                        u_last = u
                loss = criterion(outputs, labels, epoch)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if uncertainty:
                alpha = outputs + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                P = alpha / S
                P = P.detach()
                P = torch.argmax(P,dim=1,keepdim=True)
            else:
                outputs = torch.where(outputs > 0.5, 1, 0)
            train_dice = train_dice + Dice(P, labels).cpu().numpy()
            train_iou = train_iou + Iou(P, labels).cpu().numpy()
            train_len = train_len + 1

        model.clear()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        Y_train.append(epoch_loss)

        # Validation phase
        # model.eval()
        running_loss = 0.0
        val_len = 0
        val_dice = 0
        val_iou = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                if epoch < fire:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels, epoch)
                    # loss = criterion(outputs, labels)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    a = outputs + 1
                    S = torch.sum(a, dim=1, keepdim=True)
                    u = 2 / S
                    u_last = u
                    diedai = 0
                    while (True):
                        diedai = diedai + 1

                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        a = outputs + 1
                        S = torch.sum(a, dim=1, keepdim=True)
                        u = 2 / S
                        like = torch.mean(u_last - u).cpu().numpy()
                        like = abs(like)
                        if (like < likehood):
                            break
                        elif diedai > 100:
                            break
                        else:
                            u_last = u
                if uncertainty:
                    alpha = outputs + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    P = outputs / S
                    P = P.detach()
                    P = torch.argmax(P,dim=1,keepdim=True)
                else:
                    outputs = torch.where(outputs > 0.5, 1, 0)
                val_dice = val_dice + Dice(P, labels).cpu().numpy()
                val_iou = val_iou + Iou(P, labels).cpu().numpy()
                val_len = val_len + 1
            running_loss += loss.item() * inputs.size(0)
        model.clear()

        val_dice = val_dice / val_len
        val_iou = val_iou / val_len
        train_dice = train_dice / train_len
        train_iou = train_iou / train_len
        Val_Dice.append(val_dice)
        Val_Iou.append(val_iou)

        Train_Dice.append(train_dice)
        Train_Iou.append(train_iou)

        epoch_loss = running_loss / len(val_loader.dataset)
        end_time = time.time()
        print(f'Validation Loss: {epoch_loss:.4f}')
        print(f'Val_Dice:{val_dice},Val_Iou:{val_iou}')
        print(f'Train_Dice:{train_dice},Train_Iou:{train_iou}')
        print(f'time costs {end_time - start_time}s')
        Y_val.append(epoch_loss)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    color1 = (248 / 255, 230 / 255, 32 / 255)
    color2 = (53 / 255, 183 / 255, 119 / 255)
    plt.plot(X, Y_train, color=color1, label='train loss')
    plt.plot(X, Y_val, color=color2, label='val_loss')
    plt.legend()
    plt.savefig(f'{fold}_train_loss.png')
    plt.close()

    plt.plot(X, Val_Dice, color=color1, label='Val_Dice')
    plt.plot(X, Val_Iou, color=color2, label='Val_Iou')
    plt.plot(X, Train_Dice, color='r', label='Train_Dice')
    plt.plot(X, Train_Iou, color='blue', label='Train_Iou')
    plt.legend()
    plt.savefig(f'{fold}_Val.png')
    plt.close()
    return model