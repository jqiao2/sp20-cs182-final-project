import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import os

from model import Net
from datetime import datetime
from torch import nn

def main():
    # Create pytorch datatset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create training data generator
    batch_size = 10
    im_height = 64
    im_width = 64
    num_epochs = 10
    use_pretrained = True
    efficientnet_version = 'efficientnet-b4'
    input_size = Net.input_sizes[efficientnet_version]
    
    # Add data augmentation and norm for training
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    # Initialize model and put on GPU
    ENet = Net(efficientnet_version, num_classes=200, feature_extract=False, use_pretrained=use_pretrained)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ENet.model_ft.to(device)
    if False:
        ckpts = [x for x in os.listdir('ckpts/')]
        best = max(ckpts, key=lambda x: int(x.split('@')[0]))
        model.load_state_dict(torch.load('ckpts/' + best)['net'])
        print("continuing to train ", best)
    criterion = nn.CrossEntropyLoss()
    
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

    best_acc = -float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch+1))
        print('-' * 10)

        epoch_start = time.time()
        for phase in ['train', 'val']:
            phase_start = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            total, correct = 0, 0
            for idx, (inputs, targets) in enumerate(loaders[phase]):
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero out parameter gradients
                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # backward step iff training
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                    
                    _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print('\r', end='')
                print(f'\t{"Training" if phase=="train" else "Val"} {100 * idx / len(loaders[phase]):.2f}%: {correct / total:.3f}', end='')

            print(f'\t{"Training" if phase=="train" else "Val"} complete in {time.time() - phase_start:.0f}[s]')
            print('\n')
            if phase == 'val' and (correct / total) > best_acc:
                best_acc = correct / total
                torch.save({
                    'name': efficientnet_version,
                    'net': model.state_dict(),
                    'val_acc': best_acc,
                }, 'ckpts/{:.0f}@{}~{}.pt'.format(
                    best_acc*100,
                    efficientnet_version,
                    datetime.strftime(datetime.now(), "%X~%m.%d.%y")))

if __name__ == '__main__':
    main()
