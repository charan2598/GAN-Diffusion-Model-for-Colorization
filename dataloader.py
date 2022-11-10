from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets

# Batch Size
batch_size = 4

# define necessary transforms
transform = transforms.Compose([transforms.ToTensor()])

# Define dataset class or object of CIFAR
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transforms = transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transforms = transform)

# Define the dataloader for train and test
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

