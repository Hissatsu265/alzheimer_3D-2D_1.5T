import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Label mapping: AD=0, CN=1, MCI=2
        label_mapping = {'AD_2D': 0, 'CN_2D': 1, 'MCI_2D': 2}
        
        # Loop through the root directory and collect images from each plane (axial, cogtinal, sagtinal)
        # for label in ['AD', 'CN', 'MCI']:
    # ---------------------------------------------------------------------
        for label in ['AD_2D', 'CN_2D', 'MCI_2D']:
            for plane in ['axial', 'coronal', 'sagittal']:
                plane_folder = os.path.join(root_dir, label, plane)
    # ---------------------------------------------------------------------
                for img_file in os.listdir(plane_folder):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(plane_folder, img_file))
                        self.labels.append(label_mapping[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open and convert to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

