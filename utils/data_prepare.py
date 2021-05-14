from PIL import Image
import torchvision.transforms as transforms
import torch


te_tf = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class pipeline():
    def __init__(self, custom_transform=None):
        if custom_transform is not None:
            self.tf  = custom_transform
        else:
            self.tf = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def preprocess(self, img_path):
        image_raw = Image.open(img_path)
        img = self.tf(image_raw)
        return img.numpy()
        
