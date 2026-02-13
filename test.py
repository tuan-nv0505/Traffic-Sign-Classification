import torch
from torchvision import transforms
import torch.nn.functional as F
from datasets.gtsrb_dataset import GTSRBDataset
from models.super_mamba import SuperMamba
from PIL import Image
import os
from utils import get_args, get_mean_and_std

args = get_args()

DEVICE = torch.device(args.device)
TRAINED = args.trained
SIZE = args.size
PATH_DATA = args.path_data
WORKERS = args.workers

def test(path_image):
    checkpoint = torch.load(
        os.path.join(TRAINED, "fold_1/best_checkpoint.pth"),
        map_location=DEVICE,
        weights_only=True
    )
    state_dict = checkpoint['model_state_dict']
    model = SuperMamba(dims=3, depth=4, num_classes=43).to(DEVICE)
    model.load_state_dict(state_dict)

    test_transforms = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.3417820930480957, 0.3126334846019745, 0.3216340243816376),
            std=(0.27580520510673523, 0.2633080780506134, 0.26914146542549133)
        )
    ])

    img = Image.open(path_image).convert('RGB')
    in_img = img
    in_img = test_transforms(in_img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(in_img)
        probabilities = F.softmax(output, dim=1)
        conf, class_id = torch.max(probabilities, 1)
        print(f"{class_id.item()} - {conf.item() * 100:.2f}%")
        img.show()

if __name__ == '__main__':
    # test("./data/gtsrb/Final_Test/Images/00004.ppm")
    test("./images/weather_rain_7/1.jpg")