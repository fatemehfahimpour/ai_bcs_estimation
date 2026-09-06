import torch
from ResNet_model import BCSResNet18
from preprocess import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "model_results/best_resnet18.pth"
CRITERION = torch.nn.CrossEntropyLoss()


def test_model():
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    trainable_layers = checkpoint["trainable_layers"]
    print(f"Trainable layers: {trainable_layers}")

    model = BCSResNet18(trainable_layers=trainable_layers).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_data_loader()
    # testing
    criterion = CRITERION.to(DEVICE)

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image, labels in test_loader:
            image = image.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            loss = criterion(outputs, labels)

            batch_size = image.shape[0]
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += batch_size

        test_loss = total_loss / total_samples
        test_accuracy = (correct_predictions / total_samples)

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        

if __name__ == "__main__":
    test_model()
