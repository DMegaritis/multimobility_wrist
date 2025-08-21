import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys
from multimob.GSD.utils.network import Network


def tensor_data_loader(data, device, batch_size):
    """ Converting the numpy data into torch tensor format
      :param sliding_windows_data: NumPy array of shape (n_windows, 1, 3, window_size)
      :param batch_size: Integer
      :param device: cuda/cpu
      :return: Tensor Dataloader
      """
    tensor_x = torch.tensor(data, dtype=torch.float32, device=device)
    tensor_dataset = TensorDataset(tensor_x)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return tensor_dataloader


def main(data_path):
    # Load data from a NumPy file
    data = np.load(data_path)

    # Set device and batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # Create DataLoader
    num_batches = max(1, data.shape[0] // batch_size)
    dataloader = tensor_data_loader(data, device, batch_size)

    # Import model
    model = Network()

    # Load pre-trained model
    model_path = Path(__file__).parent.parent / "GSD" / "utils" / "model.p"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Store predictions
    all_predictions = torch.tensor([]).to(device)

    # Iterate over batches
    for batch_idx, batch in enumerate(dataloader):
        data_batch = batch[0].to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(data_batch))
            all_predictions = torch.cat((all_predictions, preds), dim=0)

    # Convert predictions to NumPy and return
    predictions_np = all_predictions.cpu().numpy().squeeze()
    print(predictions_np.tolist())  # Print so subprocess can capture it

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need to pass data and the sub process script")
        sys.exit(1)
    main(sys.argv[1])
