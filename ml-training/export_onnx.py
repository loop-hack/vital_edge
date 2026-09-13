import torch
from src.models.anomaly_autoencoder import HrSpo2Autoencoder

def export_model():
    print("Initializing HrSpo2Autoencoder...")
    model = HrSpo2Autoencoder()
    
    # note : Once you train your model, load your weights here before exporting:
    # model.load_state_dict(torch.load('checkpoints/autoencoder.pt'))
    
    #Putting the model in evaluation mode (critical for onnx export)
    model.eval()

    #it Create dummy input: 1 batch, 60 time steps, 2 features (e.g., HR, SpO2)
    dummy_input = torch.randn(1, 60, 2)
    
    # We output the onnx file directly into the mobile app's model folder
    output_path = '../mobile-app/src/models/hr_spo2_autoencoder.onnx'
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=['input'], 
        output_names=['reconstruction'],
        # dynamic axes allow the model to accept variable batch sizes if needed
        dynamic_axes={'input': {0: 'batch'}, 'reconstruction': {0: 'batch'}}, 
        opset_version=17, 
        export_params=True 
    )
    
    print(" Model successfully exported and quantized for edge inference..")

if __name__ == '__main__':
    export_model()