# VASP Parameter Prediction Model ✨

A lightweight (~10MB) pre-trained Transformer model for predicting VASP parameters from crystal structures.

![Banner](https://th.bing.com/th/id/OIP.NCjRipLKkVX0Q-RvQSS7KgHaEK?pid=ImgDet&w=474&h=266&rs=1)

## 🌟 Overview
This is a pre-trained model designed to predict VASP parameters (e.g., ENCUT, ISMEAR, SIGMA) based on crystal structure data. It’s perfect for researchers automating quantum mechanics calculations!

- **Size**: ~10MB
- **Inputs**: Structural and electronic features (e.g., lattice, band gap)
- **Outputs**: VASP parameters (6 regression + 3 classification)
- **Trained on**: Materials Project data

## 🚀 How to Use

### Prerequisites
- Python 3.8+
- PyTorch (`pip install torch`)
- pymatgen (`pip install pymatgen`)
- scikit-learn (`pip install scikit-learn`)

### Download
Clone or download this repository:
```bash
git clone https://github.com/cooperhuang1/VASPModel.git
cd VASPModel


Example Code
Load and use the model with a POSCAR file:
import torch
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import pickle

# Load the model
checkpoint = torch.load("vasp_model.pth", map_location="cpu")
model = checkpoint['model_state_dict']  # Requires custom model class
scaler_X_struct = checkpoint['scaler_X_struct']
scaler_X_elec = checkpoint['scaler_X_elec']
scaler_y_reg = checkpoint['scaler_y_reg']

# Load your POSCAR
poscar = Poscar.from_file("example/POSCAR_example")
structure = poscar.structure

# Extract features (you’ll need to define this function)
struct_features, elec_features = extract_features(structure)  # Placeholder

# Scale features
struct_scaled = scaler_X_struct.transform([struct_features])
elec_scaled = scaler_X_elec.transform([elec_features])

# Predict
struct_tensor = torch.tensor(struct_scaled, dtype=torch.float32)
elec_tensor = torch.tensor(elec_scaled, dtype=torch.float32)
with torch.no_grad():
    reg_pred, ismear_pred, ibrion_pred, algo_pred = model(struct_tensor, elec_tensor)
    reg_pred = scaler_y_reg.inverse_transform(reg_pred.numpy())[0]
    print(f"ENCUT: {reg_pred[0]}, ISMEAR: {torch.argmax(ismear_pred).item()}")
Note: You’ll need to define extract_features and the MultiSeqVASPTransformer class based on your original code.

📁 Files
vasp_model.pth: Pre-trained model with scalers
example/POSCAR_example: Sample input (optional)
👤 Author
Junxin Huang

Chengdu University of Technology, Research Center for Planetary Science
Email: huangjunxin167@gmail.com
WeChat: h2005827723
📜 License
MIT License (free to use and modify!)

