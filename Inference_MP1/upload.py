from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="/mnt/data/home/dhruv/Desktop/VR_Mini_Project_Part1/Inference_MP1/VRMP1", repo_id="jdhr/VRMP1_IMT2023032_IMT2023075_IMT2023078_IMT2023507", repo_type="model")
