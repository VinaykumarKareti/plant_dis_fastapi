from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
import torch
import pickle
app = FastAPI()


with open('resnet18_model12_epoch_7.pkl', 'rb') as f:
    model13 = pickle.load(f)
model13.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_labels =['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def get_class_label(class_index):
    return class_labels[class_index]

def predict_image(img):
    image = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model13(image)
        _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()
    class_label = get_class_label(class_index)
    return {"class_index": class_index, "class_label": class_label}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the image file
    img = Image.open(file.file)
    # Make prediction
    prediction = predict_image(img)
    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

