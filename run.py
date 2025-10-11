import os
import pandas as pd
import pickle
import torch
from torchvision import models
from scipy.sparse import hstack, csr_matrix
from services.Text_Model import clean_text
from services.Image_model import load_image,extract_embedding

# LOAD SAVED MODELS / OBJECTS
# Load TF-IDF vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load LightGBM model
import lightgbm as lgb
model = lgb.Booster(model_file="models/lgb_model.txt")

# Load pretrained CNN for image embeddings (same as training)
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# PREDICTOR FUNCTION
def predictor(sample_id, catalog_content, image_link):
    print(sample_id)
    '''
    Call your model/approach here
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    #Text Embedding
    clean_txt = clean_text(catalog_content)
    text_vec = vectorizer.transform([clean_txt])

    #Image Embedding
    img_tensor = load_image(image_link)
    img_vec = extract_embedding(img_tensor=img_tensor)
    img_vec = csr_matrix(img_vec.reshape(1, -1))

    #Combine
    combined = hstack([text_vec, img_vec])

    #Predict
    pred_price = model.predict(combined)[0]
    pred_price = max(0.0, float(pred_price))  # enforce non-negative price
    return round(pred_price, 2)



if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    OUTPUT_FOLDER = 'outputs/'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']),
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(OUTPUT_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
