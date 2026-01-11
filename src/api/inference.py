"""
Recommendation System Inference API
A FastAPI-based inference service for NCF and AutoRec models.
"""
import os
import sys

# Setup path
current_file_path = os.path.abspath(__file__)
api_dir = os.path.dirname(current_file_path)
src_path = os.path.dirname(api_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import models
from helpers import NCF
from autorec.utils.model import AutoRec

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
NCF_MODEL_PATH = "models/NeuMF.pth"
AUTOREC_MODEL_PATH = "models/AutoRec-best.pth"
TRAIN_DATA_PATH = "data/ml-1m.train.rating"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NCF Configuration
NCF_CONFIG = {
    "user_num": 6038,
    "item_num": 3533,
    "factor_num": 32,
    "num_layers": 3,
    "dropout": 0.0
}

# AutoRec Configuration
AUTOREC_CONFIG = {
    "user_num": 6040,
    "item_num": 3706,
    "hidden_units": 500,
    "item_based": True
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_ncf_model(model_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load NCF model from checkpoint."""
    if not os.path.exists(model_path):
        print(f"⚠ Warning: NCF model not found at {model_path}")
        return None
    
    try:
        print(f"Loading NCF model from {model_path}...")
        model = torch.load(model_path, weights_only=False)
        model.eval()
        print(f"✓ NCF model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"✗ Error loading NCF model: {e}")
        return None

def load_autorec_model(model_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load AutoRec model from checkpoint."""
    if not os.path.exists(model_path):
        print(f"⚠ Warning: AutoRec model not found at {model_path}")
        return None
    
    try:
        print(f"Loading AutoRec model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = AutoRec(
            num_users=AUTOREC_CONFIG["user_num"],
            num_items=AUTOREC_CONFIG["item_num"],
            num_hidden_units=AUTOREC_CONFIG["hidden_units"],
            item_based=AUTOREC_CONFIG["item_based"]
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A')
        val_rmse = checkpoint.get('val_rmse', 'N/A')
        print(f"✓ AutoRec model loaded successfully on {device}")
        if isinstance(val_rmse, float):
            print(f"  Epoch: {epoch}, Val RMSE: {val_rmse:.6f}")
        else:
            print(f"  Epoch: {epoch}, Val RMSE: {val_rmse}")
        return model
    except Exception as e:
        print(f"✗ Error loading AutoRec model: {e}")
        return None

def load_training_matrix(data_path: str) -> Optional[np.ndarray]:
    """Load training rating matrix for AutoRec inference.
    
    Note: The training file may only contain user-item pairs (no ratings).
    In that case, we load from the original ratings.dat file.
    """
    # First, try to load from the original ratings.dat file
    ratings_file = "data/ml-1m/ratings.dat"
    
    if os.path.exists(ratings_file):
        try:
            print(f"Loading ratings from original file: {ratings_file}...")
            ratings_df = pd.read_csv(
                ratings_file,
                sep='::',
                header=None,
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python',
                dtype={
                    'user_id': np.int32,
                    'item_id': np.int32,
                    'rating': np.float32,
                    'timestamp': np.int32
                }
            )
            
            # Rename columns for consistency
            ratings_df = ratings_df.rename(columns={'user_id': 'user', 'item_id': 'item'})
            
            # Get max IDs to determine matrix size
            max_user = ratings_df['user'].max()
            max_item = ratings_df['item'].max()
            
            # Remap to 0-indexed if needed
            unique_users = sorted(ratings_df['user'].unique())
            unique_items = sorted(ratings_df['item'].unique())
            
            if unique_users != list(range(len(unique_users))):
                user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
                ratings_df['user'] = ratings_df['user'].map(user_map)
            
            if unique_items != list(range(len(unique_items))):
                item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
                ratings_df['item'] = ratings_df['item'].map(item_map)
            
            # Create training matrix
            train_mat = np.zeros(
                (AUTOREC_CONFIG["user_num"], AUTOREC_CONFIG["item_num"]),
                dtype=np.float32
            )
            
            for _, row in ratings_df.iterrows():
                user_id = int(row['user'])
                item_id = int(row['item'])
                rating = float(row['rating'])
                
                if (0 <= user_id < AUTOREC_CONFIG["user_num"] and
                    0 <= item_id < AUTOREC_CONFIG["item_num"]):
                    train_mat[user_id, item_id] = rating
            
            num_ratings = np.count_nonzero(train_mat)
            print(f"✓ Training matrix loaded: shape {train_mat.shape}, {num_ratings} ratings")
            return train_mat
            
        except Exception as e:
            print(f"⚠ Error loading from ratings.dat: {e}")
            print(f"  Falling back to training file: {data_path}")
    
    # Fallback: try loading from training file (may only have user-item pairs)
    if os.path.exists(data_path):
        try:
            print(f"Loading training data from {data_path}...")
            train_data = pd.read_csv(
                data_path,
                sep='\t',
                header=None,
                names=['user', 'item'],
                usecols=[0, 1],
                dtype={0: np.int32, 1: np.int32}
            )
            
            train_mat = np.zeros(
                (AUTOREC_CONFIG["user_num"], AUTOREC_CONFIG["item_num"]),
                dtype=np.float32
            )
            
            # If no ratings, set to 1.0 (binary interaction)
            for _, row in train_data.iterrows():
                user_id = int(row['user'])
                item_id = int(row['item'])
                
                if (0 <= user_id < AUTOREC_CONFIG["user_num"] and
                    0 <= item_id < AUTOREC_CONFIG["item_num"]):
                    train_mat[user_id, item_id] = 1.0  # Default rating
            
            num_ratings = np.count_nonzero(train_mat)
            print(f"✓ Training matrix loaded (binary): shape {train_mat.shape}, {num_ratings} interactions")
            print(f"  ⚠ Warning: Using default rating of 1.0 (no ratings in file)")
            return train_mat
            
        except Exception as e:
            print(f"✗ Error loading training matrix: {e}")
            return None
    
    print(f"✗ No training data file found")
    return None
# Initialize models
print("=" * 70)
print("Initializing Models")
print("=" * 70)

ncf_model = load_ncf_model(NCF_MODEL_PATH, DEVICE)
autorec_model = load_autorec_model(AUTOREC_MODEL_PATH, DEVICE)
autorec_train_mat = load_training_matrix(TRAIN_DATA_PATH)
autorec_train_tensor = (
    torch.Tensor(autorec_train_mat).to(DEVICE)
    if autorec_train_mat is not None else None
)

print("=" * 70)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Recommendation System Inference API",
    description="Neural Collaborative Filtering and AutoRec model inference API",
    version="1.0.0"
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for single prediction"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_id: int = Field(..., ge=0, description="Item ID (0-indexed)")

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    user_id: int
    item_id: int
    score: float
    message: str = "Prediction successful"

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_ids: List[int] = Field(..., min_items=1, description="List of item IDs")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    user_id: int
    predictions: List[Dict[str, Any]]
    message: str = "Batch prediction successful"

class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    candidate_item_ids: Optional[List[int]] = Field(
        None, description="Optional candidate items. If None, uses all items."
    )

class RecommendationResponse(BaseModel):
    """Response for recommendations"""
    user_id: int
    recommendations: List[Dict[str, Any]]
    k: int
    message: str = "Recommendations generated successfully"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ncf_loaded: bool
    autorec_loaded: bool
    device: str
    ncf_config: Dict[str, int]
    autorec_config: Dict[str, Any]

# ============================================================================
# NCF INFERENCE FUNCTIONS
# ============================================================================

def validate_ncf_user(user_id: int):
    """Validate NCF user ID"""
    if not (0 <= user_id < NCF_CONFIG["user_num"]):
        raise HTTPException(
            status_code=400,
            detail=f"User ID must be in [0, {NCF_CONFIG['user_num']}), got {user_id}"
        )

def validate_ncf_item(item_id: int):
    """Validate NCF item ID"""
    if not (0 <= item_id < NCF_CONFIG["item_num"]):
        raise HTTPException(
            status_code=400,
            detail=f"Item ID must be in [0, {NCF_CONFIG['item_num']}), got {item_id}"
        )

def ncf_predict(user_id: int, item_id: int) -> float:
    """Predict score using NCF model"""
    if ncf_model is None:
        raise HTTPException(status_code=503, detail="NCF model not loaded")
    
    validate_ncf_user(user_id)
    validate_ncf_item(item_id)
    
    with torch.no_grad():
        user_t = torch.LongTensor([user_id]).to(DEVICE)
        item_t = torch.LongTensor([item_id]).to(DEVICE)
        score = ncf_model(user_t, item_t).cpu().item()
    
    return score

def ncf_predict_batch(user_id: int, item_ids: List[int]) -> List[Dict[str, Any]]:
    """Batch predict using NCF model"""
    if ncf_model is None:
        raise HTTPException(status_code=503, detail="NCF model not loaded")
    
    validate_ncf_user(user_id)
    for item_id in item_ids:
        validate_ncf_item(item_id)
    
    with torch.no_grad():
        user_t = torch.LongTensor([user_id] * len(item_ids)).to(DEVICE)
        item_t = torch.LongTensor(item_ids).to(DEVICE)
        print(f"User tensor: {user_t}")
        print(f"Item tensor: {item_t}")
        scores = ncf_model(user_t, item_t).cpu().numpy()
        print(f"Scores: {scores}")
    predictions = [
        {"item_id": int(iid), "score": float(score)}
        for iid, score in zip(item_ids, scores)
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)
    
    return predictions

def ncf_recommend(
    user_id: int,
    k: int = 10,
    candidate_item_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Get recommendations using NCF model"""
    if ncf_model is None:
        raise HTTPException(status_code=503, detail="NCF model not loaded")
    
    validate_ncf_user(user_id)
    
    if candidate_item_ids is None:
        candidate_item_ids = list(range(NCF_CONFIG["item_num"]))
        print(f"No candidate item ids provided, using all items")
        print(f"Candidate item ids: {candidate_item_ids}")
    else:
        for item_id in candidate_item_ids:
            validate_ncf_item(item_id)
        candidate_item_ids = list(set(candidate_item_ids))
    
    if not candidate_item_ids:
        raise HTTPException(status_code=400, detail="No valid candidate items")
    
    k = min(k, len(candidate_item_ids))
    
    with torch.no_grad():
        user_t = torch.LongTensor([user_id] * len(candidate_item_ids)).to(DEVICE)
        item_t = torch.LongTensor(candidate_item_ids).to(DEVICE)
        print(f"User tensor: {user_t}")
        print(f"Item tensor: {item_t}")
        scores = ncf_model(user_t, item_t).cpu().numpy()
    
    top_k_idx = np.argsort(scores)[::-1][:k]
    
    return [
        {"item_id": int(candidate_item_ids[i]), "score": float(scores[i])}
        for i in top_k_idx
    ]

# ============================================================================
# AUTOREC INFERENCE FUNCTIONS
# ============================================================================

def validate_autorec_user(user_id: int):
    """Validate AutoRec user ID"""
    if not (0 <= user_id < AUTOREC_CONFIG["user_num"]):
        raise HTTPException(
            status_code=400,
            detail=f"User ID must be in [0, {AUTOREC_CONFIG['user_num']}), got {user_id}"
        )

def validate_autorec_item(item_id: int):
    """Validate AutoRec item ID"""
    if not (0 <= item_id < AUTOREC_CONFIG["item_num"]):
        raise HTTPException(
            status_code=400,
            detail=f"Item ID must be in [0, {AUTOREC_CONFIG['item_num']}), got {item_id}"
        )

def autorec_predict(user_id: int, item_id: int) -> float:
    """Predict score using AutoRec model"""
    if autorec_model is None or autorec_train_tensor is None:
        raise HTTPException(
            status_code=503,
            detail="AutoRec model or training matrix not loaded"
        )
    
    validate_autorec_user(user_id)
    validate_autorec_item(item_id)
    
    with torch.no_grad():
        if AUTOREC_CONFIG["item_based"]:
            # Item-based: get item vector (column), predict for all users
            item_vec = autorec_train_tensor[:, item_id].unsqueeze(0)
            reconstructed = autorec_model(item_vec)
            reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
            score = reconstructed[0, user_id].cpu().item()
        else:
            # User-based: get user vector (row), predict for all items
            user_vec = autorec_train_tensor[user_id, :].unsqueeze(0)
            reconstructed = autorec_model(user_vec)
            reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
            score = reconstructed[0, item_id].cpu().item()
    
    return score

def autorec_predict_batch(user_id: int, item_ids: List[int]) -> List[Dict[str, Any]]:
    """Batch predict using AutoRec model"""
    if autorec_model is None or autorec_train_tensor is None:
        raise HTTPException(
            status_code=503,
            detail="AutoRec model or training matrix not loaded"
        )
    
    validate_autorec_user(user_id)
    for item_id in item_ids:
        validate_autorec_item(item_id)
    
    predictions = []
    
    with torch.no_grad():
        if AUTOREC_CONFIG["item_based"]:
            for item_id in item_ids:
                item_vec = autorec_train_tensor[:, item_id].unsqueeze(0)
                reconstructed = autorec_model(item_vec)
                print(f"Reconstructed: {reconstructed}")
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                score = reconstructed[0, user_id].cpu().item()
                print(f"Score: {score}")
                predictions.append({"item_id": int(item_id), "score": float(score)})
        else:
            user_vec = autorec_train_tensor[user_id, :].unsqueeze(0)
            reconstructed = autorec_model(user_vec)
            reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
            for item_id in item_ids:
                score = reconstructed[0, item_id].cpu().item()
                predictions.append({"item_id": int(item_id), "score": float(score)})
    
    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions

def autorec_recommend(
    user_id: int,
    k: int = 10,
    candidate_item_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Get recommendations using AutoRec model"""
    if autorec_model is None or autorec_train_tensor is None:
        raise HTTPException(
            status_code=503,
            detail="AutoRec model or training matrix not loaded"
        )
    
    validate_autorec_user(user_id)
    
    if candidate_item_ids is None:
        candidate_item_ids = list(range(AUTOREC_CONFIG["item_num"]))
    else:
        for item_id in candidate_item_ids:
            validate_autorec_item(item_id)
        candidate_item_ids = list(set(candidate_item_ids))
    
    if not candidate_item_ids:
        raise HTTPException(status_code=400, detail="No valid candidate items")
    
    k = min(k, len(candidate_item_ids))
    predictions = autorec_predict_batch(user_id, candidate_item_ids)
    
    return predictions[:k]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Recommendation System Inference API",
        "version": "1.0.0",
        "models": ["NCF", "AutoRec"],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if (ncf_model or autorec_model) else "unhealthy",
        ncf_loaded=ncf_model is not None,
        autorec_loaded=autorec_model is not None,
        device=DEVICE,
        ncf_config={
            "user_num": NCF_CONFIG["user_num"],
            "item_num": NCF_CONFIG["item_num"]
        },
        autorec_config={
            "user_num": AUTOREC_CONFIG["user_num"],
            "item_num": AUTOREC_CONFIG["item_num"]
        }
    )

# NCF Endpoints
@app.post("/ncf/predict", response_model=PredictionResponse, tags=["NCF"])
async def ncf_predict_endpoint(request: PredictionRequest):
    """Predict score for user-item pair using NCF"""
    try:
        score = ncf_predict(request.user_id, request.item_id)
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            score=score
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/ncf/predict/batch", response_model=BatchPredictionResponse, tags=["NCF"])
async def ncf_predict_batch_endpoint(request: BatchPredictionRequest):
    """Batch predict scores using NCF"""
    try:
        predictions = ncf_predict_batch(request.user_id, request.item_ids)
        return BatchPredictionResponse(
            user_id=request.user_id,
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/ncf/recommend", response_model=RecommendationResponse, tags=["NCF"])
async def ncf_recommend_endpoint(request: RecommendationRequest):
    """Get top-K recommendations using NCF"""
    try:
        recommendations = ncf_recommend(
            request.user_id,
            request.k,
            request.candidate_item_ids
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            k=len(recommendations)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# AutoRec Endpoints
@app.post("/autorec/predict", response_model=PredictionResponse, tags=["AutoRec"])
async def autorec_predict_endpoint(request: PredictionRequest):
    """Predict rating for user-item pair using AutoRec"""
    try:
        score = autorec_predict(request.user_id, request.item_id)
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            score=score
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/autorec/predict/batch", response_model=BatchPredictionResponse, tags=["AutoRec"])
async def autorec_predict_batch_endpoint(request: BatchPredictionRequest):
    """Batch predict ratings using AutoRec"""
    try:
        predictions = autorec_predict_batch(request.user_id, request.item_ids)
        return BatchPredictionResponse(
            user_id=request.user_id,
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/autorec/recommend", response_model=RecommendationResponse, tags=["AutoRec"])
async def autorec_recommend_endpoint(request: RecommendationRequest):
    """Get top-K recommendations using AutoRec"""
    try:
        recommendations = autorec_recommend(
            request.user_id,
            request.k,
            request.candidate_item_ids
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            k=len(recommendations)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Recommendation System Inference API")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"NCF Model: {NCF_MODEL_PATH} ({'✓' if ncf_model else '✗'})")
    print(f"AutoRec Model: {AUTOREC_MODEL_PATH} ({'✓' if autorec_model else '✗'})")
    print("=" * 70)
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )