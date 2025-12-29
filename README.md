# NCF

### A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17

Note that I use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data).

I randomly utilized a factor number **32**, MLP layers **3**, epochs is **20**, and posted the results in the original paper and this implementation here. I employed the **exactly same settings** with Xiangnan, including batch_size, learning rate, and all the initialization methods in Xiangnan's keras repo. From the results I observed, this repo can replicate the performance of the original NCF.
Xiangnan's keras [repo](https://github.com/hexiangnan/neural_collaborative_filtering):

| Models                       | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10 |
| ---------------------------- | --------------- | ----------------- | --------------- | ----------------- |
| MLP                          | 0.692           | 0.425             | 0.868           | 0.542             |
| GMF                          | -               | -                 | -               | -                 |
| NeuMF (without pre-training) | 0.701           | 0.425             | 0.870           | 0.549             |
| NeuMF (with pre-training)    | 0.726           | 0.445             | 0.879           | 0.555             |

This pytorch code:

| Models                       | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10 |
| ---------------------------- | --------------- | ----------------- | --------------- | ----------------- |
| MLP                          | 0.691           | 0.416             | 0.866           | 0.537             |
| GMF                          | 0.708           | 0.429             | 0.867           | 0.546             |
| NeuMF (without pre-training) | 0.701           | 0.424             | 0.867           | 0.544             |
| NeuMF (with pre-training)    | 0.720           | 0.439             | 0.879           | 0.555             |

## The requirements are as follows:

    * python==3.6
    * pandas==0.24.2
    * numpy==1.16.2
    * pytorch==1.0.1
    * gensim==3.7.1
    * tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Inference API

A FastAPI-based inference service is provided for making predictions and recommendations using the trained NCF model.

### Installation

Install the API dependencies:

```bash
pip install -r requirements_api.txt
```

### Running the API

Start the inference server:

```bash
python inference.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

Returns the health status of the API and model information.

#### 2. Single Prediction

```bash
POST /predict
```

Predict interaction score for a single user-item pair.

**Request Body:**

```json
{
  "user_id": 0,
  "item_id": 100
}
```

**Response:**

```json
{
  "user_id": 0,
  "item_id": 100,
  "score": 0.85,
  "message": "Prediction successful"
}
```

#### 3. Batch Prediction

```bash
POST /predict/batch
```

Predict scores for a user and multiple items.

**Request Body:**

```json
{
  "user_id": 0,
  "item_ids": [100, 200, 300]
}
```

**Response:**

```json
{
  "user_id": 0,
  "predictions": [
    { "item_id": 300, "score": 0.92 },
    { "item_id": 100, "score": 0.85 },
    { "item_id": 200, "score": 0.78 }
  ],
  "message": "Batch prediction successful"
}
```

#### 4. Recommendations

```bash
POST /recommend
```

Get top-K item recommendations for a user.

**Request Body:**

```json
{
  "user_id": 0,
  "k": 10,
  "candidate_item_ids": null
}
```

**Response:**

```json
{
  "user_id": 0,
  "recommendations": [
    {"item_id": 1234, "score": 0.95},
    {"item_id": 5678, "score": 0.92},
    ...
  ],
  "k": 10,
  "message": "Recommendations generated successfully"
}
```

### Example Usage

#### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "item_id": 100}'

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "k": 10}'
```

#### Using Python:

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"user_id": 0, "item_id": 100}
)
print(response.json())

# Get recommendations
response = requests.post(
    "http://localhost:8000/recommend",
    json={"user_id": 0, "k": 10}
)
print(response.json())
```

### Configuration

The API configuration can be modified in `inference.py`:

- `MODEL_PATH`: Path to the trained model file
- `DEVICE`: Device to use ('cpu' or 'cuda')
- `USER_NUM`: Number of users in the dataset
- `ITEM_NUM`: Number of items in the dataset
