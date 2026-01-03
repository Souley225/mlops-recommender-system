# Script pour convertir le modele vers les nouvelles classes
import joblib
import sys
sys.path.insert(0, '.')

# Import old classes first to enable loading
from src.models.train import PopularityModel as OldPop, ALSModel as OldALS

# Load model
model = joblib.load('models/model.joblib')
print(f'Loaded model type: {type(model).__module__}.{type(model).__name__}')

# Import new classes
from src.models.model_classes import PopularityModel, ALSModel

# Create new model with same data
if isinstance(model, OldALS) or hasattr(model, 'user_factors'):
    print('Converting ALSModel...')
    new_model = ALSModel(
        factors=model.factors,
        regularization=model.regularization,
        iterations=model.iterations,
        alpha=model.alpha,
        use_gpu=model.use_gpu,
        random_state=model.random_state
    )
    new_model.model = model.model
    new_model.user_factors = model.user_factors
    new_model.item_factors = model.item_factors
else:
    print('Converting PopularityModel...')
    new_model = PopularityModel()
    new_model.item_scores = model.item_scores
    new_model.n_items = model.n_items

# Save with new class
joblib.dump(new_model, 'models/model.joblib', compress=3)
print(f'Saved new model type: {type(new_model).__module__}.{type(new_model).__name__}')
print('Done!')
