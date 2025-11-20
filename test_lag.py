#type: ignore
import torch
from huggingface_hub import hf_hub_download
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from gluonts.evaluation import make_evaluation_predictions 
from gluonts.dataset.repository.datasets import get_dataset 
from lag_llama.gluon.estimator import LagLlamaEstimator 

# Download model from HuggingFace
model_id = "time-series-foundation-models/Lag-Llama"
ckpt_path = hf_hub_download(
    repo_id=model_id,
    filename="lag-llama.ckpt",
    cache_dir="./cache"
)

dataset = get_dataset("australian_electricity_demand") 
backtest_dataset = dataset.test 
prediction_length = dataset.metadata.prediction_length 
context_length = 3 * prediction_length 

torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

ckpt = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
estimator_args = ckpt["hyper_parameters"]["model_kwargs"] 

estimator = LagLlamaEstimator( 
    ckpt_path=ckpt_path,
    prediction_length=prediction_length, 
    context_length=context_length, 
    input_size=estimator_args["input_size"], 
    n_layer=estimator_args["n_layer"], 
    n_embd_per_head=estimator_args["n_embd_per_head"], 
    n_head=estimator_args["n_head"], 
    scaling=estimator_args["scaling"], 
    time_feat=estimator_args["time_feat"]
) 

lightning_module = estimator.create_lightning_module() 
transformation = estimator.create_transformation() 
predictor = estimator.create_predictor(transformation, lightning_module) 

forecast_it, ts_it = make_evaluation_predictions(
    dataset=backtest_dataset,  # type: ignore
    predictor=predictor) 

forecasts = list(forecast_it) 
tss = list(ts_it)

print(forecasts)
print(tss)
