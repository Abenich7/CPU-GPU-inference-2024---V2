import csv
############ TESTBENCH #################

#mapping of model names to model (its weights)
#ex: pytorch_quantized_int_8 -> model1.pth
 #   trt_full_precision -> model1.engine

 
@dataclass
class ExperimentConfig:
    model_name: str
    model_type: str        # 'pytorch' | 'trt'
    model_quantization:str # 'FP16' | 'INT8' | ...
    dataset_name: str
    dataset_size: int
    num_batches: int
    batch_size: int

models = {}    # key: (model_name, model_type, model_quantization) -> model_path

#csv format
#    model_name,model_type,model_quantization,model_path
#    resnet18,cnn,int8,/models/resnet18_int8.onnx

# ---- Load models ----
with open("models.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["model_name"],
                row["model_type"],
                row["model_quantization"]
            )
            models[key] = row["model_path"]

    
#save names->path mapping to csv


def define_model_dataset(
    model_name,
    model_type,
    model_quantization,
    dataset_name,
    
    batch_size,
):
    # ---------- MODEL ----------
    #based on model key return model path
    model_key = (model_name, model_type,model_quantization)

    model_path=models[model_key]
    
    
    #define dataloader
    dataloader=DataLoader(
           
            batch_size=batch_size,
           
        )


    return model_path, dataloader


# call inference functions 


def experiments_data(configurations):
    results = []

    for config in configurations:
        model_path, dataloader = define_model_dataset(
            config.model_name,
            config.model_type,
            config.model_quantization,
            config.batch_size,
        )

        if config.model_type == "pytorch":
            metrics = test_model(model_path, dataloader)

        elif config.model_type == "trt":
            metrics = trt_inference_script(model_path, dataloader)

        else:
            raise ValueError("Invalid model type")

        results.append({
            "model": config.model_name,
            "type": config.model_type,
            "quantization": config.model_quantization,
            "batch size": config.batch_size,
            **metrics,
        })

    return results


configs = [
    ExperimentConfig(
        model_name="resnet18_fp32",
        model_type="pytorch",
        model_quantization="",
        batch_size=32,
    ),
    ExperimentConfig(
        model_name="resnet18_fp32",
        model_type="trt",
        model_quantization='',
        batch_size=32,
    ),
]

results = experiments_data(configs)
print(results)
