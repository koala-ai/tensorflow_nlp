py_library(
    name = "dataset",
    srcs = ["dataset/dataset.py"]
)

py_library(
    name = "data_utils",
    srcs = ["dataset/data_utils.py"]
)

py_library(
    name = "model",
    srcs = ["model.py"]
)

py_library(
    name = "bilstm_model",
    srcs = ["bilstm_model.py"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":dataset",
     ":data_utils",
     ":model"]
)

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [":model", ":dataset", ":data_utils"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":predict"]
)