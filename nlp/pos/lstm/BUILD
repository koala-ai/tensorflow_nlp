py_library(
    name = "dataset",
    srcs = ["dataset/dataset.py"]
)

py_library(
    name = "rawdata",
    srcs = ["dataset/rawdata.py"]
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
     ":rawdata",
     ":model",
     ":bilstm_model"]
)

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [":model", ":rawdata", ":bilstm_model"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":predict"]
)