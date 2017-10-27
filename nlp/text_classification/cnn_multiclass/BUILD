
py_library(
    name = "dataset",
    srcs = ["dataset/dataset.py"]
)

py_library(
    name = "model",
    srcs = ["model.py"],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":dataset",
     ":model"]
)

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [":model"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":predict"]
)