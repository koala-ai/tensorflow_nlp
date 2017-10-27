
py_library(
    name = "batch_manager",
    srcs = ["dataset/batch_manager.py"]
)

py_library(
    name = "data_utils",
    srcs = ["dataset/data_utils.py"]
)

py_library(
    name = "conlleval",
    srcs = ["conlleval.py"]
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
    deps = [":data_utils"]
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [":conlleval"]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [":data_utils", ":utils"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":data_utils",
     ":batch_manager",
     ":loader",
     ":utils",
     ":model"]
)

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [":data_utils", ":utils"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":predict"]
)