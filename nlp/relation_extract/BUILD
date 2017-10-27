py_library(
    name = "utils",
    srcs = ["dataset/utils.py"]
)

py_library(
    name = "model",
    srcs = ["model.py"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":utils",
     ":model",]
)

py_library(
    name = "test",
    srcs = ["test.py"],
    deps = [":model"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":test"]
)