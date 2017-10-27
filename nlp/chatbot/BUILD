
py_library(
    name = "data_utils",
    srcs = ["dataset/data_utils.py"]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [":data_utils"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":data_utils",
            ":model"
    ]
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