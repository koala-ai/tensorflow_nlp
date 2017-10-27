py_library(
    name = "poems",
    srcs = ["dataset/poems.py"]
)

py_library(
    name = "model",
    srcs = ["model.py"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":poems",
     ":model",]
)

py_library(
    name = "generate",
    srcs = ["generate.py"],
    deps = [":model", ":poems"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":generate"]
)