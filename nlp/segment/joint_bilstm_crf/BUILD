
py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],

)

py_library(
    name = "data_utils",
    srcs = ["dataset/data_utils.py"],
    deps = [":evaluation"]
)

py_library(
    name = "batch",
    srcs = ["batch.py"],
    deps = [":data_utils"]
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [":layers"]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [":data_utils", ":layers", ":losses", ":batch"]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [":data_utils",
     ":model"]
)

py_library(
    name = "test",
    srcs = ["test.py"],
    deps = [":data_utils", ":model"]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":train", ":test"]
)