
py_library(
    name = "model",
    srcs = ["model.py"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [":model", ":reader"]
)