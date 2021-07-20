import os

dirs = [
    os.path.join("data", "train"),
    os.path.join("data", "test"),
    os.path.join("static", "css"),
    os.path.join("static", "js"),
    "templates",
    "notebooks",
    "saved_models",
    "src",

]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = [
    "params.yaml",
    "dvc.yaml",
    ".gitignore",
    os.path.join("src", "__init__.py"),
    "app.py",
    "setup.py",
    "requirements.txt",
    "README.md"
]

for file_ in files:
    with open(file_, "w") as f:
        pass
