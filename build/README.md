## Instructions

Docker image suitable for lightweight Tensorflow Object Detection model operations like exporting, inference etc. It's 
definitely not suitable for training since it doesn't utilize GPU. It also provides support for *.ipynb 
(Jupyter Notebook) editing.

### Build & run

Following commands are available at root directory of the project. Build image with:
```bash
$ make build-container
```

Run image with:
```bash
$ make run-container
```