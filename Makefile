help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

# Commands useful on host machine
build-container: ## Build testing Docker container without GPU support
	cd build; docker build -t nd013-c1-vision-starter -f Dockerfile .; cd ..

run-container: ## Run testing Docker container
	cd build; docker run -v ${PWD}:/app/project/ -p 8899:8888 --shm-size 2g -ti nd013-c1-vision-starter bash; cd ..

open-jupyter: ## Open running Jupyter Notebook in default browser
	open http://localhost:8899

tensorboard: ## Show experiment results in tensorboard
	tensorboard serve --logdir experiments/

# Commands useful in running Docker container
run-jupyter: ## Run Jupyter notebook in Docker container
	jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser
