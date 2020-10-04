# Self-Diving Car Nanodegree
## Dependencies
1. The commands for the setup are taylored to the fish shell (see [fish  shell github page](https://github.com/fish-shell/fish-shell) for installation)
2. Docker (see [official docker page](https://docs.docker.com/engine/install/ubuntu/) for installation)

## Project Setup
### Preperation
The environment for the first term can be set-up by:
```bash
docker pull udacity/carnd-term1-starter-kit
```
### Project Specific Setup
Download the .zip file from the repository, then unzip and navigate into the folde, 
```bash
set PROJECT_NAME <project-name>
unzip ~/Downloads/$PROJECT_NAME-master.zip -d (pwd)/
mv $PROJECT_NAME-master $PROJECT_NAME
```
and to start the jupyter notebook run:
```bash
# Fish shell command
cd (pwd)/$PROJECT_NAME
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v (pwd):/src udacity/carnd-term1-starter-kit 
```
and then enter the URL in your browser (remove the part before or after the 'or' and the round brackets).

### Example: Projekt 1
```bash
set PROJECT_NAME CarND-LaneLines-P1
unzip ~/Downloads/$PROJECT_NAME-master.zip -d (pwd)/
mv $PROJECT_NAME-master $PROJECT_NAME
cd (pwd)/$PROJECT_NAME
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v (pwd):/src udacity/carnd-term1-starter-kit 
```
