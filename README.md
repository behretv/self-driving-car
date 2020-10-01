# Self-Diving Car Nanodegree
## Project 1
Download the .zip file from the repository, then unzip and navigate into the folde, 
```bash
unzip ~/Downloads/CarND-LaneLines-P1-master.zip -d (pwd)/
mv CarND-LaneLines-P1-master CarND-LaneLines-P1
cd (pwd)/carND-LaneLines-P1
```
Subsequnetly, the environment for the first project can be set-up by:
```bash
docker pull udacity/carnd-term1-starter-kit
```
and to start the jupyter notebook run:
```bash
# Fish shell command
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v (pwd):/src udacity/carnd-term1-starter-kit 
```
and then enter the URL in your browser.
