# Set-up
Using docker to set up the environemnt:
```bash
# Using the fish shell (pwd) is used to get the current path
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v (pwd):/src udacity/carnd-term1-starter-kit 
```
