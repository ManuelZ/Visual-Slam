## Compile a controller

Change directory into a folder named `build`, inside the controller directory:
```
cd controllers/first_controller/build
```

Run:
```
export WEBOTS_HOME=/usr/local/webots
cmake .. -G "Unix Makefiles" && make
```