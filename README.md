## Summary

Most of the code in this repo comes from the [slambook2 repo](https://github.com/gaoxiang12/slambook2), which holds the codebase for the [The SLAM book](https://github.com/gaoxiang12/slambook-en).

What I'm doing is going through all the book and using the code to implement Visual Slam using the [Webots simulator](https://cyberbotics.com).

## To compile a Webots controller

Change directory into a folder named `build`, inside the controller directory:
```
cd controllers/first_controller/build
```

Run:
```
export WEBOTS_HOME=/usr/local/webots
cmake .. -G "Unix Makefiles" && make
```