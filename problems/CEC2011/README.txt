Modify MATLAB_PATH variable in build.sh and run.sh files to point to the installation path of Matlab in your system.

Use build.sh script to build the shared file and the demo program.
Use run.sh script to run the demo program. It basically prepends the env variable LD_LIBRARY_PATH before the execution of the program. You can export this variable manually or add it to your .bashrc file.

When building, you can safely ignore the warning about the name of the library file. We rename it afterwards. This is needed to comply with the compilation process in Windows, as all the include statements refer to CEC2011.h.
